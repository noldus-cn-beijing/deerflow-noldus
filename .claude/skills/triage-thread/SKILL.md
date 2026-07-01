---
name: triage-thread
description: 把 prod thread 的对话/workspace 全量产物/docker logs 执行轨迹从线上 ECS 拉回本机并定根因。仅供人工 /triage-thread {thread_id} 触发。
disable-model-invocation: true
argument-hint: "<thread_id> [问题描述]"
allowed-tools: [Bash, Read, Write, Grep, Glob]
version: 0.1.0
---

# triage-thread

拿到一个 prod `thread_id`，一条龙从线上 ECS 把该 thread 的**完整对话 + workspace 全量产物（plan/handoff/inputs/metrics）+ docker logs 执行轨迹**拉回本机，然后按用户描述的问题类型定位**根因**。

## Ground truth 铁律（先读，违反即判错结论）

**永远信落盘文件 + docker logs，不信 lead/subagent 的二手叙述。** 这是本 skill 的唯一心智模型，下面所有步骤都是它的展开。

- 2026-06-22 `d58adb40` 案例：lead 反复叙述「catalog 阻断」，**ground truth** 是 plan 早成功（`plan_charts.json` 的 `skipped[]` 为空），真凶是 chart-maker 15 turn 全耗在 `--help` 探测 + heredoc 被沙箱 block（docker logs 的 `SandboxAudit` 轨迹）。
- 任何「X 说了 Y」的叙述，都要去 **ground truth** 找物证：lead 说没出图 → 看 `plan_charts.json` skipped + outputs png；subagent 说失败了 → 看它的 `handoff_*.json` status + docker logs 真实命令。
- 找不到物证就**不下结论**，回报「叙述与落盘不符，需进一步查 X」。

## 前置（固定，每次跑前心里有数）

- **ECS**：`root@39.105.231.16`（本机已配免密 ssh；`settings.local.json` 已放行 `Bash(ssh * 39.105.231.16 *)`，子串匹配只限这台）。
- **容器**：`deer-flow-gateway`（compose 服务，agent runtime 内嵌）。
- **远端默认 shell 是 `sh` 不是 bash**，不认 `(`、`&&` 部分场景异常 → 内嵌 python **必须用 heredoc 或脚本文件**，别用层层转义的 `python -c "..."`。

### 两个已知坑（别再踩，踩了就走不通）

| 坑 | 现象 | 绕法 |
|---|---|---|
| **docker cp 坏了** | `Error response from daemon: mkdirat var/run: file exists` | 不用 docker cp。写入容器用 `docker exec -i ... sh -c 'cat > path' < localfile`；读出用 `docker exec -i ... cat file > local` |
| **内联 python 转义炸** | `sh: Syntax error: "(" unexpected` | 远端跑 python 走 `docker exec ... python /tmp/x.py`，脚本先写到宿主再灌进容器（见 Step 3 模板） |

### dump_thread.py 在不在镜像里（先自检一次）

`packages/agent/backend/scripts/dump_thread.py` 已 commit 到 dev，**下次 `make deploy-tar` 才进镜像**。

```bash
ssh root@39.105.231.16 'docker exec deer-flow-gateway ls /app/backend/scripts/dump_thread.py'
```

- 有输出 → 镜像里有，直接 Step 1 一键 dump。
- 报 `No such file` → 镜像里没有，按 Step 3 先把脚本灌进容器再跑。

## 执行流程

每一步带完成标志；标志没达成就停在该步诊断，别往下走。

### Step 0 — 解析参数

- `$1` = thread_id（必填，UUID 形态）。用户没给就停下问，**不要猜**。
- `$2`（可选）= 问题描述，如「没出图」「指标算错」「列对齐」。有则记下，决定 Step 5 走哪条排查路线；没有就默认走「没产出」通用路线。
- 本机落点目录：`~/triage-{TID}/`（先建好，所有拉回的文件都放这）。

```bash
TID=<贴 thread_id>
mkdir -p ~/triage-${TID}
```

**完成标志**：`TID` 非空、`~/triage-${TID}/` 存在。

### Step 1 — dump 完整对话（一键）

```bash
ssh root@39.105.231.16 \
  "docker exec deer-flow-gateway sh -c 'cd /app/backend && .venv/bin/python scripts/dump_thread.py $TID -o /tmp/${TID}.json && ls -la /tmp/${TID}.json'"
# 拉回本机
ssh root@39.105.231.16 "docker exec -i deer-flow-gateway cat /tmp/${TID}.json" > ~/triage-${TID}/thread.json
```

输出 JSON 顶层结构：`thread_id` / `checkpoint` / `metadata` / `values` / `messages`（完整对话，含 tool calls，**可能被截断**——别只靠它）/ `runs`（来自 app_db：status / token / first_human_message / last_ai_message / error）。

> dump_thread.py 复用 `make_checkpointer` + `serialize_channel_values`，走抽象层不碰表结构，兼容新版 langgraph saver。**不要自己解 msgpack blob。**
> 退出码 2 = 该 thread 在 checkpointer 里没有 checkpoint（thread_id 写错 / 还没写过）。先确认 TID 拼对。

**完成标志**：`~/triage-${TID}/thread.json` 存在、`jq '.messages | length'` > 0（或确认有 `runs` 元数据）。

### Step 2 — 查 uid（workspace 路径需要）

```bash
UID=$(ssh root@39.105.231.16 "docker exec deer-flow-gateway /app/backend/.venv/bin/python -c \"
import sqlite3
c=sqlite3.connect('/app/backend/.deer-flow/data/deerflow.db')
r=c.execute('SELECT user_id FROM threads_meta WHERE thread_id=?', ('$TID',)).fetchone()
print(r[0] if r else 'NOT_FOUND')
\"")
echo "uid=$UID"
```

拿到 uid 后 workspace 路径 = `/app/backend/.deer-flow/users/{UID}/threads/{TID}/user-data/workspace`。

> 若 uid 是 `NOT_FOUND`：该 thread 没写过元数据（极早期失败 / 纯知识问答没起 workspace）。Step 4 跳过，靠 Step 1 的 `messages` + Step 3 的 logs 定位。

**完成标志**：`uid` 是合法 UUID（或已确认 NOT_FOUND 且记录原因）。

### Step 3 — 列 workspace 全文件 + 拉 docker logs 执行轨迹

**两条并行拉，缺一不可**——workspace 是产物真相，logs 是执行真相。

#### 3a. 列某 thread workspace 全文件（按时间，看最近发生了什么）

```bash
ssh root@39.105.231.16 "docker exec deer-flow-gateway /app/backend/.venv/bin/python -c \"
import glob, os, datetime
BASE='/app/backend/.deer-flow/users/${UID}/threads/${TID}/user-data/workspace'
for f in sorted(glob.glob(BASE+'/*'), key=os.path.getmtime):
    t=datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime('%m-%d %H:%M')
    print(t, os.path.getsize(f), os.path.basename(f))
\""
```

#### 3b. docker logs（subagent 真实命令 + 卡点；对话 tool_result 常被截断，**必看这个**）

```bash
# 该 thread 所有 sandbox 命令（chart-maker 等真实跑了什么）
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep SandboxAudit" | head -60 > ~/triage-${TID}/logs_sandbox.txt

# 错误 / 异常 / max_turns 提前终止
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep -iE 'max_turns|Traceback|Error:|verdict.: .block|missing|NoZones'" | head -40 > ~/triage-${TID}/logs_errors.txt

# 某 subagent 的完整 turn 轨迹
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep -E 'Subagent|captured AI message|max_turns'" | head -40 > ~/triage-${TID}/logs_turns.txt
```

> `--since 72h` 按需调（出问题距现在多久）。`grep SandboxAudit` 每行的 `command` 字段 = subagent 实际执行的 bash，`verdict` = `pass`/`block`。

**完成标志**：workspace 文件清单已拿到 + 至少一个 logs 文件非空（或确认该 thread 在 72h 内无日志、需要拉长时间窗）。

### Step 4 — 按「关键文件」深挖（拉回本机读）

根据 Step 3a 的文件清单，**只拉回与问题相关的** `handoff_*.json` / `plan_*.json` / `experiment-context.json`，别一股脑全拉（898MB checkpoints.db 在隔壁，别碰）。模板：

```bash
# 单文件拉回（docker cp 坏了，走 cat）
ssh root@39.105.231.16 "docker exec -i deer-flow-gateway cat /app/backend/.deer-flow/users/${UID}/threads/${TID}/user-data/workspace/<FILE>" > ~/triage-${TID}/<FILE>
```

**workspace 里值得看的关键文件**（按问题类型选读，详见 Step 5）：

| 文件 | 看什么 |
|---|---|
| `experiment-context.json` | `column_aliases`（列对齐）、`parameter_overrides`、`paradigm`、`ev19_template` |
| `plan_metrics.json` / `plan_charts.json` | plan 的 `skipped[]`（**别信 lead 转述，看这里**）、`parameters_used` |
| `handoff_code_executor.json` | `per_subject`（算好的指标值）、`data_quality_warnings`、`sealed_by` |
| `handoff_chart_maker.json` / `handoff_data_analyst.json` / `handoff_report_writer.json` | 各 subagent 的 `status` / `failed_charts` / `failed reason` / 错误 |
| `m_*_s*.json` / `inputs_*.json` | 逐 subject 中间产物 |
| `groups.json` | 分组映射 |

> 想在容器内跑自定义探查脚本（避开转义坑，复制即用）：
> ```bash
> cat > /tmp/inspect.py << 'PYEOF'
> import json, os, glob
> BASE="/app/backend/.deer-flow/users/{UID}/threads/{TID}/user-data/workspace"
> # ... 你的探查代码 ...
> PYEOF
> scp /tmp/inspect.py root@39.105.231.16:/tmp/inspect.py
> ssh root@39.105.231.16 'docker exec -i deer-flow-gateway sh -c "cat > /tmp/inspect.py" < /tmp/inspect.py'
> ssh root@39.105.231.16 'docker exec deer-flow-gateway /app/backend/.venv/bin/python /tmp/inspect.py'
> ```
> 想灌 dump_thread.py 进容器（镜像里没有时）：把 inspect.py 换成 dump_thread.py，目标路径 `/app/backend/scripts/dump_thread.py`。

**完成标志**：与问题相关的关键文件都已拉回 `~/triage-${TID}/` 且 `jq` 能解析（JSON 合法）。

### Step 5 — 按问题类型定根因

把用户描述（Step 0 的 `$2`，或从 Step 1 的 `messages` 末尾推断用户在抱怨什么）套到下面路线。**每条路线都要求：先在 ground truth 找物证，再下结论。**

#### 路线 A —「某 subagent 没产出 / 没出图 / 没出报告」（默认通用路线）

1. Step 1 dump → 看对应 tool/task 的结果（被截断也没关系，先定位是哪个 subagent）。
2. Step 4 读该 subagent 的 `handoff_*.json` → `status`（`completed`/`partial`/`failed`/`sealed`?）/ `failed_*`。
3. Step 3b 拉该 thread 的 `SandboxAudit` + `max_turns` → 看它**真实跑了什么命令、是不是耗尽 turn / 被沙箱 block / 报错**。
4. **判据**：lead 说「X 没干」但 `handoff_*.json` 有产出 + `sealed_by` 非空 → 是**展示/前端**问题不是执行问题；`handoff` 缺失 + logs 有 `max_turns` → turn 耗尽；logs 有 `verdict: block` → 沙箱拦截（看 block 的命令是什么）。

#### 路线 B —「指标算错了 / 参数不对」

1. Step 4 读 `handoff_code_executor.json` 的 `per_subject`（逐 subject 的指标值）+ `data_quality_warnings`（`code=METRIC_VALIDATION` 的范围违规）。
2. Step 4 读 `plan_metrics.json` 的 `parameters_used`（zone 分组对不对、output_unit）。
3. 对照 `experiment-context.json` 的 `column_aliases`（列对齐源头）。
4. **判据**：值 NaN/Inf 或越界 → `data_quality_warnings` 会记；参数错 → 比对 `parameters_used` vs 用户数据实际列。

#### 路线 C —「列对齐 / zone 识别问题」

1. Step 4 读 `experiment-context.json` 的 `column_aliases`（有没有？对不对？没对齐就是 FewZones 走列对齐、缺归属列才是 NoZones）。
2. Step 4 读 `plan_charts.json` 的 `skipped[]`（真 skip 了才信执行层无图，没 skip 就是执行层问题）。
3. 必要时核代码：`packages/ethoinsight/ethoinsight/catalog/resolve.py` 的 `resolve_charts` / `_build_zone_aliases_overrides` / `_missing_columns`。
4. **判据**：`column_aliases` 缺 → 反问没问到位；`skipped[]` 非空且 reason 是列缺失 → 对齐层拒；`skipped[]` 空但无图 → 执行层（走路线 A 第 3 步）。

#### 路线 D —「seal 漏调 / subagent 卡住没结束」

1. Step 4 各 `handoff_*.json` 的 `sealed_by` 字段（缺 = 漏调 seal）。
2. Step 3b 看 subagent 是否在 ReAct 无 tool_call 即结束（seal 漏调家族根因）。
3. **判据**：有产出无 seal → harness auto-seal 兜底是否触发（`sealed_by=run_plan`/`_attempt_auto_seal_from_artifacts`）；认知产物（code-executor/data-analyst）永不被 auto-seal，漏调即真失败。

#### 路线 E —「流式中断 / 前端 404 / 性能」

1. Step 3b 看 thread 是否有 `Traceback` / timeout / event-loop 阻塞。
2. Step 4 读 `handoff_chart_maker.json` 的产物路径，对照前端请求的 artifact 路径（`present_files` 拒绝磁盘不存在的文件 → 防幻影文件名 404）。
3. 性能类不在本 skill 范围（用 `/noldus-insight-e2e` 的 perf panel，且必须 prod build）。

### Step 6 — 交叉比对 + 回报根因

- **交叉比对**：找另一个**同数据成功**的 thread 对照（如 `d58adb40` 失败时比 `62e20ed0` 成功），差异点常即根因。用户没给对照 thread 就跳过这步，但提醒用户「若有同数据成功的 thread，给 ID 可交叉验证」。
- **回报格式**（写进 `~/triage-${TID}/triage-report.md` 并贴给用户）：
  1. **现象**（用户描述 + Step 1 messages 末尾确认的现象）。
  2. **ground truth 物证**（哪个文件 / 哪行 log 支撑结论，贴关键片段）。
  3. **根因**（一句话，能追溯到物证）。
  4. **推翻的叙述**（lead/subagent 说了什么、为何不信，体现 ground truth 铁律）。
  5. **建议下一步**（修代码 / 改 prompt / 加结构门——但**先证旧路不可救再叠兜底**，见下纪律）。
- 拉回的数据含真实用户内容（对话、文件路径），**别外发**，只在本机分析。
- 容器里留下的 `/tmp/*.json` / `/tmp/inspect*.py` 无害（只读），排查完可顺手清：`ssh root@39.105.231.16 'docker exec deer-flow-gateway rm -f /tmp/<name>'`。

## 完成标志（全部满足才算 triage 完成）

- `~/triage-${TID}/thread.json` 存在且可解析。
- 至少一条 docker logs 轨迹（`logs_sandbox.txt` / `logs_errors.txt` / `logs_turns.txt` 之一）已拉回。
- 与问题相关的关键 workspace 文件已拉回（或确认 workspace 不存在并记录原因）。
- `triage-report.md` 写好，根因有 ground truth 物证支撑（不是「我觉得」）。

## 数据落点地图（深入时查表）

容器内路径（`working_dir=/app/backend`，`DEER_FLOW_HOME=/app/backend/.deer-flow`）：

| 数据 | 路径 | 说明 |
|---|---|---|
| **对话/checkpoint DB** | `/app/backend/.deer-flow/checkpoints.db`（活文件、大） | 新版 langgraph saver，表名 `checkpoints/writes/store_migrations/store`（**非**默认 `checkpoint_blobs`）。**别自己解 msgpack**，走 dump_thread.py |
| **应用元数据 DB** | `/app/backend/.deer-flow/data/deerflow.db` | `threads_meta/runs/run_events/users/...`。**`run_events.backend: memory` 不落盘**，查不到对话，只查元数据 |
| **thread workspace**（用户隔离） | `/app/backend/.deer-flow/users/{uid}/threads/{tid}/user-data/workspace/` | uid 先查（Step 2） |
| `/mnt/user-data/*` | = workspace 的 `user-data/` | sandbox 内见的虚拟路径 |
