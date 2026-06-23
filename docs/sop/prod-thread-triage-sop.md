# 生产 ECS 按 thread_id 定位 + 拉回数据 SOP

> **面向执行的 Claude Code agent**。目标：拿到一个 prod thread_id 后，一条龙从线上 ECS 把该 thread 的**完整对话 + workspace 全量产物（plan/handoff/inputs/metrics）+ docker logs 执行轨迹**拉回本机，供离线分析根因。
>
> **为什么存在**：prod 的 thread 数据在 ECS 容器内，落点、传输方式、shell 转义都有坑；每次排查重新摸索成本高。本 SOP 把 2026-06-22 排查 `d58adb40` 验证过的链路固化下来。

---

## 0. 前置（只查一次，后续跳过）

### 0.1 ECS 与容器（固定）

- **ECS**：`root@39.105.231.16`（本机已配免密 ssh）
- **容器**：`deer-flow-gateway`（compose 服务）
- **远端默认 shell 是 `sh` 不是 bash**，不认 `(`、`&&` 部分场景异常 → 内嵌 python **必须用 heredoc 或脚本文件**，别用层层转义的 `python -c "..."`。

### 0.2 权限规则（settings.local.json，已配）

`.claude/settings.local.json` 的 `permissions.allow` 里已放行：

```
Bash(ssh * 39.105.231.16 *)
```

子串匹配，限这台 ECS。**仅本机、个人、gitignore**。新 agent 在本仓库跑 SOP 不会被反复拦。

### 0.3 两个已知坑（别再踩）

| 坑 | 现象 | 绕法 |
|---|---|---|
| **docker cp 坏了** | `Error response from daemon: mkdirat var/run: file exists` | 不用 docker cp。写入容器用 `docker exec -i ... sh -c 'cat > path' < localfile`；读出用 `docker exec -i ... cat file > local` |
| **内联 python 转义炸** | `sh: Syntax error: "(" unexpected` | 远端跑 python 走 `docker exec ... python /tmp/x.py`，脚本先写到宿主再灌进容器（见 §2.2 模板） |

### 0.4 dump_thread.py 必须在镜像里

[packages/agent/backend/scripts/dump_thread.py](../../packages/agent/backend/scripts/dump_thread.py) 已 commit 到 dev，**下次 `make deploy-tar` 才进镜像**。

- **镜像里有**（部署过 dev 最新版后）：直接 §1 一键 dump。
- **镜像里没有**（老部署）：按 §2.2 先把脚本灌进容器再跑。
- 自检：`ssh root@39.105.231.16 'docker exec deer-flow-gateway ls /app/backend/scripts/dump_thread.py'`，有输出=在镜像里。

---

## 1. 一键 dump 指定 thread 的完整对话

```bash
TID=<贴 thread_id>
ssh root@39.105.231.16 \
  "docker exec deer-flow-gateway sh -c 'cd /app/backend && .venv/bin/python scripts/dump_thread.py $TID -o /tmp/${TID}.json && ls -la /tmp/${TID}.json'"
# 拉回本机
ssh root@39.105.231.16 "docker exec -i deer-flow-gateway cat /tmp/${TID}.json" > ~/${TID}.json
```

输出 JSON 顶层结构：`thread_id` / `checkpoint` / `metadata` / `messages`（完整对话，含 tool calls）/ `app_db` / `thread_meta` / `runs`。

> dump_thread.py 复用 `make_checkpointer` + `serialize_channel_values`，走抽象层不碰表结构，兼容新版 langgraph saver。**不要自己解 msgpack blob**。

---

## 2. 数据落点地图（按需深入时用）

### 2.1 容器内路径（`working_dir=/app/backend`，`DEER_FLOW_HOME=/app/backend/.deer-flow`）

| 数据 | 路径 | 说明 |
|---|---|---|
| **对话/checkpoint DB** | `/app/backend/.deer-flow/checkpoints.db`（898MB、活文件） | 新版 langgraph saver，表名 `checkpoints/writes/store_migrations/store`（**非**默认 `checkpoint_blobs`）。注意是 `.deer-flow/checkpoints.db` 不是 `/app/backend/checkpoints.db` |
| **应用元数据 DB** | `/app/backend/.deer-flow/data/deerflow.db` | `threads_meta/runs/run_events/users/...`。**`run_events.backend: memory` 不落盘**，查不到对话，只查元数据 |
| **thread workspace**（用户隔离） | `/app/backend/.deer-flow/users/{uid}/threads/{tid}/user-data/workspace/` | uid 要先查（见 §2.3） |
| `/mnt/user-data/*` | = workspace 的 `user-data/` | sandbox 内见的路径 |

### 2.2 在容器内跑自定义 python（避开转义坑，复制即用）

```bash
cat > /tmp/inspect.py << 'PYEOF'
import json, os, glob
BASE="/app/backend/.deer-flow/users/ef64773c-7309-4ff4-b10a-4ffb63dee9aa/threads/d58adb40-0d84-44c8-bb36-43f1b76ac059/user-data/workspace"
# ... 你的探查代码 ...
PYEOF
# 灌进容器（docker cp 坏了，走 stdin）
scp /tmp/inspect.py root@39.105.231.16:/tmp/inspect.py
ssh root@39.105.231.16 'docker exec -i deer-flow-gateway sh -c "cat > /tmp/inspect.py" < /tmp/inspect.py'
# 跑
ssh root@39.105.231.16 'docker exec deer-flow-gateway /app/backend/.venv/bin/python /tmp/inspect.py'
```

> 想灌 dump_thread.py 进容器（镜像里没有时）：把上面 inspect.py 换成 dump_thread.py，目标路径 `/app/backend/scripts/dump_thread.py`。

### 2.3 从 thread_id 查 uid（workspace 路径需要）

```bash
TID=<thread_id>
ssh root@39.105.231.16 "docker exec deer-flow-gateway /app/backend/.venv/bin/python -c \"
import sqlite3
c=sqlite3.connect('/app/backend/.deer-flow/data/deerflow.db')
r=c.execute('SELECT user_id FROM threads_meta WHERE thread_id=?', ('$TID',)).fetchone()
print(r[0] if r else 'NOT_FOUND')
\""
```

拿到 uid 后拼 workspace 路径：`/app/backend/.deer-flow/users/{uid}/threads/{TID}/user-data/workspace`。

### 2.4 workspace 里值得看的关键文件

| 文件 | 看什么 |
|---|---|
| `experiment-context.json` | `column_aliases`（列对齐）、`parameter_overrides`、`paradigm` |
| `plan_metrics.json` / `plan_charts.json` | plan 的 `skipped[]`（**别信 lead 转述，看这里**）、`parameters_used` |
| `handoff_code_executor.json` | `per_subject`（算好的指标值）、`data_quality_warnings`、`sealed_by` |
| `handoff_chart_maker.json` / `handoff_data_analyst.json` / `handoff_report_writer.json` | 各 subagent 的 `failed_charts` / `status` / 错误 |
| `m_*_s*.json` / `inputs_*.json` | 逐 subject 中间产物 |
| `groups.json` | 分组映射 |

```bash
# 列某 thread workspace 全文件（按时间，看最近发生了什么）
ssh root@39.105.231.16 "docker exec deer-flow-gateway /app/backend/.venv/bin/python -c \"
import glob, os, datetime
BASE='/app/backend/.deer-flow/users/{UID}/threads/{TID}/user-data/workspace'
for f in sorted(glob.glob(BASE+'/*'), key=os.path.getmtime):
    t=datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime('%m-%d %H:%M')
    print(t, os.path.getsize(f), os.path.basename(f))
\""
```

---

## 3. 拉 docker logs 执行轨迹（定位 subagent 真实命令）

**判断 chart/code-executor/data-analyst 等 subagent 实际干了什么、卡在哪步，看这个，别只看对话里的 tool_result（常被截断）。**

```bash
TID=<thread_id>
# 该 thread 的所有 sandbox 命令（chart-maker 真实跑了什么）
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep SandboxAudit" | head -60

# 该 thread 的错误 / 异常 / max_turns 提前终止
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep -iE 'max_turns|Traceback|Error:|verdict.: .block|missing|NoZones'" | head -40

# 某 subagent 的完整 turn 轨迹（按 trace id）
ssh root@39.105.231.16 \
  "docker logs --since 72h deer-flow-gateway 2>&1 | grep '$TID' | grep -E 'Subagent|captured AI message|max_turns'"
```

> `--since 72h` 按需调。`grep SandboxAudit` 的每行 `command` 字段就是 subagent 实际执行的 bash 命令，`verdict` 是 `pass`/`block`。

---

## 4. 典型排查模式（按问题类型套用）

### 4.1 「某 subagent 没产出 / 没出图 / 没出报告」

1. §1 dump 对话 → 看对应 tool/task 的结果（被截断也没关系）。
2. §2.4 读该 subagent 的 `handoff_*.json` → 看 `status` / `failed_*`。
3. §3 拉该 thread 的 `SandboxAudit` + `max_turns` 轨迹 → 看它**真实跑了什么命令、是不是耗尽 turn / 被沙箱 block / 报错**。
4. **铁律：永远信落盘文件 + docker logs，不信 lead/subagent 的二手叙述。**（2026-06-22 案例：lead 反复说「catalog 阻断」，实际 plan 早成功，真凶是 chart-maker turn 耗尽。）

### 4.2 「指标算错了 / 参数不对」

1. §2.4 读 `handoff_code_executor.json` 的 `per_subject` + `data_quality_warnings`。
2. §2.4 读 `plan_metrics.json` 的 `parameters_used`（看 zone 分组对不对）。
3. 对照 `experiment-context.json` 的 `column_aliases`（列对齐源头）。

### 4.3 「列对齐 / zone 识别问题」

1. §2.4 读 `experiment-context.json` 的 `column_aliases`（有没有？对不对？）。
2. §2.4 读 `plan_charts.json` 的 `skipped[]`（真 skip 了才信，没 skip 就是执行层问题）。
3. 列对齐代码：[packages/ethoinsight/ethoinsight/catalog/resolve.py](../../packages/agent/backend/packages/ethoinsight/ethoinsight/catalog/resolve.py) 的 `resolve_charts` / `_build_zone_aliases_overrides` / `_missing_columns`。

---

## 5. 守工程纪律（排查时别造新坑）

- **根因未隔离前别叠兜底机制**（memory: `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）——先证旧路不可救。
- **判断 chart/seal/列对齐问题，先 grep 同事的 review-packages**（SSOT 在那边，memory: `feedback_ssot_lives_in_review_packages`），别提议 SSOT 没列的。
- 拉回的数据含真实用户内容（对话、文件路径），**别外发**，只在本机分析。
- 容器里留下的 `/tmp/*.json` / `/tmp/inspect*.py` 无害（只读），排查完可顺手清：`ssh root@39.105.231.16 'docker exec deer-flow-gateway rm -f /tmp/<name>'`。

---

## 6. 完整示例（2026-06-22 d58adb40 复盘）

参见 [docs/handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md](../handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md) —— 就是按本 SOP 的链路排查的：dump 对话 → 读 plan_charts.json（发现 0 skip，推翻 lead 叙述）→ 读 handoff_chart_maker.json（6/16 旧失败原因）→ 挖 docker logs（发现 chart-maker 15 turn 全在 `--help` 探测 + heredoc 被 block）→ 交叉比对另一个 thread `62e20ed0`（同数据成功出图）→ 锁定真凶。
