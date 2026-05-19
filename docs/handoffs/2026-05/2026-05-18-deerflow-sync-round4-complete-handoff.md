# 2026-05-18 DeerFlow Sync 第 4 轮完成交接（11 commit）

> **状态**：第 3 轮 4 个 PENDING + 第 4 轮上游 6 个 delta 全部已合并并推送 `origin/dev`。本仓库当前已经吃下 `deerflow-noldus/main` (= `bytedance/deer-flow upstream-real/main`) HEAD `39f901d3` 的全部**业务可合**改动。
>
> **预读**：本文件 + [2026-05-18 deerflow-sync-pending-surgical-merge-handoff.md](2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md)（前置计划）+ [2026-05-18 deerflow-sync-progress.md](2026-05-18-deerflow-sync-progress.md)（更早一轮记录）

**IMPORTANT**：commit history 在推送前经过一次 force-push 修正（因为 6d3cffb4 初版是空 commit，后改为真正的 surgical merge 导致需要 amend 链接上游 commit）。如果在新会话中看到远端 commit hashes 与本文件不符，读取 `git log --oneline 7981af52..dev | tail -15` 即可定位。

---

## 1. 任务目标

把 noldus-insight 仓库的 deerflow agent 架构跟上上游 `deerflow-noldus/main`，**保留所有 Noldus 定制**（handoff 占位符、shared 占位符、guardrail provider、recursion_limit、AI msg 硬限、copy_context user_id 传递、extra_env / DEERFLOW_PATH_*）。

**预期产出**：
- 第 3 轮 PENDING 的 4 个 commit（A-8/A-9/A-10/A-11）surgical merge
- 评估并合入上游剩余的 8 个 delta commit（6 个合入 + 2 个跳过）
- 保持现有测试通过，不引入新失败

---

## 2. 当前进展（已完成）

✅ **第 3 轮 PENDING 4 个 commit 全部合入**（推到 `origin/dev`）：

| 本地 commit | 上游 commit | 摘要 |
|---|---|---|
| `cc2bb889` | A-11 `7de9b582` (#2774) | Runtime 类型 alias（替换 13 处 ToolRuntime[ContextT, ThreadState]） |
| `32b49c17` | A-8 `9892a7d4` (#2838) | token bucketing 基础设施（journal.py + token_collector.py + tests）|
| `5ddb6df2` | A-10 `813d3c94` 局部 (#2701) | subagent config 对齐（SubagentConfig.system_prompt → str \| None + resolve_subagent_model_name） |
| `58abb0b0` | A-9 `eab7ae3d` (#2882) + A-8/A-10 残余 | subagent token usage 流式到 header + executor/task_tool 三者交织改动 |

✅ **第 4 轮 6 个上游 delta 全部处理完毕**：

| 本地 commit | 上游 commit | 摘要 | 处理 |
|---|---|---|---|
| `f78b520b` | `181d8365` (#2939) | tool result 邻接性 normalize | 空 commit 标记（本地与上游最终态 0 diff，已等同合入） |
| `47094ba9` | `6e8e6a96` (#2924) | blocking IO detector | 整文件合（4 新文件 + conftest.py surgical merge 93 行）|
| `60f280bd` | `39f901d3` (#2989) | runs 持久化恢复 | 空 commit 跳过（本地 RunStore 未接入，合无意义） |
| `871e913a` | `380255f7` (#2873/#2881) | sandbox /mnt/user-data 契约 per-thread | 整文件 cp local_sandbox_provider.py（local 与上游父版本 0 diff）+ surgical merge tools.py 的 is_local_sandbox 1 处 + 新增 contract test |
| `ce560629` | `6d3cffb4` (#2958) | frontend 消息去重 | 取上游 978 行 hooks.ts baseline，回填 Noldus normalizeStoredRunId/getRunMetadataStorage/useThreadStream（archived-messages+cachedMessages+数组返回），**真正的 surgical merge 463 行 delta** |
| `31275138` | `7c42ab3e` (#2940) | frontend async chat submit 等待清空 | surgical merge 2 个 page.tsx + 1 个 input-box.tsx（不动 ai-elements/prompt-input.tsx，本地 CLAUDE.md 禁止改 registry 生成代码） |

✅ **明确跳过的 2 个**（按用户指令）：
- `48e038f7` Discord 增强 — 业务无关
- `ba864112` langsmith bump — 本地 langsmith 0.6.4 远落后上游 0.7.36，uv.lock 已经分叉太远

✅ **验证状态**：
- backend `make test`：2564/2595 通过，31 个 pre-existing 失败（stash 验证过）
- backend `make lint`：仅 3 个 pre-existing E501（code_executor.py prompt 长行，与本次 sync 无关）
- frontend `pnpm typecheck`：通过

---

## 3. 关键上下文

### 仓库结构

- 主分支：`dev`（当前 = `origin/dev` = `d2c2889f`）
- DeerFlow 上游远端：
  - `deerflow` → `git@github.com:noldus-cn-beijing/deerflow-noldus.git`（镜像）
  - `upstream-real` → `https://github.com/bytedance/deer-flow.git`（真上游）
  - 两者 HEAD 都是 `39f901d3`（fetch 时间 2026-05-18）

### 本仓库 fork 策略要点（CLAUDE.md 第 9 条 + sync SOP）

**受保护文件**（绝不整文件 cp）：
- `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`（797 → 现在 ~870 行）
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（314 → 现在 ~440 行）
- `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`（1658 → 现在 ~1660 行）
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`
- `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py`（注意：local_sandbox**_provider**.py 本轮已对齐上游，但 local_sandbox.py 仍含 Noldus 定制 extra_env）

**Noldus 定制清单（**必须保留**）：
- `{{handoff://<name>}}` 占位符系统 + `HANDOFF_FILE_REGISTRY` （task_tool.py）
- `{{shared://}}` → `/mnt/shared/` 替换（task_tool.py + sandbox/tools.py）
- `HandoffIsolationProvider` + `ScriptInvocationOnlyProvider` + `GuardrailMiddleware` （executor.py）
- `authorized_handoff_paths` 传递链（task_tool → executor → provider）
- `recursion_limit = max_turns * 2 + 1`（executor.py）
- AI msg 计数硬限 max_turns（executor.py）
- `copy_context()` user_id 传递到 isolated loop（executor.py）
- `max_turns: int | None` 参数（task_tool.py）
- `extra_env` 参数 + `DEERFLOW_PATH_*` 环境变量（local_sandbox.py）

### 本轮新建/重要文件

- `packages/agent/backend/packages/harness/deerflow/subagents/token_collector.py`（A-8 新建）
- `packages/agent/backend/packages/harness/deerflow/tools/types.py`（A-11 新建：`Runtime = ToolRuntime[dict[str, Any], ThreadState]`）
- `packages/agent/backend/tests/support/detectors/blocking_io.py`（6e8e6a96 新建）
- `packages/agent/backend/tests/test_local_sandbox_virtual_path_contract.py`（380255f7 新增，21 用例）
- 5 个新测试文件 + 1 个 cp 的（test_task_tool_core_logic.py）

### Pre-existing 失败 31 个（与本次 sync 无关）

按模块归类：
- `test_auth*` / `test_auth_config.py` / `test_auth_type_system.py`：JWT 环境变量 / `get_effective_user_id` 缺失
- `test_provisioner_pvc_volumes.py`：k8s provisioner pvc subPath
- `test_run_event_store.py` / `test_run_repository.py`：DB schema / asyncio 标记
- `test_aio_sandbox_provider.py`：`get_effective_user_id` 缺失（导致 2 个测试）
- `test_tool_deduplication.py`：sync wrapper 包装逻辑
- `test_ethoinsight_code_skill.py`：`max_turns == 12` 期望与现实 20 不符（code_executor 配置 drift）
- `test_ethoinsight_planning_skill.py`：`extensions_config.json` 没注册 ethoinsight-planning

**stash 验证**：把本次所有 sync 改动 stash 掉后这些测试仍然失败，证实 pre-sync 即失败。

---

## 4. 关键发现

### 4.1 handoff 文档与现实的偏差（已通过 review 修正）

第 3 轮 PENDING handoff 文档（`2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md`）有 3 处描述偏差，本次 review 全部识别并按"先按上游、再加 Noldus 独有"的用户原则处理：

1. **A-9 token_usage_middleware.py**：handoff 说"本地无 Noldus 定制 → 整文件 cp"。实际本地是 37 行 stub（4 月起一直没合上游 attribution 系统），上游 358 行带 step attribution 体系。整文件 cp 完成，本地 lead_agent 现已挂载完整 attribution。
2. **A-10 executor.py**：handoff 说"本地的 `_build_initial_state` 和上游已经接近"。实际本地用 `_build_system_prompt` 字符串拼接（system_prompt + skill 合并成字符串传给 `create_agent(system_prompt=...)`），上游用单 SystemMessage 注入 state。已重构对齐上游。
3. **A-8/A-9 task_tool.py 缺函数**：本地在第 3 轮 sync 中**主动删了** `_get_runtime_app_config`、`_merge_skill_allowlists`、`resolve_subagent_model_name`、`tool_groups` 等上游路径。本次按用户指令把 task_tool.py 重写为上游 A-9 baseline（422 行）后回填 Noldus 定制。同时给 `get_subagent_config` / `get_available_subagent_names` / `get_available_tools` / `SubagentExecutor.__init__` 加了 `*, app_config=None` keyword 兼容（local 端忽略，仅签名匹配）。

### 4.2 本次发现的本地架构事实

- **本地 frontend hooks.ts 已通过 surgical merge 对齐上游**：6d3cffb4（消息去重）初版标记为跳过，但在用户 push 后升级为真正的 surgical merge——取上游 978 行 baseline，回填 Noldus 的 `normalizeStoredRunId` / `getRunMetadataStorage` + `useThreadStream` 本地定制（archived-messages API + `cachedMessages` Map + 数组返回格式）。现在 hooks.ts 为 1079 行，pnpm typecheck 通过。**不兼容点**：`useThreadStream` 用 `/api/threads/{id}/archived-messages` 加载历史（上游用 per-run `/runs/{run_id}/messages`），两种路径不互通但可共存。
- **本地 RunManager 未接入 RunStore 持久化**：`RunManager()` 调用方（仅 `app/gateway/deps.py:93`）没传 store，构造函数也不接受 store 参数。CLAUDE.md 第 13 条说本仓库已吃下 Tier 4，但 runs 持久化部分还**没**完成。上游 #2989 修的是 RunStore fallback，本地合无意义。
- **本地 langsmith 0.6.4**：远落后于上游 0.7.36（上游 ba864112 bump 到 0.8.0）。uv.lock 跟上游差 4681 行——已经是独立分叉，不能用 commit 合并方式升级。要升 langsmith 必须单独 `uv add langsmith==X.Y.Z`。
- **本地没 update_agent_tool.py**：上游 A-11 测试覆盖 13 个工具含 update_agent，本地仅 12 个（删掉了 update_agent 用例）。

### 4.3 sandbox local_sandbox_provider 升级影响面

380255f7 把 LocalSandboxProvider 从单例 + 无 PathMapping 改成 per-thread + OrderedDict LRU + PathMapping(/mnt/user-data + 子目录 + /mnt/acp-workspace)。sandbox id 现在有两种格式：`"local"`（legacy，acquire() 无 thread_id）和 `"local:{thread_id}"`（per-thread）。`is_local_sandbox()` 已更新支持两种格式。

**潜在影响**（**待 dogfood 验证**）：
- 任何检查 `sandbox_id == "local"` 的代码会漏掉 per-thread 实例 → 已全文搜过，仅 `is_local_sandbox()` 一处
- `_agent_written_paths` 现在 per-sandbox（每个 thread 各自），不再 process-wide。如果有跨 thread 共享 written paths 假设的代码会断 → 没找到这种代码
- 内存：OrderedDict LRU cap 256 个 thread，超出会自动驱逐

---

## 5. 未完成事项

### 高优先级（必做）

1. **端到端 dogfood 验证** — 跑 `cd packages/agent && make dev`，上传 EPM 数据，验证：
   - 4 个 ethoinsight subagent 都能跑（code-executor、data-analyst、report-writer、knowledge-assistant）
   - `{{handoff://}}` 占位符正常工作（HandoffIsolationProvider 仍然拦截非授权 read_file）
   - `{{shared://}}` 占位符替换正常
   - **token bucketing 实际生效**：在 RunRow 上看到 `subagent_tokens > 0`（A-8 的核心收益）
   - **subagent token 流到 header**：前端 header 显示的 token usage 包含 subagent 的（A-9 的核心收益）
   - **per-thread sandbox 不影响现有路径**（380255f7 的回归点）

2. **frontend 含附件提交测试** — 7c42ab3e 的修复（含附件时返回 sendPromise）需要手动验证：传一个大文件，确认输入框等附件上传完才清空。本地无 e2e 测试目录，只能手测。

### 中优先级

3. **`make lint` 残留 3 个 E501** — `code_executor.py` prompt 长字符串超过 240 字符。和本次 sync 无关，但要清干净的话单独修。

4. **检查 31 个 pre-existing 失败** — 跟本轮 sync 无关，但建议跟相关 owner 同步：
   - `get_effective_user_id` 缺失（aio_sandbox + auth_type_system）→ Tier 4 不完整
   - `test_ethoinsight_code_skill.py` 期望 `max_turns == 12`，现实是 20 → 配置 drift
   - `test_ethoinsight_planning_skill.py` extensions_config.json 没注册 → 配置 drift

### 低优先级 / 不要做

5. **不要**尝试合 6d3cffb4 frontend message dedupe — 架构不兼容
6. **不要**尝试合 39f901d3 runs persistence restore — 需要先做 RunStore 接入大改造
7. **不要**手改 `frontend/src/components/ai-elements/prompt-input.tsx` — CLAUDE.md 明文说是 Vercel AI SDK registry 生成代码

---

## 6. 建议接手路径

### 新会话第一步

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline 7981af52..HEAD | head -12   # 看本轮 sync 的 10 个 commit
git status                                     # 应该是 clean（仅 gateway/.env 等 pre-existing 未提交改动）
```

### 关键文件速查

- 本交接文档：`docs/handoffs/2026-05/2026-05-18-deerflow-sync-round4-complete-handoff.md`
- 前置 handoff（描述 4 个 PENDING）：`docs/handoffs/2026-05/2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md`
- 仓库根 CLAUDE.md：项目定位 + DeerFlow 同步策略
- backend CLAUDE.md：harness/app 分层 + 受保护文件清单

### dogfood 命令

```bash
cd packages/agent
make dev   # 启动所有服务 → localhost:2026
# 上传 EPM 数据 → 端到端分析 → 看 header token usage / 4 subagent 行为
```

### 验证 sync 测试不回归

```bash
cd packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -k "subagent or token or sandbox or run_journal or task_tool" --ignore=tests/test_metric_catalog_live.py -q
```

预期：全绿。如果有新失败，先 stash 本次改动 + git checkout 7981af52 跑同样命令，对比是否 pre-existing。

---

## 7. 风险与注意事项

### 容易混淆的点

- **handoff 文档 ≠ 现实**：第 3 轮 PENDING handoff 有 3 处描述偏差，已在本轮修正。看 handoff 文档时，**关键描述（"本地状态"、"drift 大小"）必须 git diff 验证**，不要假设 handoff 正确。
- **commit message 里的 hash**：本地 commit 标记上游 hash 用 "A-N {hash}" / "上游 {hash}" 格式，**不是 cherry-pick**。grep 上游 hash 在本地 git log 里只能识别"本地命名带了这个 hash"的 commit，不能识别"本地内容等同合入但 commit message 没标"的情况。
- **本地 langsmith 是 0.6.4**：远落后上游。看上游 commit 涉及 uv.lock 时通常不能合（除非用 `uv add` 重新解依赖）。

### 不建议的方向

- **不要**直接 cherry-pick 上游 commit 到受保护文件 — `task_tool.py` / `executor.py` / `sandbox/tools.py` / `lead_agent/prompt.py` 必须 surgical merge
- **不要**用 `./scripts/sync-deerflow.sh --auto-apply` — 这个脚本只看本地是否改过，不识别"间接依赖 Tier 4 模块"的情况（CLAUDE.md 第 9 条血泪教训）
- **不要**为了"跟上上游"而引入 Tier 4 体系尚未支持的模块（per-user filesystem isolation / unified persistence 的旧版残留 / `runtime/events` / 新 checkpointer 抽象）
- **不要**改 `frontend/src/components/ai-elements/`（Vercel AI SDK registry 生成代码，下次更新会被覆盖）
- **不要**为了改 frontend 消息去重而重写 hooks.ts（本地架构跟上游差异巨大，重写成本 ≫ 收益）

---

## 8. 下一位 Agent 的第一步建议

**情况 A：用户要继续 dogfood**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
# 用浏览器打开 localhost:2026，上传 demo-data/ 下的 EthoVision EPM 数据
# 观察：4 subagent 是否正常调度、header token usage 是否含 subagent token、handoff/shared 占位符是否解析
```

**情况 B：用户要做下一轮 sync**

先 fetch：
```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow main
git log --oneline deerflow/main..upstream-real/main   # 本应该是空
git log --oneline d2c2889f..deerflow/main             # 本轮起点之后的新 commit
```

如果有新 commit，逐个评估。评估流程：
1. `git show --stat <hash>` 看涉及的文件
2. 每个受影响文件跑 `git diff <hash>^:<upstream-path> <local-path> | wc -l` 看 drift 大小
3. drift = 0 → 整文件 cp 安全
4. drift 较小 → surgical merge（找出 diff 中 Noldus 定制部分逐段保留）
5. drift 巨大 / 架构分叉 → 空 commit 标记跳过（写明原因）

**情况 C：用户要修 pre-existing 失败**

按风险/价值优先级处理：
- `get_effective_user_id` 缺失（影响 aio_sandbox + auth_type_system，4 个测试）—— Tier 4 系统模块漏装
- `test_ethoinsight_code_skill.py::max_turns == 12 vs 20` —— 决定是改测试还是改配置
- `test_ethoinsight_planning_skill.py` extensions_config.json 注册缺失 —— 一行 JSON 改动

**情况 D：用户要开 PR**

```bash
git log --oneline 7981af52..d2c2889f
# 10 个 sync commit 一起开 PR：
gh pr create --base main --head dev --title "deerflow sync round 3+4: 10 commits (A-8/A-9/A-10/A-11 + 6 upstream delta)"
```

PR body 可以引用本交接文档。

---

## 9. 当前 git 状态

```
本地 dev tip:  31275138 sync(deerflow): 7c42ab3e frontend 异步 chat submit 等待清空
origin/dev:    31275138 （已同步，经一次 force-push 修正 commit history）

deerflow/main: 39f901d3 fix(runs): restore historical runs from persistent store after gateway restart
                       （我们已合入了「可合的全部」，39f901d3 / 48e038f7 / ba864112 已记录跳过原因）
upstream-real/main: 39f901d3（等于 deerflow/main，无 lag）

全部 11 个 sync commit:
  cc2bb889 sync(deerflow): A-11 Runtime 类型 alias（上游 7de9b582 / #2774）
  32b49c17 sync(deerflow): A-8 token bucketing 基础设施（上游 9892a7d4 / #2838）
  5ddb6df2 sync(deerflow): A-10 subagent config 上游对齐（上游 813d3c94 / #2701 局部）
  58abb0b0 sync(deerflow): A-9 subagent token usage 流式到 header（上游 eab7ae3d / #2882）
  f78b520b sync(deerflow): 181d8365 tool result 邻接性 normalize（上游 #2939）
  47094ba9 sync(deerflow): 6e8e6a96 blocking IO detector（上游 #2924）
  60f280bd sync(deerflow): 39f901d3 runs 持久化恢复（上游 #2989）— 不适用，跳过
  871e913a sync(deerflow): 380255f7 sandbox /mnt/user-data 契约（上游 #2873/#2881）
  ce560629 sync(deerflow): 6d3cffb4 frontend 消息去重（上游 #2958）
  31275138 sync(deerflow): 7c42ab3e frontend 异步 chat submit 等待清空（上游 #2940）
```

未提交 pre-existing 改动（与本次 sync 无关，留给原 owner）：
- `packages/agent/.env.example`
- `packages/agent/backend/app/gateway/app.py`
- `packages/agent/backend/app/gateway/auth_middleware.py`
- `packages/agent/backend/app/gateway/routers/feedback_issue.py`（未跟踪）
- `packages/agent/backend/app/gateway/static/`（未跟踪）

---

**会话结束位置**：用户确认 push 后会话即结束于此 handoff 命令调用。
