# 2026-05-25 DeerFlow 上游 sync (e19bec14 → f9b70713) — PR-A 已合 / PR-B 待合 / PR-C 待启动 交接

## 当前任务目标

把上游 deerflow `e19bec14..f9b70713` 这 15 个新 commit 全部以 surgical merge 方式合入 Noldus dev 分支，**保留所有 Noldus 业务定制**（中文 prompt / 5 个 ethoinsight subagent / 4 个工具注册 / EV19+Intent guardrails / shared workspace / TrainingDataMiddleware / ThinkTagMiddleware / MCP 4096 截断 / sandbox extra_env / /mnt/shared / loop_detection 中文化等），按风险分 3 个 PR 提交。

用户决策（已确认）：
- 范围：**全量** — 包括 Tier 4 体系（runtime/runs/manager.py 等）整文件升级
- 提交方式：**按风险分 3 个 PR**（A 低 / B 中 / C 高）

完整计划文件: [/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md](/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md)

## 当前进展

### ✅ PR-A 已合入 dev (PR #33, HEAD = `5de19945`)

分支 `sync/deerflow-pra-low-risk-fixes` 已删除。涵盖 6 个上游 commit / 5 个本地 commit：

| 上游 commit | 改动 |
|---|---|
| 8b697245 | sandbox async readiness polling + sandbox/tools.py 7 个 async coroutine helper |
| 1c5c5857 | write_file 错误信息 2000 字符截断 |
| f9b70713 | sandbox provider chmod 0o644 (`needs_upload_permission_adjustment` 属性) |
| e93f6584 | task_tool `_find_usage_recorder` 显式 4-shape BaseCallbackManager 处理 |
| f0bae286 | dangling_tool_call middleware: 同 tool_call_id 多 ToolMessage 队列化 |
| 8785658a | thread_state.py: `merge_todos` reducer (保留本地 `shared_path` 字段) |

测试: 2792 passed / 19 skipped / 0 failed。1 skip: `test_default_lazy_tool_acquisition_uses_async_provider` (需 langchain >= 1.2.15, 本地 1.2.3)。

### ✅ PR-B 已 push, 等用户在 GitHub 上建 PR 并合入 (分支 `sync/deerflow-prb-middleware-mcp-tracing`, HEAD = `6353bc25`)

worktree 路径: [/home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/](file:///home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/)

PR description: [/tmp/PR-B-description.md](file:///tmp/PR-B-description.md)
建议 PR 标题: `sync(deerflow): PR-B 中间件/MCP/tracing 增强 (e19bec14 → f9b70713 第 2/3 批)`
GitHub URL: https://github.com/noldus-cn-beijing/noldus-insight/pull/new/sync/deerflow-prb-middleware-mcp-tracing

涵盖 4 个上游 commit + 1 跳过 / 5 个本地 commit：

| 上游 commit | 改动 | 本地 commit |
|---|---|---|
| (基础设施) | 引入 6 个上游新增文件 (safety_finish_reason_middleware / safety_termination_detectors / safety_finish_reason_config / mcp/session_pool / runtime/runs/naming / tracing/metadata) | 78c9b570 |
| be0eae98 | SafetyFinishReasonMiddleware 集成 (agent.py + tool_error_handling + worker.py + app_config.py + 651+225+176 行测试) | 11fee2f9 |
| c881d958 | MCP 持久化 session pool (mcp/tools.py 简化版 → head 整文件 + 保留 4096 截断 + 409 行测试) | 78c7f9b2 |
| 923f516d | run_name resolver (worker.py + app/gateway/services.py + 34 行 + 4 行测试) | c79f4f83 |
| df951542 | Langfuse trace metadata 最小集 (worker.py + tracing/__init__.py 导出, model_name=None) | 6353bc25 |

**跳过 dcc6f1e6** loop_detection defer warning injection — 本地 loop_detection_middleware.py 含大量 Noldus 定制 (tool freq 3/5 阈值 / task subagent hash / 中文 warning), 与上游 `_pending_warnings` + `wrap_model_call` 重构架构差异过大，10 个 hunk 的 3-way merge 风险高，**留独立 PR 处理**。

测试: 2869 passed / 19 skipped / 0 failed。

### ⏳ PR-C 待启动 — Tier 4 manager/journal/persistence

涵盖 4 个上游 commit (尚未启动)：

| 上游 commit | 改动 | 风险 |
|---|---|---|
| 31513c2c | persistence/{run,feedback,thread_meta,user}/sql.py: coerce_iso tz-aware (4 文件) | 低 |
| 2eeb5979 | persistence/run/sql.py + runtime/journal.py + runs/manager.py + store: active progress counters | 高 |
| 66d6a6a4 | runs/manager.py + persistence/run/sql.py + store: harden finalization (306 行) | 高 |
| 0fb05825 | runs/manager.py: run creation atomic + cancellation rollback (61 行) | 高 |

用户已决策: **接受上游完整 manager.py / journal.py / persistence/run/sql.py 整文件升级**（本地 manager.py 210 行 是 5-07/08 摇下的简化版 in-memory，非定制特色；上游 654 行完整版 SQL persistence）。

## 关键上下文

### 仓库与分支

- 仓库根: `/home/wangqiuyang/noldus-insight/`
- 主分支: `dev` (HEAD `5de19945`，PR-A 已合)
- 远程:
  - `origin` → `github.com:noldus-cn-beijing/noldus-insight.git`
  - `deerflow` → `github.com:noldus-cn-beijing/deerflow-noldus.git` (上游 fork, HEAD `f9b70713`)
  - `upstream-real` → `github.com/bytedance/deer-flow.git`
- 上次 sync 基准 commit: `e19bec14` (5-21 PR #22+#23)
- 上游 head: `f9b70713`
- 新增 15 个 commit, 6 个新文件

### 工作区 (git worktree)

- 主仓库: `/home/wangqiuyang/noldus-insight` on `dev`
- **PR-B 工作区**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/` on `sync/deerflow-prb-middleware-mcp-tracing`
- 其他: `/home/wangqiuyang/noldus-insight/.claude/worktrees/pr3-lead-robustness/` on `worktree-pr3-lead-robustness` (与本 sync 无关)

`.claude/worktrees/` 已在 `.gitignore` (line 51)。

### 测试运行 (避坑)

⚠️ [[feedback_worktree_uv_editable_install_pitfall]] worktree 内**没有 .venv**，backend pytest 直接跑会因 uv editable install 指向主仓库代码不准。正确做法：

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/packages/agent/backend
MAIN_VENV=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
PYTHONPATH=$PWD/packages/harness:$PWD \
  $MAIN_VENV/bin/python -m pytest tests/ --tb=line -q
```

`DEER_FLOW_CONFIG_PATH` 必须显式指定，否则 3 个 lead_agent 测试会因找不到 config.yaml fail (config.yaml 在 `packages/agent/config.yaml` 不在 worktree 根目录)。

### Lint

```bash
RUFF=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/ruff
$RUFF check packages/agent/backend/<file>
```

注意 backend dev HEAD 有 57 个 pre-existing lint errors (本地测试文件 import 顺序问题)，不在 sync 范围。

### gh CLI 不可用

环境中 `gh` 命令不可用 (段错误 / 不存在)。**push 后告诉用户去 GitHub URL 手工创建 PR**, body 从 `/tmp/PR-?-description.md` 复制。

### Tier 4 体系定义 (PR-C 关键背景)

上游 2026 初引入的多用户隔离体系。Noldus 在 5-07/08 已合入但**简化版**:
- `runtime/runs/manager.py`: 本地 210 行 vs 上游 654 行 (简化版无 SQL persistence, in-memory RunRecord 无 model_name 字段)
- `runtime/journal.py`: 本地 ~570 行 vs 上游 +128 行
- `persistence/run/sql.py`: 本地 ~270 行 vs 上游 +231 行
- `persistence/{feedback,thread_meta,user}/sql.py`: 都有

判断"Tier 4 依赖"的 import：
```python
from deerflow.runtime.user_context import ...
from deerflow.persistence import ...
from deerflow.runtime.events import ...
from deerflow.runtime.checkpointer import ...
from deerflow.runtime.journal import ...
from deerflow.utils.time import ...
from deerflow.config.database_config import ...
from deerflow.config.run_events_config import ...
from deerflow.skills.storage import ...
```

⚠️ [[CLAUDE.md 第 13 条]] 项目状态修正 (5-12)：本仓库已经吃下 Tier 4。多用户研究助手。`get_effective_user_id` / `UserRow` / `@require_permission` 都已存在。

## 关键发现 (PR-A/PR-B 已验证)

### dcc6f1e6 不能照搬

本地 [loop_detection_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py) Noldus 定制:
- `_DEFAULT_TOOL_FREQ_WARN = 3` / `_DEFAULT_TOOL_FREQ_HARD_LIMIT = 5` (上游 30/50)
- 第 64 行 `_stable_tool_key`: task 工具按 `subagent_type::description` 哈希 (子代理身份)
- 中文 warning message (针对 deepseek/qwen)
- 注释提及 wrap_model_call 但实际没有 `_pending_warnings` 队列 + `wrap_model_call` hook 实现

上游 dcc6f1e6 重构是 10 个 hunk 的大改动，3-way merge 会引入大量冲突。需要独立 PR 仔细处理（或者跳过）。本地用 deepseek/qwen 为主，OpenAI/Moonshot `tool_call_ids did not have response messages` 错误风险目前可控。

### df951542 完整集成留独立 PR

本 PR-B 只合入 worker.py + tracing/__init__.py 最小集。完整集成还需:
- `agents/lead_agent/agent.py`: `build_tracing_callbacks()` 注册到 graph invocation root, 让 `propagate_attributes` 把 session/user 抬到 trace 顶
- `models/factory.py`: `attach_tracing` kwarg (避免 in-graph create_chat_model 重复 callback)
- `client.py` (DeerFlowClient): 嵌入式客户端同步注入
- `agents/middlewares/title_middleware.py` 和 `_create_summarization_middleware`: `attach_tracing=False`
- 配套测试 (test_worker_langfuse_metadata.py 248 / test_client_langfuse_metadata.py 159 / test_tracing_metadata.py 137)

注: 本地 `RunRecord` 没有 `model_name` 字段，所以 PR-B 用 `inject_langfuse_metadata(model_name=None)`，langfuse tags 不带 `model:xxx`。**PR-C 整文件升级 manager.py 后, RunRecord 会含 model_name 字段, 届时可补上**。

### sync 脚本盲区

`scripts/sync-deerflow.sh` 的 PROTECTED_FILES 名单**遗漏了**:
- `tools/tools.py` (含 BUILTIN_TOOLS 注册 ethoinsight 4 工具)
- `tools/builtins/__init__.py` (含 `__all__`)
- `subagents/builtins/__init__.py` (含 BUILTIN_SUBAGENTS 注册 5 个 ethoinsight subagent)
- `agents/factory.py` / `agents/__init__.py` / `subagents/registry.py` 等注册类文件
- `agents/middlewares/loop_detection_middleware.py`、`tools/builtins/setup_agent_tool.py` 等含 Noldus 定制的"安全文件"

如果使用 `--auto-apply` 会**洗掉这些注册**（5-21 PR #23 翻车根因，[[feedback_sync_protected_files_registry_loss]]）。**人工核查必做**。

### head-to-head 必做

[[feedback_head_to_head_before_claiming_no_merge]] 评估上游 protected diff 不能只看 +/- 统计，必须 grep 实际文件验证。

### grill handoff 现场核实

[[feedback_grill_handoff_must_be_verified]] 接收 handoff 不能直接信任，要 grep 核实声称的 trace。

## 未完成事项 (按优先级)

### 🔴 高优先级 — PR-C 启动

**前置条件**: 用户在 GitHub 上合入 PR-B 后

1. 删除 PR-B worktree, 切回主仓库 `dev`, `git pull origin dev` 拉取 PR-B 合入后的 HEAD
2. 开 PR-C worktree: `git worktree add .claude/worktrees/deerflow-prc -b sync/deerflow-prc-tier4-manager-upgrade dev`
3. **PR-C 前置 grep 检查** (在 worktree 内):
   ```bash
   for f in runtime/runs/manager.py runtime/journal.py persistence/run/sql.py \
            persistence/feedback/sql.py persistence/thread_meta/sql.py persistence/user/sql.py \
            runtime/runs/store/base.py runtime/runs/store/memory.py; do
     grep -lE "set_experiment_paradigm|identify_ev19|prep_metric_plan|shared_path|/mnt/shared|ethoinsight" \
       "packages/agent/backend/packages/harness/deerflow/$f" 2>/dev/null
   done
   ```
   - 如果**全部为空**: 可整文件接受上游 head
   - 如果**任何文件命中**: 降级为 surgical patch
4. 整文件接受上游 head (按依赖顺序):
   - `persistence/feedback/sql.py`、`persistence/thread_meta/sql.py`、`persistence/user/sql.py` (31513c2c, 独立)
   - `runtime/runs/store/base.py`、`runtime/runs/store/memory.py` (RunStore 接口扩展)
   - `runtime/journal.py` (2eeb5979 配套)
   - `persistence/run/sql.py` (2eeb5979 + 66d6a6a4 + 31513c2c 配套)
   - `runtime/runs/manager.py` (0fb05825 + 66d6a6a4 + 2eeb5979 配套, 210→654 行)
5. 必查的调用方:
   - `runtime/runs/__init__.py` (RunManager 导出)
   - `runtime/runs/worker.py` (PR-B 已改, 注意 record.model_name 现在可用了, 可补上)
   - `runtime/runs/scheduler.py`
   - `runtime/app/lifespan.py` (RunManager 注入点)
   - `runtime/app/factory.py`
   - 业务文件 grep `from .manager import` 或 `from deerflow.runtime.runs.manager import`
6. 同时补 PR-B 中 df951542 的 `inject_langfuse_metadata(model_name=record.model_name)` (PR-C 后 RunRecord 会有此字段)
7. 跑测试: `make test`，关注:
   - `tests/test_run_manager*`
   - `tests/test_run_store*`
   - `tests/test_persistence_*`
   - `tests/test_thread_meta_search.py` (31513c2c 时区错位的回归覆盖)
8. 配套测试合入: 上游 df951542 引入的 `test_run_worker_rollback.py` (102 行) PR-B 时跳过了, PR-C 后可以合
9. 启动 `make dev` 跑 demo thread 强力验证: run 创建 / cancel / 超时 / completion / progress 都正常 (这是 PR-C 最大风险点)
10. push + 让用户在 GitHub 建 PR (gh CLI 不可用): https://github.com/noldus-cn-beijing/noldus-insight/pull/new/sync/deerflow-prc-tier4-manager-upgrade
11. PR description 写入 `/tmp/PR-C-description.md`
12. 建议 PR title: `sync(deerflow): PR-C Tier 4 manager/journal/persistence 整文件升级 (e19bec14 → f9b70713 第 3/3 批)`

### 🟡 中优先级 — 三个 PR 都合后

合入完成后:
1. **更新 sync 同步基准**: 当前 `scripts/sync-deerflow.sh` 检测的 `LAST_SYNC_COMMIT` 还停在 `f0dd8cb` (5-21 sync 没做 subtree squash)。考虑两种修复:
   - 做一次 `git subtree split --prefix=packages/agent/ -b deerflow-sync-snapshot` + push subtree
   - 或者修改 `scripts/sync-deerflow.sh` 让它支持 `LAST_SYNC_COMMIT` 显式覆盖（环境变量或 .deerflow-sync-state 文件）
2. **更新 scripts/sync-deerflow.sh 的 PROTECTED_FILES**: 把以下文件加入 (避免 5-21 翻车重演):
   - `tools/tools.py`
   - `tools/builtins/__init__.py`
   - `subagents/builtins/__init__.py`
   - `agents/factory.py` / `agents/__init__.py`
   - `subagents/registry.py`
3. **更新 docs/sop/deerflow-sync-sop.md**: 把"head-to-head 必做" + "注册类文件必看 BUILTIN_TOOLS / __all__"两个教训写进去

### 🟢 低优先级 — 独立后续 PR

1. **dcc6f1e6 loop_detection defer warning injection**: 需要仔细 surgical merge 本地中文 + 阈值 + task hash 定制与上游 `_pending_warnings` + `wrap_model_call` 架构。OpenAI/Moonshot `tool_call_ids did not have response messages` 修复值得吃下，但优先级不高。
2. **df951542 完整集成**: agent.py callback root attach + client.py 同步 + models/factory.py `attach_tracing` kwarg + title_middleware/summarization `attach_tracing=False` + 测试 (test_worker_langfuse_metadata.py 248 / test_client_langfuse_metadata.py 159 / test_tracing_metadata.py 137)。Langfuse 控制台能否看到 session/user 维度聚合。
3. **langchain 升级**: 1.2.3 → 1.2.15+，解除 8b697245 `test_default_lazy_tool_acquisition_uses_async_provider` 测试的 skip
4. **backend/app/gateway/routers/uploads.py 评估**: PR-A 时跳过未处理，本地版本比上游简化（缺 `claim_unique_filename` + f9b70713 的 `_make_file_sandbox_readable`）。决定是回归上游完整版还是保留本地简化版。

## 风险与注意事项

### ⚠️ 不要做的事

1. **不要用 `git show deerflow/main:<file> > <local_file>` 直接覆盖含 Noldus 定制的文件** — 永远先 grep marker (set_experiment_paradigm / identify_ev19 / prep_metric_plan / shared_path / /mnt/shared / extra_env / ethoinsight / ArchivingSummarization / ThinkTag / TrainingData / GateEnforcement / HandoffIsolation / Ev19Template / BUILTIN_TOOLS / `__all__`)
2. **不要使用 `./scripts/sync-deerflow.sh --auto-apply`** — 脚本"安全文件"判断有误
3. **不要 force-push 到 main/master**
4. **不要 amend 已合入 PR 的 commit**
5. **不要在 worktree 内跑 backend pytest 不设 PYTHONPATH** — 会用主仓库代码 (uv editable install 副作用)

### ⚠️ Auto mode 与 plan mode

当前会话开始时是 plan mode (生成了 `/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md`), 后来切到 auto mode 一直执行到现在。新会话默认应该尊重用户的偏好。

### ⚠️ 中间件链插入位置

PR-B 的 SafetyFinishReasonMiddleware 注册位置: 在 `_build_middlewares` 函数中、`GateEnforcementMiddleware` 之后、`ClarificationMiddleware` 之前。变量名用 `app_config.safety_finish_reason`（不是上游的 `resolved_app_config`）。

LangChain after_model 反序执行: 列表中**最后注册的反而最先跑**。Safety 应该最后注册让它最先清理 tool_calls。

### ⚠️ RunRecord.model_name

本地不存在。PR-B 的 `inject_langfuse_metadata(model_name=None)` 是临时妥协。PR-C 升级 manager.py 后会有此字段，记得回头补上。

### ⚠️ Worker.py 已被 PR-B 改动多次

PR-B 给 worker.py 加了:
1. `runtime_ctx["__run_journal"] = journal` (be0eae98 配套)
2. `from .naming import resolve_root_run_name` import + `config.setdefault("run_name", ...)` (923f516d)
3. `import os` + `from deerflow.runtime.user_context import get_effective_user_id` + `from deerflow.tracing import inject_langfuse_metadata` + `inject_langfuse_metadata(...)` 调用 (df951542 最小集)

PR-C 接受上游 worker.py 时**不能整文件覆盖**，需要保留 PR-B 这 3 个改动。或者: 用上游 head worker.py 整文件接受 + 重新 apply PR-B 这 3 个改动。

## 下一位 Agent 的第一步建议

### 起始确认

1. 读这份 handoff
2. 读 plan 文件: [/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md](file:///home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md)
3. 询问用户 PR-B 是否已合: `git fetch origin && git log --oneline origin/dev -5`
   - 看到 `Merge pull request #XX from .../sync/deerflow-prb-middleware-mcp-tracing` → PR-B 已合
   - 没看到 → 等用户合 PR-B 再继续

### 如果 PR-B 已合

```bash
cd /home/wangqiuyang/noldus-insight
git checkout dev
git pull origin dev
git worktree remove .claude/worktrees/deerflow-prb 2>&1
git branch -D sync/deerflow-prb-middleware-mcp-tracing 2>&1
git worktree add .claude/worktrees/deerflow-prc -b sync/deerflow-prc-tier4-manager-upgrade dev
cd .claude/worktrees/deerflow-prc

# 前置 grep 检查
for f in runtime/runs/manager.py runtime/journal.py persistence/run/sql.py \
         persistence/feedback/sql.py persistence/thread_meta/sql.py persistence/user/sql.py \
         runtime/runs/store/base.py runtime/runs/store/memory.py; do
  grep -lE "set_experiment_paradigm|identify_ev19|prep_metric_plan|shared_path|/mnt/shared|ethoinsight" \
    "packages/agent/backend/packages/harness/deerflow/$f" 2>/dev/null
done
```

然后按计划 PR-C 执行。

### 如果 PR-B 还没合

继续做这些后台准备工作（不要修改 dev 状态）:
1. 核查上游 head `runtime/runs/manager.py` 与本地差异，提前找出潜在调用方影响
2. 检查 RunRecord 字段差异（本地无 model_name, 上游可能有）
3. 列出 PR-C 预估 commit 数（至少 4 个 surgical commit, 每个对应一个上游 sha）
4. 等用户合入 PR-B 通知

## 关键文件清单

### Plan / Handoff

- [/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md](file:///home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md) — 主计划
- [/tmp/PR-A-description.md](file:///tmp/PR-A-description.md) — PR-A description (已合)
- [/tmp/PR-B-description.md](file:///tmp/PR-B-description.md) — PR-B description (待用户建 PR)
- 本文件 — 交接

### 项目文档

- [/home/wangqiuyang/noldus-insight/CLAUDE.md](/home/wangqiuyang/noldus-insight/CLAUDE.md) — 仓库说明 (sync 规则 + Tier 4 + 项目状态)
- [/home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md](/home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md) — backend 架构
- [/home/wangqiuyang/noldus-insight/docs/sop/deerflow-sync-sop.md](/home/wangqiuyang/noldus-insight/docs/sop/deerflow-sync-sop.md) — sync SOP
- [/home/wangqiuyang/noldus-insight/scripts/sync-deerflow.sh](/home/wangqiuyang/noldus-insight/scripts/sync-deerflow.sh) — 同步脚本

### PR-B worktree (待 PR-B 合后清理)

- [/home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/](file:///home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-prb/) — 5 个 commit on `sync/deerflow-prb-middleware-mcp-tracing`

### 相关 memory (auto-memory)

- `feedback_worktree_uv_editable_install_pitfall.md` — worktree pytest 盲区
- `feedback_sync_protected_files_registry_loss.md` — 5-21 PR #23 翻车记忆
- `feedback_head_to_head_before_claiming_no_merge.md` — head-to-head 必做
- `feedback_grill_handoff_must_be_verified.md` — handoff 不能直接信任
- `project_2026-05-21_deerflow_sync_complete.md` — 5-21 sync 完成记录 (本次 sync 的起点)
