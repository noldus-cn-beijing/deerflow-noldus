# 2026-05-25 DeerFlow upstream sync 完成 (e19bec14 → f9b70713) — 全部 5 PR 已合 + 善后待办

## 当前任务目标

DeerFlow 上游 `e19bec14..f9b70713` 这 15 个新 commit 全部以 surgical merge 方式合入 Noldus dev 分支，**保留所有 Noldus 业务定制**（中文 prompt / 5 ethoinsight subagent / 4 工具注册 / EV19+Intent guardrails / shared workspace / TrainingDataMiddleware / ThinkTagMiddleware / MCP 4096 截断 / sandbox extra_env / /mnt/shared / loop_detection 中文化 / verdict 三分类等）。

**状态：✅ 完成。** dev HEAD `37bcbba4`，2989 passed / 19 skipped / 0 failed。

## 当前进展

### ✅ 全部 5 PR 已合入 dev

| PR | HEAD merge | 内容 | 测试通过数 |
|---|---|---|---|
| PR-A (#33) | 5de19945 | DeerFlow sync 低风险修复 6 commit | 2792 |
| PR-B (#34) | d30adeb7 | SafetyFinishReason + MCP session pool + run_name + Langfuse 最小集 5 commit | 2869 |
| PR-C (#36) | 11c28d51 | Tier 4 manager/journal/persistence 整文件升级 9 commit | 2960 |
| PR loop_detection (#35) | 2c5296ea | dcc6f1e6 defer warning injection surgical merge 1 commit | 2882 |
| PR langfuse-full (#38) | 37bcbba4 | df951542 Langfuse 完整集成 6 commit | **2989** |

dev HEAD = `37bcbba4`。worktrees 全部清理，分支全部删除（除了无关的 `pr3-lead-robustness`）。

### ✅ 上游 15 commit 处理一览

| 上游 commit | 描述 | 落点 |
|---|---|---|
| 8b697245 | sandbox async readiness polling + 7 async helper | PR-A |
| 1c5c5857 | write_file 错误信息 2000 字符截断 | PR-A |
| f9b70713 | sandbox provider chmod 0o644 | PR-A |
| e93f6584 | task_tool 4-shape BaseCallbackManager 处理 | PR-A |
| f0bae286 | dangling_tool_call 同 tool_call_id 多 ToolMessage | PR-A |
| 8785658a | thread_state merge_todos reducer | PR-A |
| be0eae98 | SafetyFinishReasonMiddleware | PR-B |
| c881d958 | MCP 持久化 session pool | PR-B |
| 923f516d | run_name resolver | PR-B |
| df951542 | Langfuse trace metadata 最小集 | PR-B (worker.py 单点) |
| df951542 | Langfuse 完整集成 | PR-2 (#38, agent.py + client.py + factory.py + title + tests) |
| 31513c2c | persistence/*/sql.py coerce_iso | PR-C |
| 2eeb5979 | active progress counters | PR-C |
| 66d6a6a4 | finalization hardening | PR-C |
| 0fb05825 | run creation atomic + cancellation rollback | PR-C |
| dcc6f1e6 | loop_detection defer warning injection | PR loop_detection (#35) |

## 关键上下文

### 仓库与分支

- 仓库根: `/home/wangqiuyang/noldus-insight/`
- 主分支: `dev` (HEAD `37bcbba4`)
- 远程:
  - `origin` → `github.com:noldus-cn-beijing/noldus-insight.git`
  - `deerflow` → `github.com:noldus-cn-beijing/deerflow-noldus.git` (上游 fork, HEAD `f9b70713`)
  - `upstream-real` → `github.com/bytedance/deer-flow.git`
- 5 个合入的 PR 编号: #33 (PR-A), #34 (PR-B), #35 (loop_detection), #36 (PR-C), #38 (langfuse-full)
- 注意：PR #37 在 GitHub 上不存在 / 用户合错过一次（PR-C 先合到 loop_detection 分支再修正）

### worktrees

- 主仓库: `/home/wangqiuyang/noldus-insight` on `dev`
- 其他: `/home/wangqiuyang/noldus-insight/.claude/worktrees/pr3-lead-robustness` on `worktree-pr3-lead-robustness` (与本 sync 无关，是另一会话遗留)

`.claude/worktrees/` 已在 `.gitignore`。

### 测试运行（避坑）

⚠️ [[feedback_worktree_uv_editable_install_pitfall]] worktree 内**没有 .venv**。正确方式：

```bash
cd packages/agent/backend
MAIN_VENV=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
PYTHONPATH=$PWD/packages/harness:$PWD \
  $MAIN_VENV/bin/python -m pytest tests/ --tb=line -q
```

### Lint

```bash
RUFF=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/ruff
$RUFF check packages/agent/backend/<file>
```

注意 dev HEAD 有 1 个 pre-existing F841 lint error（`workflow_mode` 未使用，在 `agents/lead_agent/agent.py:490`）— 不在本次 sync 范围。

### gh CLI 不可用

环境中 `gh` 命令不可用（段错误 / 不存在）。**push 后告诉用户去 GitHub URL 手工创建 PR**，body 从 `/tmp/PR-*-description.md` 复制。

## 关键发现

### sync 脚本盲区（**未修复，待办**）

`scripts/sync-deerflow.sh` 的 PROTECTED_FILES 名单**遗漏了**:

- `tools/tools.py` (含 BUILTIN_TOOLS 注册 ethoinsight 4 工具)
- `tools/builtins/__init__.py` (含 `__all__`)
- `subagents/builtins/__init__.py` (含 BUILTIN_SUBAGENTS 注册 5 个 ethoinsight subagent)
- `agents/factory.py` / `agents/__init__.py` / `subagents/registry.py` 等注册类文件
- `agents/middlewares/loop_detection_middleware.py`、`tools/builtins/setup_agent_tool.py` 等含 Noldus 定制的"安全文件"

如果使用 `--auto-apply` 会**洗掉这些注册**（5-21 PR #23 翻车根因，[[feedback_sync_protected_files_registry_loss]]）。**人工核查必做**。

### head-to-head 必做

[[feedback_head_to_head_before_claiming_no_merge]] 评估上游 protected diff 不能只看 +/- 统计，必须 grep 实际文件验证。这次 PR-C 时验证了 `persistence/feedback/sql.py` — grep 默认 12 marker 没命中但实际含 verdict/revised_text/message_id（训练飞轮）定制，必须 surgical merge。

### sync 同步基准未更新

当前 `scripts/sync-deerflow.sh` 检测的 `LAST_SYNC_COMMIT` 还停在 `f0dd8cb` (5-21 sync 没做 subtree squash)。下次 sync 之前要修。

### PR-C 关键架构变化

manager.py 从 210 行简化版升级到 654 行完整版。新增字段：
- `RunRecord`: `model_name`, `store_only`, 9 个 token counters
- 新方法: `update_run_completion`, `update_model_name`, `update_run_progress`, `reconcile_orphaned_inflight_runs`, `list_by_thread(user_id, limit)`

调用方 `app/gateway/deps.py` 已更新：`RunManager(store=app.state.run_store)` + SQLite startup recovery。

### langfuse 完整集成核心约定

📌 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 顶部新增了 module docstring "INVARIANT — tracing callback placement"。**未来任何在 lead_agent 模块内或可达的中间件内新加 `create_chat_model(...)` 必须传 `attach_tracing=False`**，否则会发出重复 span 且 langfuse_* keys 被剥导致 session_id/user_id 进不到 trace。当前 4 个 site:
1. bootstrap agent (`make_lead_agent` 内)
2. default lead agent (`make_lead_agent` 内)
3. `_create_summarization_middleware` 内有 2 处 (config.model_name 分支 / 默认分支)
4. `title_middleware.py` 的 `TitleMiddleware._build_title_kwargs`

未来上游有 `test_make_lead_agent_attaches_tracing_callbacks_at_graph_root` 这个 regression guard 测试，可以下次 sync 一并合（见下文未完成事项）。

## 未完成事项（按优先级）

### 🟡 中优先级 — sync 善后

1. **更新 `scripts/sync-deerflow.sh` 的 PROTECTED_FILES**: 把以下文件加入（避免 5-21 翻车重演）:
   - `tools/tools.py`
   - `tools/builtins/__init__.py`
   - `subagents/builtins/__init__.py`
   - `agents/factory.py` / `agents/__init__.py`
   - `subagents/registry.py`
   - `persistence/feedback/sql.py` (本次发现含 verdict 训练飞轮定制)

2. **修正 sync 同步基准**: 当前 `LAST_SYNC_COMMIT` 停在 `f0dd8cb` (5-21)，应推进到 `f9b70713` (本次 sync 起点)。两种修复:
   - 做一次 `git subtree split --prefix=packages/agent/ -b deerflow-sync-snapshot` + push subtree
   - 或者修改 `scripts/sync-deerflow.sh` 让它支持 `LAST_SYNC_COMMIT` 显式覆盖（环境变量或 `.deerflow-sync-state` 文件）

3. **更新 `docs/sop/deerflow-sync-sop.md`**: 把以下教训写进去:
   - "head-to-head 必做"（不能只看 +/- 统计判 "无可合"）
   - "注册类文件必看 BUILTIN_TOOLS / `__all__`"
   - "含 verdict/revised_text/message_id 的飞轮文件必须 surgical merge" (本次新增教训)
   - "in-graph `create_chat_model` 必须传 `attach_tracing=False` 约定" (本次新增)

### 🟢 低优先级 — 独立后续 PR

1. **langchain 升级**: 1.2.3 → 1.2.15+，解除 8b697245 `test_default_lazy_tool_acquisition_uses_async_provider` 测试的 skip

2. **backend/app/gateway/routers/uploads.py 评估**: PR-A 时跳过未处理，本地版本比上游简化（缺 `claim_unique_filename` + f9b70713 的 `_make_file_sandbox_readable`）。决定回归上游完整版还是保留本地简化版

3. **dev hot reload (#3107 BUG-001)**: 上游另一组 PR 引入 `startup_config` 参数 + `get_config` 重构 + `get_run_context` 调整。PR-C 时跳过未做（超出 PR-C 范围）。可独立 PR 同步

4. **`test_make_lead_agent_attaches_tracing_callbacks_at_graph_root` 等上游测试**: PR-2 跳过的 ~415 行测试用 `_make_lead_agent` 签名 + `app_config` 形参，本地 `make_lead_agent` 不同。其中 regression guard 测试有价值（防未来 contributor 忘传 `attach_tracing=False`）。配合 #3107 hot reload PR 一起做

5. **dcc6f1e6 Noldus 定制回归测试**: loop_detection PR 时合入了 5 项 Noldus 定制（task per subagent_type bucket / ethoinsight 提示 / 阈值 3/5 等），但**缺专门的回归覆盖**。可以独立加一个 `test_loop_detection_noldus_customizations.py`

### 🔴 高优先级 — 无（sync 任务已结束）

## 风险与注意事项

### ⚠️ 不要做的事

1. **不要用 `git show deerflow/main:<file> > <local_file>` 直接覆盖含 Noldus 定制的文件** — 永远先 grep marker (`set_experiment_paradigm` / `identify_ev19` / `prep_metric_plan` / `shared_path` / `/mnt/shared` / `extra_env` / `ethoinsight` / `ArchivingSummarization` / `ThinkTag` / `TrainingData` / `GateEnforcement` / `HandoffIsolation` / `Ev19Template` / `BUILTIN_TOOLS` / `__all__` / `verdict` / `revised_text` / `message_id`)
2. **不要使用 `./scripts/sync-deerflow.sh --auto-apply`** — 脚本"安全文件"判断有误
3. **不要 force-push 到 main/master**
4. **不要 amend 已合入 PR 的 commit**
5. **不要在 worktree 内跑 backend pytest 不设 PYTHONPATH** — 会用主仓库代码 (uv editable install 副作用)
6. **新增 in-graph `create_chat_model` 必须传 `attach_tracing=False`** — 见 `lead_agent/agent.py` 顶部 INVARIANT docstring

### ⚠️ 易混淆

- 本次 5 PR 涉及 PR 编号 #33, #34, #35, #36, #38。**#37 不存在**（一次合错产生的中间状态）。本仓库实际 PR 列表去 GitHub 直接查
- `RunManager(store=...)` 现已是必填 — 调用方都要传 store（参考 deps.py:128 起的实例）
- `_strip_loop_warning_text` 已删除（loop_detection 新架构后不需要）— 任何 channels 代码不要试图恢复这个 helper

## 下一位 Agent 的第一步建议

### 起始确认

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin
git log --oneline dev -10
# 看到 37bcbba4 Merge pull request #38 ... 就表示本次 sync 全部已合
```

### 如果用户要做善后清理（中优先级）

按上面 🟡 中优先级 3 个项目依次做。建议起一个 worktree:
```bash
git worktree add .claude/worktrees/sync-cleanup -b chore/sync-cleanup-protected-files dev
```

涉及文件:
- `scripts/sync-deerflow.sh` — 加 PROTECTED_FILES
- `docs/sop/deerflow-sync-sop.md` — 更新教训
- 可能引入 `.deerflow-sync-state` 文件

### 如果用户要做下一波 sync

```bash
# 1. fetch 上游
git fetch deerflow
git log --oneline f9b70713..deerflow/main | head -20  # 看新增 commit

# 2. 按风险分批（参考本次 PR-A/B/C 拆法）
# 3. 必查 [[feedback_head_to_head_before_claiming_no_merge]] + [[feedback_sync_protected_files_registry_loss]]
```

### 如果用户切换到完全不同的任务

不需要做任何 sync 相关的事。直接进入新任务即可。

## 关键文件清单

### Plan / Handoff

- 本文件 — 本次 sync 完成交接
- [/home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md](file:///home/wangqiuyang/.claude/plans/commit-surgical-merge-home-wangqiuyang-breezy-thunder.md) — 主计划 (5-25 创建)
- [docs/handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md](docs/handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md) — 本次 sync 开始时的交接（PR-A 已合 / PR-B 待合 / PR-C 待启动 状态）
- `/tmp/PR-A-description.md` / `/tmp/PR-B-description.md` / `/tmp/PR-C-description.md` / `/tmp/PR-loop-detection-description.md` / `/tmp/PR-langfuse-full-description.md` — 5 个 PR description

### 项目文档

- [CLAUDE.md](CLAUDE.md) — 仓库说明 (sync 规则 + Tier 4 + 项目状态)
- [packages/agent/backend/CLAUDE.md](packages/agent/backend/CLAUDE.md) — backend 架构
- [docs/sop/deerflow-sync-sop.md](docs/sop/deerflow-sync-sop.md) — sync SOP（待更新）
- [scripts/sync-deerflow.sh](scripts/sync-deerflow.sh) — 同步脚本（PROTECTED_FILES 待加）

### 新增的关键架构约定

- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 顶部 module docstring — INVARIANT tracing callback placement
- `packages/agent/backend/packages/harness/deerflow/tracing/__init__.py` — exports `build_langfuse_trace_metadata`, `build_tracing_callbacks`, `inject_langfuse_metadata`
- `packages/agent/backend/packages/harness/deerflow/tracing/metadata.py` (105 行新增) — Langfuse metadata builder + injector

### 相关 memory (auto-memory)

- `feedback_worktree_uv_editable_install_pitfall.md` — worktree pytest 盲区
- `feedback_sync_protected_files_registry_loss.md` — 5-21 PR #23 翻车记忆
- `feedback_head_to_head_before_claiming_no_merge.md` — head-to-head 必做
- `feedback_grill_handoff_must_be_verified.md` — handoff 不能直接信任
- `project_2026-05-21_deerflow_sync_complete.md` — 5-21 sync 完成记录 (本次 sync 的起点)
