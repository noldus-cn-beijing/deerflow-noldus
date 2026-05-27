# 2026-05-25 DeerFlow sync 善后 PR push 完待用户合 + 下次 sync 起点已就位

## 当前任务目标

5-25 DeerFlow upstream sync 全部完成后（dev HEAD `37bcbba4`，5 个 PR #33-#38 共 15 commit），按 [前一 handoff (2026-05-25-deerflow-sync-all-prs-merged)](2026-05-25-deerflow-sync-all-prs-merged-handoff.md) 中提到的"中优先级 sync 善后 3 项"做闭环：

1. ✅ 更新 `scripts/sync-deerflow.sh` PROTECTED_FILES（加 6+ 注册类/飞轮文件）
2. ✅ 修正 sync 同步基准（推进到 `f9b70713`）
3. ✅ 更新 `docs/sop/deerflow-sync-sop.md`（4 个新教训）

**状态：✅ 全部完成。** PR push 完成（`chore/sync-protected-files-and-sop`，HEAD `94faeed0`），待用户在 GitHub 上手工合入。

## 当前进展

### ✅ 已 push 待合 — PR (HEAD `94faeed0`)

分支: `chore/sync-protected-files-and-sop` (基于 dev `37bcbba4`)
GitHub URL: https://github.com/noldus-cn-beijing/noldus-insight/pull/new/chore/sync-protected-files-and-sop
PR description: [/tmp/PR-sync-cleanup-description.md](file:///tmp/PR-sync-cleanup-description.md)
建议 PR title: `chore(sync): 5-25 sync 善后 — PROTECTED_FILES 扩 + .deerflow-sync-state 机制 + SOP 4 教训`

worktree 路径: [/home/wangqiuyang/noldus-insight/.claude/worktrees/sync-cleanup/](file:///home/wangqiuyang/noldus-insight/.claude/worktrees/sync-cleanup/) on `chore/sync-protected-files-and-sop`（PR 合入后清理）

#### 2 个 commit 内容

**Commit 1** `55e35756` chore(sync): sync-deerflow.sh 扩 PROTECTED_FILES + 加 .deerflow-sync-state 机制

`PROTECTED_FILES` 13 → 22，新增 9 个文件：
- 注册类（5-21 PR #23 翻车根因）: `tools/tools.py`, `tools/builtins/__init__.py`, `agents/__init__.py`, `agents/factory.py`, `subagents/registry.py`
- 飞轮 schema（5-25 PR #36 发现）: `persistence/feedback/sql.py`（含 verdict 三分类 + revised_text + message_id, 默认 grep 不命中）
- Loop detection 中文化（5-25 PR #35）: `agents/middlewares/loop_detection_middleware.py`
- 其它（5-06 上游脚本误标）: `tools/builtins/setup_agent_tool.py`
- Guardrail unique-name fix: `guardrails/middleware.py`

同步基准机制改进：
- 引入 `.deerflow-sync-state` 文件持久化记录 `last_sync_commit`
- 支持 `DEERFLOW_LAST_SYNC` 环境变量一次性覆盖
- subtree squash commit message 作为 fallback
- 初始化 `.deerflow-sync-state` 记录 `f9b70713`

脚本鲁棒性：
- 兼容 worktree 运行（`.git` 既可以是目录也可以是文件）
- 完成 sync 后输出更新 `.deerflow-sync-state` 提示

**Commit 2** `94faeed0` docs(sync-sop): 加 5-25 sync 4 个教训 + 配套 PROTECTED_FILES 表 + state file step

4 个新教训：
1. **head-to-head 必做** — 不能只看 +/- 统计判 "跳过"（反例 PR-B dcc6f1e6 +228/-78 判跳过，实际 80% 是上游新架构应整文件接受 + 重新打 Noldus 定制）
2. **注册类文件必看 BUILTIN_TOOLS / `__all__` 集合** — 反例 5-21 PR #23 翻车洗掉 4 个 ethoinsight 工具注册
3. **飞轮 / 训练数据 schema 必须 surgical merge** — 反例 PR-C 时 `persistence/feedback/sql.py` 默认 12 grep marker 都不命中，实际含 verdict 三分类 + revised_text + message_id 飞轮 schema；新增 grep pattern `verdict | revised_text | message_id`
4. **in-graph create_chat_model 必须传 `attach_tracing=False`** — 5-25 PR #38 引入的架构约定，详见 `lead_agent/agent.py` 顶部 docstring "INVARIANT — tracing callback placement"

配套 SOP 更新：
- Step 2 加同步基准来源说明 + 优先级
- Step 6 新增更新 `.deerflow-sync-state` 提交流程
- 选项 C "整文件接受 + 重新打 Noldus 定制"（5-25 推荐策略）
- 受保护文件清单按 4 类组织

### ✅ 验证

- `bash -n scripts/sync-deerflow.sh` 语法 OK
- dry-run with `.deerflow-sync-state f9b70713` → "上游无新改动" ✅
- dry-run with `DEERFLOW_LAST_SYNC=f0dd8cb` → 165 commit / 20 protected（vs 旧版仅捕获 ~13）✅

### ✅ 改动一览

| 文件 | 改动 |
|---|---|
| `scripts/sync-deerflow.sh` | +63 行: PROTECTED_FILES 13→22, `.deerflow-sync-state` 读取, env var 覆盖, worktree 兼容 |
| `.deerflow-sync-state` | 新增, 记录 `last_sync_commit: f9b70713` + 历史背景 |
| `docs/sop/deerflow-sync-sop.md` | +145 行: 4 教训章节 + Step 2/6 更新 + PROTECTED 表 4 类组织 + 选项 C |

只改 scripts/docs/state，**不动任何 Python 代码 / 测试 / 配置**。

## 关键上下文

### 仓库与分支

- 仓库根: `/home/wangqiuyang/noldus-insight/`
- 主分支: `dev` (HEAD `3f5dcada`，PR-2 langfuse-full 合后用户又自己 commit 了 docs，再之上没有改动)
- 远程: `origin` → `github.com:noldus-cn-beijing/noldus-insight.git`
- 5-25 sync 5 PR 全合: #33 (PR-A), #34 (PR-B), #35 (loop_detection), #36 (PR-C), #38 (langfuse-full)
- 本次善后 PR 分支: `chore/sync-protected-files-and-sop` HEAD `94faeed0`（待用户合）

### worktrees

- 主仓库: `/home/wangqiuyang/noldus-insight` on `dev`
- 善后 worktree: `/home/wangqiuyang/noldus-insight/.claude/worktrees/sync-cleanup` on `chore/sync-protected-files-and-sop`（**待用户合 PR 后清理**）
- 无关 worktree: `/home/wangqiuyang/noldus-insight/.claude/worktrees/pr3-lead-robustness`

### gh CLI 不可用

环境中 `gh` 命令不可用（段错误 / 不存在）。**push 后告诉用户去 GitHub URL 手工创建 PR**，body 从 `/tmp/PR-sync-cleanup-description.md` 复制。

### 下次 sync 起点

`.deerflow-sync-state` 文件已就位记录 `last_sync_commit: f9b70713`：

```yaml
# /home/wangqiuyang/noldus-insight/.deerflow-sync-state（PR 合入后）
last_sync_commit: f9b70713
last_sync_date: 2026-05-25
last_sync_commits_count: 15
last_sync_prs: "#33 (PR-A), #34 (PR-B), #35 (loop_detection), #36 (PR-C), #38 (langfuse-full)"
```

下次 `./scripts/sync-deerflow.sh --dry-run` 自动从此点起算。

## 关键发现 / 决策

### 22 个 PROTECTED_FILES 全部存在

`ls -f` 验证所有 22 个 path 都对应实际文件，无打错。

### NOLDUS_ONLY vs UPSTREAM_HAS

调研 23 个含 Noldus markers 的本地文件：
- **NOLDUS_ONLY**（上游没有）22 个 — 脚本根本不会 touch，不需保护：`experiment_context.py` / 6 个 ethoinsight subagent / 7 个 guardrail provider / 3 个 Noldus 中间件等
- **UPSTREAM_HAS**（上游有）1 个 — 已加入 PROTECTED_FILES: `guardrails/middleware.py`

### 脚本测试用 DEERFLOW_LAST_SYNC=f0dd8cb 触发的 20 protected

含本次新增 7 个 + 老 13 个里的 13 个（不是全部 13 个上游都改了）。证明扩 PROTECTED_FILES 是有效的真正发挥作用。

## 未完成事项

### 🔴 高优先级 — 等用户合 PR

**等用户合 PR `chore/sync-protected-files-and-sop` (HEAD `94faeed0`) 后**：
1. `cd /home/wangqiuyang/noldus-insight && git pull origin dev`
2. `git worktree remove .claude/worktrees/sync-cleanup`
3. `git branch -D chore/sync-protected-files-and-sop`

### 🟡 中优先级 — 都是别的事

5-25 sync 本身的善后**已全部闭环**。其它未完事项（来自上一 handoff）属于"下次某个 sync 时处理"或"独立任务"：

- langchain 升级 1.2.3 → 1.2.15+（解 `test_default_lazy_tool_acquisition_uses_async_provider` 的 skip）
- `backend/app/gateway/routers/uploads.py` 评估（本地比上游简化）
- 上游 #3107 BUG-001 mtime hot reload (`startup_config` 重构)
- `test_make_lead_agent_attaches_tracing_callbacks_at_graph_root` regression guard 测试
- dcc6f1e6 Noldus 5 项定制专门的回归测试

### 🟢 低优先级 — 无

## 风险与注意事项

### ⚠️ PR 合入后必做

`.deerflow-sync-state` 文件**必须随 PR 一起合入 dev**，否则下次 sync 还是回退到 fallback。已包含在本 PR commit 1 中（`git ls-files .deerflow-sync-state` 应该返回路径）。

### ⚠️ 下次 sync 流程

下次 sync 时按更新后的 SOP（含 4 个教训）走：

1. **head-to-head**: `diff <(git show deerflow/main:<上游路径>) <本地路径> | head -50` 再判跳过/接受
2. **注册类 / 飞轮文件**: 已在 PROTECTED_FILES，不会被脚本"安全合入"
3. **新增 grep pattern**: `verdict | revised_text | message_id` 也要查
4. **in-graph `create_chat_model`**: 任何新加都必须传 `attach_tracing=False`

### ⚠️ 不要做的事

1. 不要在本地不 pull 远端就改 `.deerflow-sync-state` — PR 合入即生效，无需手动维护
2. 不要 force-push 到 `chore/sync-protected-files-and-sop` 分支（用户可能正在 review）
3. 不要因为脚本扩 PROTECTED_FILES 就回头改之前的 5 个 PR — 那些已合入的状态正确

## 下一位 Agent 的第一步建议

### 起始确认

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin
git log --oneline dev origin/chore/sync-protected-files-and-sop -5
```

### 如果 PR `#???` 已合入 dev

```bash
# 1. 清理 worktree + 分支
cd /home/wangqiuyang/noldus-insight
git checkout dev && git pull origin dev
git worktree remove .claude/worktrees/sync-cleanup 2>&1
git branch -D chore/sync-protected-files-and-sop 2>&1

# 2. 验证 .deerflow-sync-state 已在 dev
test -f .deerflow-sync-state && cat .deerflow-sync-state

# 3. 跑一次 dry-run 验证脚本正常
./scripts/sync-deerflow.sh --dry-run
# 期望: "上游无新改动，已是最新！" (deerflow/main HEAD 还是 f9b70713)
```

### 如果 PR 还没合

继续做其它任务即可，不要碰 `chore/sync-protected-files-and-sop` 分支。worktree 留着等用户合 PR。

### 如果有新的 deerflow 上游 commit 需要 sync

按更新后的 SOP 走（注意 4 个教训）：

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow
./scripts/sync-deerflow.sh --dry-run
# 脚本会自动从 .deerflow-sync-state f9b70713 起算
```

## 关键文件清单

### PR / 善后

- [/tmp/PR-sync-cleanup-description.md](file:///tmp/PR-sync-cleanup-description.md) — 本次善后 PR description
- 本文件 — 善后完成交接

### 项目文档（本次更新）

- [scripts/sync-deerflow.sh](scripts/sync-deerflow.sh) — sync 脚本（已更新）
- [.deerflow-sync-state](.deerflow-sync-state) — 同步基准状态（新增）
- [docs/sop/deerflow-sync-sop.md](docs/sop/deerflow-sync-sop.md) — sync SOP（含 4 个新教训）

### 5-25 sync 主链条 handoff（按时间顺序）

- [docs/handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md](docs/handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md) — sync 进行中
- [docs/handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md](docs/handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md) — sync 全 5 PR 合完
- 本文件 — sync 善后完成

### 相关 memory (auto-memory)

- `feedback_sync_protected_files_registry_loss.md` — 5-21 PR #23 翻车记忆（本次 PR 直接对应教训 2）
- `feedback_head_to_head_before_claiming_no_merge.md` — head-to-head 必做（教训 1）
- `feedback_worktree_uv_editable_install_pitfall.md` — worktree pytest 盲区（教训 4 配套）
- `project_2026-05-25_deerflow_sync_all_prs_merged.md` — 5-25 sync 全合 + 善后待办

## 总结

5-25 DeerFlow upstream sync 任务链完整闭环：

1. ✅ 上游 15 commit 全合入 dev — 5 PR (#33/#34/#35/#36/#38)
2. ✅ 测试 2989 passed / 19 skipped / 0 failed
3. ✅ 善后 PR push 完待用户合 — `chore/sync-protected-files-and-sop` HEAD `94faeed0`

下次 sync 时直接 `./scripts/sync-deerflow.sh --dry-run` 自动从 `f9b70713` 起算，22 个 PROTECTED_FILES 防住 5-21 翻车类问题。
