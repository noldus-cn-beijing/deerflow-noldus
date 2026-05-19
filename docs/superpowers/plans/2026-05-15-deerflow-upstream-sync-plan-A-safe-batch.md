# DeerFlow 上游同步 Plan A：第一批 11 个安全 commit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 deerflow 上游 2026-05-08 ~ 2026-05-14 之间 11 个低-中风险 bug fix / 性能改进 commit 用 surgical merge 方式合入本地 harness，保持 Noldus 所有定制不丢，并把上游配套的单测一并合入。

**Architecture:** 每个上游 commit 一个 task。每个 task 走完整的"读上游 diff → 区分安全文件 vs 受保护文件 → 安全文件整文件 checkout → 受保护文件手工融合 → 拉上游配套测试 → 跑 make test + make lint → 单独 commit"流程。频繁 commit，每个 commit 对应一个上游 PR，可独立回退。

**Tech Stack:** git surgical merge、pytest、ruff、本地 harness (`packages/agent/backend/packages/harness/deerflow/`)、上游 `deerflow/main`

**前置 / 范围外：**

- **不在本 plan 内**：13 个待评估 commit（DynamicContextMiddleware `c1b7f1d1` 改 lead_agent/prompt.py、`94da8f67` 改 serve.sh + docker/、`bedbf229` 改 mcp/tools.py、`30a58462` 改 sandbox/tools.py、`de253e4a` model_name 链路、4 个 dynamic_context 依赖修复、3 个建议跳过的 commit）——见 Plan B
- **基线**：本地 dev HEAD `fca62e33`，上游 `deerflow/main` tip `ba864112`（fetch 后取最新）
- **Tier 4 已吃下**：本地 `packages/agent/backend/packages/harness/deerflow/persistence/`、`runtime/user_context.py`、`runtime/checkpointer/` 均存在；CLAUDE.md 第 11 条 "v0.1 单用户不要 Tier 4" 是过期内容，第 12 条修正后实际现状是多用户研究助手

**关键参考文件（执行 agent 先读）：**

- `CLAUDE.md` § "DeerFlow fork 策略"（取长补短规则）、第 11/12 条（Tier 4 状态）
- `docs/sop/deerflow-sync-sop.md`（如存在）
- `scripts/sync-deerflow.sh`（不直接跑，只参考它的受保护文件列表）

**受保护文件列表（本 plan 多次引用，复制自 `scripts/sync-deerflow.sh` `PROTECTED_FILES`）：**

```
agents/lead_agent/prompt.py
agents/lead_agent/agent.py
agents/middlewares/llm_error_handling_middleware.py
agents/middlewares/thread_data_middleware.py
agents/thread_state.py
config/paths.py
mcp/tools.py
sandbox/tools.py
sandbox/sandbox.py
sandbox/local/local_sandbox.py
subagents/builtins/__init__.py
subagents/executor.py
tools/builtins/task_tool.py
```

路径前缀均为 `packages/agent/backend/packages/harness/deerflow/`。

**单元说明：**

- `UPSTREAM_PATH` = `backend/packages/harness/deerflow`（上游仓库内 harness 路径）
- `LOCAL_PATH` = `packages/agent/backend/packages/harness/deerflow`（本地仓库内 harness 路径）
- 上游 test 路径前缀：`backend/tests/`，本地 test 路径前缀：`packages/agent/backend/tests/`

---

## 全局准备（必须在 Task 1 之前完成）

### Step P.1：建专用 worktree

本会话有两个执行 agent 正在并行修改 frontend/ 和 backend/，必须用独立 worktree 避免冲突。

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow main
git worktree add .claude/worktrees/deerflow-sync-plan-A -b sync/deerflow-plan-A dev
cd .claude/worktrees/deerflow-sync-plan-A
```

确认在新 worktree 内：

```bash
pwd  # 期望: /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-A
git branch --show-current  # 期望: sync/deerflow-plan-A
git status --short  # 期望: 空
git log -1 --format='%H %s'  # 期望: fca62e33... 或 dev tip
```

### Step P.2：核实 baseline + 上游 tip

```bash
git rev-parse deerflow/main  # 记录到笔记
git log -1 --format='%H %ci %s' deerflow/main
```

### Step P.3：跑一次基线测试，确认起点是绿的

```bash
cd packages/agent/backend
source .venv/bin/activate  # 如果还没 activate
make test 2>&1 | tail -20
make lint 2>&1 | tail -10
```

**期望**：全绿。如果起点本来就有失败的测试，记录下来作为"pre-existing"基线，本 plan 不引入新失败即可。

### Step P.4：建 fix-log 文件

```bash
mkdir -p docs/handoffs/2026-05
touch docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-A-progress.md
```

每个 task 完成后追加一行进度，方便交接 / 中断后续接。

```markdown
# 2026-05-15 DeerFlow Sync Plan A 进度

| Task | Commit | 状态 | commit hash | 备注 |
|---|---|---|---|---|
| 1 | 2a1ac06b | ⏳ | | |
| 2 | 2eb11f97 | ⏳ | | |
...
```

---

## Task 1: `2a1ac06b` — persistence token usage 分组表达式复用

**上游 PR**：#2910 `fix(persistence): reuse token usage model grouping expression`

**改动范围**：纯 persistence 重构，把重复的 SQL 表达式提取成局部变量。无 Noldus 改动冲突。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/run/sql.py`（上游改 5 行）
- Test：`packages/agent/backend/tests/test_run_repository.py`（上游加 48 行）

- [ ] **Step 1.1：看上游 diff**

```bash
git show 2a1ac06b -- backend/packages/harness/deerflow/persistence/run/sql.py
git show 2a1ac06b -- backend/tests/test_run_repository.py
```

- [ ] **Step 1.2：confirm 本地 sql.py 与上游差异范围**

```bash
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/persistence/run/sql.py
```

预期：本地与上游已有差异，但本 commit 改动的那 5 行不与本地 Noldus 改动冲突。如果有冲突立刻 STOP 把发现写进 progress 表的备注列、拉用户判断。

- [ ] **Step 1.3：surgical merge sql.py**

读本地 `packages/agent/backend/packages/harness/deerflow/persistence/run/sql.py`，把上游 `2a1ac06b` 的 5 行改动手工合入对应位置。

具体改动的关键句：上游把内联的 `func.coalesce(...).label(...)` 表达式抽成局部变量 `grouping_expr` 然后在两处复用。在本地 sql.py 里找到对应函数（很可能是 `_token_usage_query` 或类似名），做同样抽取。

- [ ] **Step 1.4：合入配套测试**

```bash
git show deerflow/main:backend/tests/test_run_repository.py > /tmp/upstream-test_run_repository.py
diff /tmp/upstream-test_run_repository.py packages/agent/backend/tests/test_run_repository.py
```

- 如果本地 test_run_repository.py 已存在且本 commit 改动可直接 patch：把上游加的 48 行手工合入本地。
- 如果本地不存在：`cp /tmp/upstream-test_run_repository.py packages/agent/backend/tests/test_run_repository.py`

- [ ] **Step 1.5：跑测试**

```bash
cd packages/agent/backend
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

期望：全绿。新增的 token_usage 相关测试通过。

- [ ] **Step 1.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight  # 回到 worktree 根
git add packages/agent/backend/packages/harness/deerflow/persistence/run/sql.py \
        packages/agent/backend/tests/test_run_repository.py
git commit -m "$(cat <<'EOF'
sync(deerflow): persistence token usage 分组表达式复用（上游 #2910 / 2a1ac06b）

把内联 SQL 分组表达式提到局部变量，避免重复构造。
上游 PR: https://github.com/bytedance/deer-flow/pull/2910

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 1.7：更新 progress**

把 progress 表 Task 1 状态改为 ✅、填上 commit hash。

---

## Task 2: `2eb11f97` — runtime 持久化 run message summaries

**上游 PR**：#2850 `fix(runtime): persist run message summaries`

**改动范围**：`runtime/journal.py` 56 行 + 上游新测试 93 行。无 Noldus 改动冲突（本地 journal.py 与上游 DIFFERS，但本 commit 的改动区域应是新增方法）。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/journal.py`
- Test：`packages/agent/backend/tests/test_run_journal.py`

- [ ] **Step 2.1：看上游 diff**

```bash
git show 2eb11f97 -- backend/packages/harness/deerflow/runtime/journal.py
git show 2eb11f97 -- backend/tests/test_run_journal.py
```

- [ ] **Step 2.2：对比本地 journal.py 与上游**

```bash
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/runtime/journal.py
```

识别本地有哪些 Noldus 改动（如果有），确保 `2eb11f97` 的改动不与之冲突。如果冲突，STOP 上报。

- [ ] **Step 2.3：surgical merge journal.py**

把上游 `2eb11f97` 在 journal.py 的 56 行改动逐处合入本地版本，保留所有 Noldus 已有改动。

- [ ] **Step 2.4：合入配套测试**

```bash
git show deerflow/main:backend/tests/test_run_journal.py > /tmp/upstream-test_run_journal.py
diff /tmp/upstream-test_run_journal.py packages/agent/backend/tests/test_run_journal.py
```

把上游新增的 93 行测试合入本地（如本地已有该文件，patch；否则整文件 cp）。注意 Task 7 (`9892a7d4`) 还会再加一批 journal 测试，本 task 先合 2eb11f97 这部分。

- [ ] **Step 2.5：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_run_journal.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 2.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/runtime/journal.py \
        packages/agent/backend/tests/test_run_journal.py
git commit -m "$(cat <<'EOF'
sync(deerflow): runtime 持久化 run message summaries（上游 #2850 / 2eb11f97）

让 run message summaries 在 journal 层正确持久化，避免重启丢失。
上游 PR: https://github.com/bytedance/deer-flow/pull/2850

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2.7：更新 progress**

---

## Task 3: `f1a0ab69` — tools/tools.py 防止 tool_search promotions 在重入时丢失

**上游 PR**：#2885 `fix(tools): preserve tool_search promotions across re-entrant get_available_tools`

**改动范围**：`tools/tools.py` 53 行（safe file，非受保护）+ 612 行新测试 + `test_tool_deduplication.py` 12 行 patch。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/tools.py`
- Add: `packages/agent/backend/tests/test_deferred_tool_promotion_real_llm.py`（222 行新）
- Add: `packages/agent/backend/tests/test_deferred_tool_registry_promotion.py`（390 行新）
- Modify: `packages/agent/backend/tests/test_tool_deduplication.py`（12 行 patch）

- [ ] **Step 3.1：看上游 diff**

```bash
git show f1a0ab69 -- backend/packages/harness/deerflow/tools/tools.py
git show f1a0ab69 -- backend/tests/
```

- [ ] **Step 3.2：对比本地 tools.py 与上游**

```bash
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/tools/tools.py
```

注意：本地 `tools/tools.py` 是 ⚠ DIFFERS（与上游 tip 有差异），但**不在受保护列表里**。本 commit 的改动是 promotions 逻辑修复，预期跟本地 Noldus 改动正交。

- [ ] **Step 3.3：surgical merge tools.py**

把 `f1a0ab69` 在 tools.py 的 53 行改动手工合入本地 tools.py。保留本地已有改动。

- [ ] **Step 3.4：合入两个新测试 + patch test_tool_deduplication.py**

```bash
git show deerflow/main:backend/tests/test_deferred_tool_promotion_real_llm.py > packages/agent/backend/tests/test_deferred_tool_promotion_real_llm.py
git show deerflow/main:backend/tests/test_deferred_tool_registry_promotion.py > packages/agent/backend/tests/test_deferred_tool_registry_promotion.py
git show f1a0ab69 -- backend/tests/test_tool_deduplication.py | tail -30
```

最后一条命令读 dedup 测试的 patch hunk，手工合入本地 test_tool_deduplication.py 的对应位置。

- [ ] **Step 3.5：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_deferred_tool_promotion_real_llm.py tests/test_deferred_tool_registry_promotion.py tests/test_tool_deduplication.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

注意 `test_deferred_tool_promotion_real_llm.py` 含 "real_llm" 字样，可能依赖外部 LLM。如果它需要 API key 跑、本地没配置，应 skip 而不是 fail（看上游测试本身的 marker）。

- [ ] **Step 3.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/tools/tools.py \
        packages/agent/backend/tests/test_deferred_tool_promotion_real_llm.py \
        packages/agent/backend/tests/test_deferred_tool_registry_promotion.py \
        packages/agent/backend/tests/test_tool_deduplication.py
git commit -m "$(cat <<'EOF'
sync(deerflow): 修复 tool_search promotions 在 get_available_tools 重入时丢失（上游 #2885 / f1a0ab69）

修 promotions 在 deferred tool filter 链重入时被覆盖的 bug。
上游 PR: https://github.com/bytedance/deer-flow/pull/2885

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3.7：更新 progress**

---

## Task 4: `20d2d2b3` — dangling tool_call middleware 处理无效 tool_call

**上游 PR**：#2890 / #2891 `fix(middleware): Handle invalid tool calls in dangling pairing middleware`

**改动范围**：`agents/middlewares/dangling_tool_call_middleware.py` 83 行 + 50 行新测试。Safe file。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py`
- Modify or Add: `packages/agent/backend/tests/test_dangling_tool_call_middleware.py`

- [ ] **Step 4.1：看上游 diff**

```bash
git show 20d2d2b3 -- backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py
git show 20d2d2b3 -- backend/tests/test_dangling_tool_call_middleware.py
```

- [ ] **Step 4.2：对比本地版本**

```bash
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py
```

- [ ] **Step 4.3：surgical merge**

把 `20d2d2b3` 在该 middleware 的 83 行改动合入本地，保留本地改动。

- [ ] **Step 4.4：合入测试**

```bash
git show deerflow/main:backend/tests/test_dangling_tool_call_middleware.py > /tmp/upstream-test_dangling.py
diff /tmp/upstream-test_dangling.py packages/agent/backend/tests/test_dangling_tool_call_middleware.py 2>/dev/null || cp /tmp/upstream-test_dangling.py packages/agent/backend/tests/test_dangling_tool_call_middleware.py
```

或手工合入新增 50 行测试。

- [ ] **Step 4.5：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_dangling_tool_call_middleware.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 4.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py \
        packages/agent/backend/tests/test_dangling_tool_call_middleware.py
git commit -m "$(cat <<'EOF'
sync(deerflow): dangling tool_call middleware 处理无效 tool_call（上游 #2890+#2891 / 20d2d2b3）

middleware 现在能优雅处理 tool_call schema 不合法的边界场景。
上游 PR: https://github.com/bytedance/deer-flow/pull/2890

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4.7：更新 progress**

---

## Task 5: `2b5bece7` — local sandbox singleton 跟 provider lifecycle reset

**上游 PR**：#2834 `fix(harness): reset local sandbox singleton with provider lifecycle`

**改动范围**：`sandbox/local/local_sandbox_provider.py` 10 行 + `sandbox/sandbox_provider.py` 13 行 + 145 行新测试。两文件本地均 ⚠ DIFFERS 但不在受保护列表（`local_sandbox.py` 是受保护的，但本 commit 不动它）。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/sandbox/sandbox_provider.py`
- Modify or Add: `packages/agent/backend/tests/test_local_sandbox_provider_mounts.py`

- [ ] **Step 5.1：看上游 diff**

```bash
git show 2b5bece7 -- backend/packages/harness/deerflow/sandbox/
git show 2b5bece7 -- backend/tests/test_local_sandbox_provider_mounts.py
```

- [ ] **Step 5.2：对比本地 + surgical merge**

```bash
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/sandbox/local/local_sandbox_provider.py
git diff fca62e33 deerflow/main -- backend/packages/harness/deerflow/sandbox/sandbox_provider.py
```

把上游 `2b5bece7` 的改动合入两个 provider 文件。重点：上游加了在 provider 生命周期重置时清空 singleton 的逻辑，避免测试 leak。本地若已有相关 lifecycle 改动注意叠加。

- [ ] **Step 5.3：合入测试**

```bash
git show deerflow/main:backend/tests/test_local_sandbox_provider_mounts.py > /tmp/upstream-test_local_sandbox.py
[ -e packages/agent/backend/tests/test_local_sandbox_provider_mounts.py ] && diff /tmp/upstream-test_local_sandbox.py packages/agent/backend/tests/test_local_sandbox_provider_mounts.py || cp /tmp/upstream-test_local_sandbox.py packages/agent/backend/tests/test_local_sandbox_provider_mounts.py
```

- [ ] **Step 5.4：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_local_sandbox_provider_mounts.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 5.5：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox_provider.py \
        packages/agent/backend/packages/harness/deerflow/sandbox/sandbox_provider.py \
        packages/agent/backend/tests/test_local_sandbox_provider_mounts.py
git commit -m "$(cat <<'EOF'
sync(deerflow): local sandbox singleton 跟随 provider 生命周期 reset（上游 #2834 / 2b5bece7）

测试场景下避免 sandbox singleton 跨 provider 实例泄漏。
上游 PR: https://github.com/bytedance/deer-flow/pull/2834

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5.6：更新 progress**

---

## Task 6: `e9deb6c2` — thread metadata 过滤推到 SQL（性能）

**上游 PR**：#2865 `perf(harness): push thread metadata filters into SQL`

**改动范围**：5 个 persistence 文件 + 一个新文件 `json_compat.py`（195 行新）+ 大量测试改动（558 行）+ gateway `threads.py` 路由 38 行。本批跨 backend/app/gateway 边界，注意确认 router 改动跟本地 Noldus auth/feedback 等改动不冲突。

**Files：**
- Add: `packages/agent/backend/packages/harness/deerflow/persistence/json_compat.py`（新建，195 行整文件拉）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/__init__.py`（3 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/base.py`（9 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/memory.py`（4 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/sql.py`（46 行）
- Modify: `packages/agent/backend/app/gateway/routers/threads.py`（38 行 — **不在 harness 路径下，跨 Noldus 应用层**）
- Modify or Add: `packages/agent/backend/tests/test_thread_meta_repo.py`（504 行新增 / 改）
- Modify or Add: `packages/agent/backend/tests/test_threads_router.py`（54 行）

- [ ] **Step 6.1：看上游 diff（全 8 文件）**

```bash
git show e9deb6c2 --stat
git show e9deb6c2 -- backend/app/gateway/routers/threads.py
git show e9deb6c2 -- backend/packages/harness/deerflow/persistence/
git show e9deb6c2 -- backend/tests/test_thread_meta_repo.py | head -150
```

- [ ] **Step 6.2：⚠ 先检查 `app/gateway/routers/threads.py` 是否被本地 Noldus 改过**

```bash
# 这个文件不在 harness path 下，但上游也会改它，需要确认本地 Noldus 改动
git log --oneline -- packages/agent/backend/app/gateway/routers/threads.py | head -10
```

如果本地有 Noldus 改动（如 better-auth 集成、@require_permission、租户隔离等），surgical merge 时务必保留它们。如果冲突太大，STOP 上报让用户决定是否拆出 router 改动单独做。

- [ ] **Step 6.3：整文件拉 `json_compat.py`**

```bash
git show deerflow/main:backend/packages/harness/deerflow/persistence/json_compat.py > packages/agent/backend/packages/harness/deerflow/persistence/json_compat.py
```

- [ ] **Step 6.4：surgical merge 4 个 persistence 文件**

逐文件合入。`thread_meta/__init__.py` 多半是 import json_compat；`thread_meta/base.py` 加新方法签名；`thread_meta/memory.py` 实现 in-memory 的新过滤；`thread_meta/sql.py` 把 Python 端的过滤改成 SQL where 子句。

- [ ] **Step 6.5：surgical merge gateway `threads.py`**

把上游 38 行改动手工合入本地版本。注意保留所有 Noldus auth/permission/上下文注入逻辑。

- [ ] **Step 6.6：合入测试**

```bash
git show deerflow/main:backend/tests/test_thread_meta_repo.py > /tmp/upstream-test_thread_meta.py
git show deerflow/main:backend/tests/test_threads_router.py > /tmp/upstream-test_threads_router.py
diff /tmp/upstream-test_thread_meta.py packages/agent/backend/tests/test_thread_meta_repo.py | head -50
diff /tmp/upstream-test_threads_router.py packages/agent/backend/tests/test_threads_router.py | head -50
```

把上游新加 / 修改的测试用例合入本地，保留本地 Noldus 测试场景。

- [ ] **Step 6.7：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_thread_meta_repo.py tests/test_threads_router.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 6.8：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/persistence/json_compat.py \
        packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/*.py \
        packages/agent/backend/app/gateway/routers/threads.py \
        packages/agent/backend/tests/test_thread_meta_repo.py \
        packages/agent/backend/tests/test_threads_router.py
git commit -m "$(cat <<'EOF'
sync(deerflow): thread metadata 过滤推到 SQL（perf，上游 #2865 / e9deb6c2）

把原本在 Python 端做的 thread metadata 过滤推到 SQL where 子句，
新增 json_compat.py 处理 SQLite / Postgres JSON 函数差异。
上游 PR: https://github.com/bytedance/deer-flow/pull/2865

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6.9：更新 progress**

---

## Task 7: `7caf03e9` — Postgres extra for store/checkpointer

**上游 PR**：#2584 `fix(packaging): add postgres extra for store/checkpointer support`

**改动范围**：4 个 provider/config 文件各 4-5 行 + 测试。改动小，单纯让 postgres 作为可选 extra 被 store/checkpointer 正确识别。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/config/checkpointer_config.py`（5 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/engine.py`（4 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/provider.py`（4 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/store/provider.py`（4 行）
- Modify or Add: `packages/agent/backend/tests/test_checkpointer.py`（42 行）
- Modify: `packages/agent/backend/tests/test_persistence_scaffold.py`（15 行）

- [ ] **Step 7.1：看上游 diff**

```bash
git show 7caf03e9 --stat
git show 7caf03e9 -- backend/packages/harness/deerflow/
```

- [ ] **Step 7.2：surgical merge 4 个 provider/config 文件**

每个文件改动量都很小（4-5 行），手工合入。重点：上游在判断 backend 字符串时加了对 postgres extra 的识别（如 `if "postgres" in backend_url and not _has_postgres_extra(): raise ...`）。

- [ ] **Step 7.3：合入测试**

```bash
git show deerflow/main:backend/tests/test_checkpointer.py > /tmp/upstream-test_checkpointer.py
git show 7caf03e9 -- backend/tests/test_persistence_scaffold.py
```

合入两个测试文件的改动。

- [ ] **Step 7.4：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_checkpointer.py tests/test_persistence_scaffold.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 7.5：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/config/checkpointer_config.py \
        packages/agent/backend/packages/harness/deerflow/persistence/engine.py \
        packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/provider.py \
        packages/agent/backend/packages/harness/deerflow/runtime/store/provider.py \
        packages/agent/backend/tests/test_checkpointer.py \
        packages/agent/backend/tests/test_persistence_scaffold.py
git commit -m "$(cat <<'EOF'
sync(deerflow): postgres extra 支持 store/checkpointer（上游 #2584 / 7caf03e9）

让 store/checkpointer 在 backend_url 是 postgres 时正确识别需要安装 postgres extra。
上游 PR: https://github.com/bytedance/deer-flow/pull/2584

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7.6：更新 progress**

---

## Task 8: `9892a7d4` — subagent token usage 归并到 parent run（先于 Task 9）

**上游 PR**：#2838 `fix: bucket subagent token usage into parent run totals`

**⚠ 顺序硬约束**：必须早于 Task 9（`eab7ae3d`）做。两者都改 `task_tool.py`，`eab7ae3d` 假设 `9892a7d4` 已在。

**改动范围**：4 个 harness 文件 + 3 个测试 + 1 个前端测试。其中 `task_tool.py` 和 `executor.py` 是 🛡 受保护文件，本地有 Noldus 改动（task_tool 加了 recursion_limit + max_turns 硬限制，executor 同样）。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/journal.py`（68 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`（🛡 16 行）
- Add: `packages/agent/backend/packages/harness/deerflow/subagents/token_collector.py`（新建 63 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（🛡 141 行）
- Modify or Add: `packages/agent/backend/tests/test_run_journal.py`（238 行新增）
- Add: `packages/agent/backend/tests/test_subagent_token_collector.py`（新建 161 行）
- Modify: `packages/agent/backend/tests/test_task_tool_core_logic.py`（227 行）
- Modify: `packages/agent/frontend/tests/unit/core/threads/api.test.ts`（6 行小 patch）

- [ ] **Step 8.1：看上游 diff（重点看两个受保护文件）**

```bash
git show 9892a7d4 --stat
git show 9892a7d4 -- backend/packages/harness/deerflow/tools/builtins/task_tool.py
git show 9892a7d4 -- backend/packages/harness/deerflow/subagents/executor.py
```

- [ ] **Step 8.2：识别本地受保护文件的 Noldus 改动**

```bash
git diff f0dd8cb HEAD -- packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py
git diff f0dd8cb HEAD -- packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

确认本地 Noldus 在这两个文件里的改动：
- `task_tool.py`：可能有 max_turns 硬限制、{{shared://}} 占位符支持、handoff_suffix 等
- `executor.py`：recursion_limit 修复 + max_turns 硬限制

surgical merge 时这些 Noldus 改动 **必须保留**。

- [ ] **Step 8.3：整文件拉 `token_collector.py`（新文件）**

```bash
git show deerflow/main:backend/packages/harness/deerflow/subagents/token_collector.py > packages/agent/backend/packages/harness/deerflow/subagents/token_collector.py
```

- [ ] **Step 8.4：surgical merge `journal.py`（68 行）**

把上游 9892a7d4 在 journal.py 的改动合入本地（注意：本 task 跟 Task 2 都改 journal.py，Task 2 已经先合过；这次叠加 9892a7d4 的改动）。

- [ ] **Step 8.5：surgical merge 🛡 `executor.py`**

上游 9892a7d4 加了 16 行处理 token_collector 的逻辑。手工把这 16 行嵌入本地 executor.py 的对应位置，**保留本地 recursion_limit + max_turns 改动**。

- [ ] **Step 8.6：surgical merge 🛡 `task_tool.py`（141 行 — 本批最大的受保护文件改动）**

这个 commit 在 task_tool.py 上加了 141 行涉及 token bucketing 的逻辑。需要：
1. 读本地 task_tool.py 完整内容
2. 读上游 9892a7d4 在该文件的 diff（141 行）
3. 把改动逐处合入对应位置，保留本地 max_turns 硬限制 + Noldus 中文化注释 + {{shared://}} 占位符等

如果合到一半发现冲突无法判断，STOP，把冲突点写进 progress 表的备注列让用户决定。

- [ ] **Step 8.7：合入 3 个 backend 测试 + 1 个 frontend 测试 patch**

```bash
git show deerflow/main:backend/tests/test_subagent_token_collector.py > packages/agent/backend/tests/test_subagent_token_collector.py
git show 9892a7d4 -- backend/tests/test_run_journal.py | head -100
git show 9892a7d4 -- backend/tests/test_task_tool_core_logic.py | head -100
git show 9892a7d4 -- frontend/tests/unit/core/threads/api.test.ts
```

手工合入测试 patch（test_run_journal.py + test_task_tool_core_logic.py 是 patch 而非整文件；test_subagent_token_collector.py 是新文件）。前端 6 行 patch 合入 `packages/agent/frontend/tests/unit/core/threads/api.test.ts`。

- [ ] **Step 8.8：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_subagent_token_collector.py tests/test_run_journal.py tests/test_task_tool_core_logic.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10

cd ../frontend
pnpm test tests/unit/core/threads/api.test.ts 2>&1 | tail -20
```

- [ ] **Step 8.9：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/runtime/journal.py \
        packages/agent/backend/packages/harness/deerflow/subagents/executor.py \
        packages/agent/backend/packages/harness/deerflow/subagents/token_collector.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/tests/test_run_journal.py \
        packages/agent/backend/tests/test_subagent_token_collector.py \
        packages/agent/backend/tests/test_task_tool_core_logic.py \
        packages/agent/frontend/tests/unit/core/threads/api.test.ts
git commit -m "$(cat <<'EOF'
sync(deerflow): subagent token usage 归并到 parent run（上游 #2838 / 9892a7d4）

新增 token_collector 把 subagent 的 token 用量归集到 parent run 总量，
便于飞轮训练数据按整次会话计费。保留本地 task_tool/executor 的 Noldus 改动。
上游 PR: https://github.com/bytedance/deer-flow/pull/2838

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8.10：更新 progress**

---

## Task 9: `eab7ae3d` — subagent token 用量流到 header（terminal task events）

**上游 PR**：#2882 `feat: stream subagent token usage to header via terminal task events`

**⚠ 顺序硬约束**：必须晚于 Task 8。

**改动范围**：`token_usage_middleware.py` 61 行 + 🛡 `task_tool.py` 55 行 + 测试 + 前端 3 文件。**前端改动属于 G4 方案 C agent 的边界**，需要先确认 G4 方案 C agent 是否还在跑（如果未完成，本 task 的前端 3 个文件改动要等 G4 完成后再合，避免冲突）。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/token_usage_middleware.py`（61 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（🛡 55 行）
- Modify: `packages/agent/backend/tests/test_memory_queue_user_isolation.py`（5 行小 patch）
- Modify or Add: `packages/agent/backend/tests/test_task_tool_core_logic.py`（153 行）
- Modify: `packages/agent/backend/tests/test_token_usage_middleware.py`（49 行）
- Modify: `packages/agent/frontend/src/app/workspace/messages/message-token-usage.tsx`（41 行）
- Modify: `packages/agent/frontend/src/core/messages/usage.ts`（4 行）
- Modify: `packages/agent/frontend/src/core/threads/hooks.ts`（18 行）

- [ ] **Step 9.1：先确认 G4 方案 C agent 状态**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline origin/dev | grep -iE "G4|stage broadcast|frontend stage" | head -5
```

如果 G4 方案 C agent 已经把它的 commit push 到 origin/dev：本 task 把前端 3 个文件改动合入时要在它的基础上叠加。
如果 G4 方案 C agent **还在跑**：STOP，把本 task 的前端部分挂起，回主会话上报，让主会话调度。

- [ ] **Step 9.2：看上游 diff**

```bash
git show eab7ae3d --stat
git show eab7ae3d -- backend/packages/harness/deerflow/
git show eab7ae3d -- backend/tests/
git show eab7ae3d -- frontend/
```

- [ ] **Step 9.3：surgical merge `token_usage_middleware.py`（61 行）**

合入 token_usage 中间件的改动。本地该文件 ⚠ DIFFERS 但不在受保护列表。

- [ ] **Step 9.4：surgical merge 🛡 `task_tool.py`（55 行 — 叠加在 Task 8 之上）**

把上游 eab7ae3d 在 task_tool.py 的 55 行改动合入。注意这是叠加在 Task 8 已合入的 141 行之上，需要仔细看上游 diff 的 base 是 9892a7d4 之后的版本。

- [ ] **Step 9.5：合入 backend 测试**

3 个 backend 测试文件的 patch 合入。

- [ ] **Step 9.6：合入 frontend 3 个文件**

如 Step 9.1 确认 G4 方案 C 已完成：

```bash
git show eab7ae3d -- frontend/src/app/workspace/messages/message-token-usage.tsx
git show eab7ae3d -- frontend/src/core/messages/usage.ts
git show eab7ae3d -- frontend/src/core/threads/hooks.ts
```

手工合入 3 个前端文件的改动。注意叠加在 G4 方案 C 之后的状态上。

- [ ] **Step 9.7：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_token_usage_middleware.py tests/test_task_tool_core_logic.py tests/test_memory_queue_user_isolation.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10

cd ../frontend
pnpm check 2>&1 | tail -20
```

- [ ] **Step 9.8：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/token_usage_middleware.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/tests/test_memory_queue_user_isolation.py \
        packages/agent/backend/tests/test_task_tool_core_logic.py \
        packages/agent/backend/tests/test_token_usage_middleware.py \
        packages/agent/frontend/src/app/workspace/messages/message-token-usage.tsx \
        packages/agent/frontend/src/core/messages/usage.ts \
        packages/agent/frontend/src/core/threads/hooks.ts
git commit -m "$(cat <<'EOF'
sync(deerflow): subagent token usage 通过 terminal task events 流到 header（上游 #2882 / eab7ae3d）

subagent 跑完后，把它的 token 用量通过 terminal task events 推到前端，
header 上能实时看到累计开销，飞轮训练材料更完整。
上游 PR: https://github.com/bytedance/deer-flow/pull/2882

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9.9：更新 progress**

---

## Task 10: `813d3c94` — system_prompt + skills 合并成单 SystemMessage

**上游 PR**：#2701 `fix(subagents): consolidate system_prompt and skills into single SystemMessage`

**改动范围**：`subagents/config.py` 2 行 + 🛡 `subagents/executor.py` 19 行 + 184 行测试。executor.py 受保护文件叠加修改。

**价值**：把 system_prompt 和 skills 合成单条 SystemMessage 后，对 Anthropic prefix cache 更友好；本地 token 节省可观。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/config.py`（2 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`（🛡 19 行 — 叠加在 Task 8 之上）
- Modify: `packages/agent/backend/tests/test_subagent_executor.py`（184 行测试新增 / 改）

- [ ] **Step 10.1：看上游 diff**

```bash
git show 813d3c94 --stat
git show 813d3c94 -- backend/packages/harness/deerflow/subagents/
git show 813d3c94 -- backend/tests/test_subagent_executor.py | head -80
```

- [ ] **Step 10.2：surgical merge config.py（2 行）**

本地该文件本来跟上游 DIFFERS，但改动是改默认参数 / 类型注解级的小动作。

- [ ] **Step 10.3：surgical merge 🛡 `executor.py`（19 行）**

把上游 19 行合并到本地 executor.py，叠加在 Task 8 已合入的 16 行之上，并保留 Noldus 的 recursion_limit + max_turns。

- [ ] **Step 10.4：合入测试**

```bash
git show 813d3c94 -- backend/tests/test_subagent_executor.py | head -200
```

手工合入 184 行测试改动到本地 `test_subagent_executor.py`。

- [ ] **Step 10.5：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_subagent_executor.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step 10.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/subagents/config.py \
        packages/agent/backend/packages/harness/deerflow/subagents/executor.py \
        packages/agent/backend/tests/test_subagent_executor.py
git commit -m "$(cat <<'EOF'
sync(deerflow): subagent system_prompt + skills 合并成单 SystemMessage（prefix cache 友好，上游 #2701 / 813d3c94）

subagent 现在把 system_prompt 和 skills 合并成一条 SystemMessage，
对 Anthropic prefix cache 更友好，缓存命中率提高、token 成本降低。
上游 PR: https://github.com/bytedance/deer-flow/pull/2701

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 10.7：更新 progress**

---

## Task 11: `7de9b582` — 引入 Runtime type alias，消除 Pydantic 序列化警告

**上游 PR**：#2774 `fix(tools): introduce Runtime type alias to eliminate Pydantic serialization warning`

**改动范围**：8 个文件，但都是机械的类型替换（把 `Runtime[Any, Any]` 之类的写法换成新 alias）。其中 🛡 `sandbox/tools.py`、🛡 `task_tool.py` 受保护。新增 `tools/types.py`（11 行 alias 定义）。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`（🛡 32 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/present_file_tool.py`（11 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py`（4 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（🛡 7 行 — 叠加在 Task 9 之上）
- Add: `packages/agent/backend/packages/harness/deerflow/tools/builtins/update_agent_tool.py`（4 行 — 本地 ❌ MISSING，需要先确认是否引入这个 tool）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/view_image_tool.py`（8 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py`（11 行）
- Add: `packages/agent/backend/packages/harness/deerflow/tools/types.py`（新建 11 行）
- Add: `packages/agent/backend/tests/test_tool_args_schema_no_pydantic_warning.py`（新建 91 行）

- [ ] **Step 11.1：判断 `update_agent_tool.py` 是否需要引入**

```bash
ls packages/agent/backend/packages/harness/deerflow/tools/builtins/update_agent_tool.py 2>&1
grep -rn "update_agent_tool\|update_agent" packages/agent/backend/packages/harness/deerflow/ --include="*.py" | head
```

本地若一直没有 update_agent_tool.py 且无 import 引用，本 commit 对它的 4 行改动应**跳过**（不引入新 tool）。如果发现有引用、是 Tier 4 之后该有的、只是当时没合：整文件拉。

- [ ] **Step 11.2：整文件拉 `tools/types.py`（新文件）**

```bash
git show deerflow/main:backend/packages/harness/deerflow/tools/types.py > packages/agent/backend/packages/harness/deerflow/tools/types.py
```

- [ ] **Step 11.3：看上游 diff**

```bash
git show 7de9b582 --stat
git show 7de9b582 -- backend/packages/harness/deerflow/sandbox/tools.py
git show 7de9b582 -- backend/packages/harness/deerflow/tools/builtins/task_tool.py
```

- [ ] **Step 11.4：surgical merge 🛡 `sandbox/tools.py`（32 行）**

上游改动是把 32 处 `Runtime[Any, Any]` 类型注解替换为新 alias。机械替换，保留本地 Noldus 的 {{shared://}} 占位符、`extra_env` 等改动。

- [ ] **Step 11.5：surgical merge 🛡 `task_tool.py`（7 行 — 叠加在 Task 9 之上）**

7 行类型替换，叠加在前面任务之上。

- [ ] **Step 11.6：surgical merge 4 个非受保护 tools**

`present_file_tool.py`、`setup_agent_tool.py`、`view_image_tool.py`、`skill_manage_tool.py` —— 各自合入对应行数的类型替换。

- [ ] **Step 11.7：合入测试**

```bash
git show deerflow/main:backend/tests/test_tool_args_schema_no_pydantic_warning.py > packages/agent/backend/tests/test_tool_args_schema_no_pydantic_warning.py
```

新文件直接整拉。

- [ ] **Step 11.8：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_tool_args_schema_no_pydantic_warning.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

期望：之前可能出现的 Pydantic serialization warning 消失；新测试通过。

- [ ] **Step 11.9：commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/sandbox/tools.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/present_file_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/view_image_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/types.py \
        packages/agent/backend/tests/test_tool_args_schema_no_pydantic_warning.py
git commit -m "$(cat <<'EOF'
sync(deerflow): 引入 Runtime type alias 消除 Pydantic 序列化警告（上游 #2774 / 7de9b582）

新增 tools/types.py 提供 Runtime 别名，所有 builtin tool 替换 Runtime[Any, Any]
为新 alias，消除运行时 Pydantic serialization warning。机械类型替换，保留 Noldus 定制。
上游 PR: https://github.com/bytedance/deer-flow/pull/2774

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11.10：更新 progress**

---

## 收尾（所有 Task 完成后）

### Step F.1：跑完整 dogfood 端到端验证

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-A/packages/agent
make stop && make dev
# 等 5-10 秒让所有服务起来
```

然后用 Playwright 跑一次完整 EPM 单只分析 / 任意一个 paradigm，验证：
- code-executor 能正常 dispatch
- subagent 不报错
- 前端 header 能看到 token 用量（Task 9 的收益）
- 没有 Pydantic serialization warning（Task 11 的收益）

如果跑不通，回头看 progress 表，按 task 倒序逐个 revert 直到找到回归来源。

### Step F.2：跑完整测试套件

```bash
cd packages/agent/backend
make test 2>&1 | tail -50
make lint 2>&1 | tail -20

cd ../frontend
pnpm check 2>&1 | tail -20
```

期望：全绿。

### Step F.3：归档进度文档

把 `docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-A-progress.md` 补完整，添加：

- 总耗时
- 每个 commit 实际遇到的 surgical merge 难点
- 跑测试的关键发现（如某个测试需要 LLM API key 跳过、某个测试上游本来就慢）
- 哪些 commit 是 Plan A 没碰、留给 Plan B 的

```bash
git add docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-A-progress.md
git commit -m "docs: deerflow sync plan A 完成交接（11 个 commit）"
```

### Step F.4：把分支推到 origin（等用户授权）

**⚠ 不要主动 push**。完成所有 task 后，给主会话发消息：

> Plan A 全 11 个 commit 已落 sync/deerflow-plan-A 分支。本地 make test + make lint 全绿，dogfood 端到端验证通过。等用户授权后 merge 进 dev / push origin。

不要自己 push、不要 fast-forward dev。

### Step F.5：退出 worktree

退出执行 agent 自身的 worktree（保留分支，等待用户 merge）：

```bash
cd /home/wangqiuyang/noldus-insight
# 不要 remove worktree，把分支留着等用户合
git worktree list
```

---

## 风险与中断恢复

### 风险 1：surgical merge 时受保护文件冲突难判断

**症状**：上游改动跟 Noldus 改动在同一行 / 同一函数。

**处理**：STOP，不要瞎合。把冲突点写进 progress 表的备注列：

```markdown
| 8 | 9892a7d4 | ⚠ BLOCKED | | task_tool.py 第 234-256 行：上游加 token bucketing 入口，本地这段是 max_turns 检查，无法判断是否要把 max_turns 检查移到 token bucketing 之后 |
```

然后向主会话上报，让用户决定。**不要自己拍板**。

### 风险 2：测试在某个 commit 后变红，但下个 commit 应该会修

**症状**：Task N 跑完 `make test` 出现 1-2 个 fail，但 fail 的测试在 Task N+1 改动范围内。

**处理**：先 commit Task N 本身的改动并跑过它配套测试；在 progress 表标注 "测试 X 暂红，等 Task N+1"。Task N+1 跑完应当转绿。

### 风险 3：G4 方案 C agent 还在跑、Task 9 前端部分撞车

**症状**：Step 9.1 检查发现 G4 还没 push。

**处理**：把 Task 9 拆成两半：
- 9a：只合 backend 部分（middleware + task_tool + 3 个 backend 测试），commit
- 9b：等 G4 完成后再合 frontend 3 个文件，单独 commit

不要等 G4，跳过前端继续做 Task 10、11；最后回头补 9b。

### 风险 4：Spec 阶段 1 agent 也在改 task_tool.py / handoff 相关

**症状**：Spec 阶段 1 agent 在跑的 plan（`2026-05-15-spec-phase-1-dual-layer-handoff-plan.md`）改了 task_tool.py、4 个 subagent。

**处理**：本 plan 也改 task_tool.py、executor.py（Task 8/9/10/11）。**Spec 阶段 1 优先级更高**，本 plan 在专用 worktree 跑没问题，但合回 dev 时如果 Spec 阶段 1 已经 push，需要 rebase 本分支到 Spec 阶段 1 之后再 push。**不要 force push**。

### 中断恢复

如果跑到一半中断：

1. 读 `docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-A-progress.md` 看到哪了
2. `cd .claude/worktrees/deerflow-sync-plan-A`
3. `git log --oneline | head -20` 看哪些 task 已 commit
4. 从下一个 ⏳ 的 task 接着做

---

## Plan A 完成后下一步

向主会话回报，并 flag：

- Plan B（13 个剩余 commit）什么时候做？建议等 Plan A 验证稳定 + DynamicContextMiddleware（c1b7f1d1）单独立 plan 后再开
- 是否需要更新 `scripts/sync-deerflow.sh` 的 baseline tracking 机制（当前因为大量 cherry-pick 已经失效）
- 是否需要更新 `CLAUDE.md` 第 11 条（已被第 12 条修正但原文仍在）和 `docs/sop/deerflow-sync-sop.md` 第 145 行（"Tier 4 整 PR 跳过"已过期）
