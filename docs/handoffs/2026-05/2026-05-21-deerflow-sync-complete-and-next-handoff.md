# 2026-05-21 DeerFlow 上游同步完成 + 下一步交接

## 当前任务目标

**主要任务（已完成）**：把 DeerFlow fork 从 `f0dd8cb` 同步到上游最新 `e19bec1`（合 150 个上游 commit）。✅ 已合入 dev（PR #23）。

**下一步任务（待启动）**：
1. **ReadCacheMiddleware** — 解决"agent 反复读同一文件造成 loop closed"问题（用户明确要求做、设计已对齐）
2. **thinking 慢问题诊断** — 等用户发 langgraph.log 看根因（可能是过度思考、反复决策、或 context 初始化慢）

## 当前进展

### ✅ 已完成（本次会话）

1. **PR #23 (sync) 已 merge 到 dev**，HEAD 现在是 `2b5100e8`。包含 3 个 commit：
   - `598667e6`: 主 sync — 适配上游 f0dd8cb..e19bec1 (150 commits)，修了 7 个根因（R1 config.yaml YAML typo / R2 LoopDetectionMiddleware.from_config / R3 Paths.user_agent_dir/user_agents_dir / R4 multi-user acp-workspace 测试对齐 / R5 build_lead_runtime_middlewares 传 app_config + LLMErrorHandlingMiddleware 接 app_config kwarg / R6 prompt 补"范式推断失败"段 / R7 cherry-pick bf5607b9 patched_reasoning）
   - `367795fb`: A 修 task_tool polling timeout 漏 cancel（上游 e19bec14 修复点）+ C 修 executor atexit 重复注册
   - `5b4688a7`: B 加 try_set_terminal CAS 状态机（race 防护，9 处赋值点重构）+ D 加 ResolvedPath NamedTuple（local_sandbox 单次解析）

2. **PR #22 (ev19) 也已 merge 到 dev**（用户操作），包含 `bf5607b9` (patched_reasoning) + `40b695de` (memory user_id 穿透 fix)。

3. **测试**：2677 passed / 17 skipped / 1 failed
   - 唯一失败 `test_memory_router::test_update_memory_fact_route_preserves_omitted_fields` 是 **dev 上预存在 bug**（40b695de 改 router 后没更新测试断言），跟 sync 无关，请勿误判为本会话引入

4. **5 个老 worktree 清理完毕**：deerflow-upstream-sync / fix+ev19-template-identify-tool / lead-visibility-fix / stepwise-gate / subagent-role-split-impl，仓库现在干净

## 关键上下文

### 仓库现状

- **主分支** dev = `2b5100e8`（远程已同步）
- **没有 active worktree**，主仓 `/home/wangqiuyang/noldus-insight` 干净
- **剩余孤立 worktree-* 分支引用**：`worktree-metric-catalog-implementation`、`worktree-spec-phase-1-handoff`（无对应目录，不影响）

### DeerFlow Sync 结论（基于 head-to-head 对比）

| 上游 vs Noldus 差异 | 状态 |
|---|---|
| 126 安全文件 + 12 新文件 | ✅ 已合入 |
| llm_error_handling / paths / mcp/tools / sandbox/* 改进 | ✅ Noldus 早有同等或更好版本 |
| try_set_terminal CAS 状态机 (B) | ✅ 已 surgical merge |
| ResolvedPath NamedTuple (D) | ✅ 已 surgical merge |
| task_tool polling timeout cancel (A) | ✅ 已 surgical merge |
| atexit unregister (C) | ✅ 已 surgical merge |
| prompt.py / agent.py / executor.py 中文业务 prompt | ❌ 故意保留 Noldus 定制（CLAUDE.md 第 13 条原则）|

**结论**：sync session **100% 完成**，dev 已经"完全跟上上游 e19bec1 的可合入部分"。

### 重要文件路径

| 用途 | 路径 |
|---|---|
| 项目根 CLAUDE.md | `/home/wangqiuyang/noldus-insight/CLAUDE.md` |
| Backend CLAUDE.md | `/home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md` |
| LoopDetectionMiddleware | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` |
| 中间件链定义 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:_build_middlewares` |
| Sandbox tools (read_file 等) | `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` |
| Local sandbox | `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py` |
| Subagent executor | `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` |
| config.yaml (gitignored) | `/home/wangqiuyang/noldus-insight/packages/agent/config.yaml` |

## 关键发现

### 1. CLAUDE.md 第 13 条（项目状态修正）已生效

仓库已经吃下 Tier 4 体系（multi-user persistence、`@require_permission`、`get_effective_user_id`、UserRow 等）。**测试需要 conftest.py 注入 `test-user-autouse` 才能跑 multi-user 相关路径**。R4 的 acp-workspace 测试修复就是基于这个事实。

### 2. memory updater.py 走"委托 sync" 路径

上游 `aupdate_memory` 现在通过 `asyncio.to_thread(_do_update_memory_sync)` 委托到同步路径，避免 cross-loop httpx 连接池复用 bug（issue #2615）。dev 上之前是直接 async（`model.ainvoke`），rebase 时被改成委托 sync（保留 40b695de 的 user_id 透传）。**这是有意保留的方向**，下一任不要"修回"async。

### 3. SubagentStatus.is_terminal 必须用 .value 比较而非 enum identity

测试 fixture 用 `importlib.reload(executor)` 重新加载模块，会创建新的 SubagentStatus enum 类。`is_terminal` 内部用 `self in {SubagentStatus.X, ...}` 会失败（两个不同的 enum 类）。**已改成 value 字符串比较**（executor.py 的 `is_terminal` property），不要再回到 identity 比较。

### 4. patched_reasoning.py 已在 dev 上

不要再尝试 cherry-pick `bf5607b9`（patched_reasoning）—— 已经在 dev 上了。如果在新 worktree 创建时 base 是 origin/dev，就自动有。

### 5. dev 上预存在测试 fail（非本会话引入）

`test_memory_router::test_update_memory_fact_route_preserves_omitted_fields` 在 dev 上 fail。原因：40b695de 改了 `update_fact()` 路径加 `user_id` 参数，但测试断言没更新。如果下一任跑测试看到 1 个 fail / 2677 passed，**这就是它，不要花时间排查**。要修的话改测试 `tests/test_memory_router.py:261` 的 `assert_called_once_with(...)` 加 `user_id='test-user-autouse'`。

## 未完成事项

### 🔴 高优先级 1：ReadCacheMiddleware（用户已确认要做）

**用户原话**：「但是不必要的read 文件，多次read 文件，造成loop closed，但是实际上它没有必要读」+「加 read 缓存中间件」+「缓存内容」

**设计已对齐**：
- 拦截 `read_file` tool_call，缓存键 `(thread_id, sandbox_id, path)` 整文件命中
- 命中 → 返回缓存内容 + 前缀 `[cache] file already read above`（agent 看到知道是缓存）
- 不破坏 LoopDetection 计数（cache 命中仍流过 LoopDetection 累计）
- 缓存范围：只缓存 `/mnt/user-data/uploads/*` + `/mnt/skills/*`（不缓存 `/mnt/user-data/workspace/*` 和 `/mnt/user-data/outputs/*`，因为 agent 会写这些目录）
- 缓存附带 mtime，命中时校验 mtime 一致才返回缓存
- TTL：thread 级别（thread 结束清掉，不跨 thread）
- 缓存上限：每 thread 最多 N 个文件 / 总 M MB，LRU 淘汰
- 配置开关：`config.yaml` 加 `read_cache.enabled: true|false`
- 逃生口：在 lead prompt 加一条"如需强制重读用 `force_reread: true` 参数"
- 实现位置参考：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` 学结构

**实施细节决策点**（设计已对齐，写时按这个做）：
- 用 `before_tool` hook（参考 LangChain middleware 类型 `langchain.agents.middleware.AgentMiddleware`）
- **cache 命中后仍要让 read_file 流过 LoopDetection 计数** —— 否则 agent 重复 read 既不报警也不阻断
- 返回缓存时**不阻断 agent 决策权**，只返回内容 + 提示文字（"a 选项"），不要拒绝重读（"b 选项"）

**TDD 要求**：必须配单测，放 `packages/agent/backend/tests/test_read_cache_middleware.py`

### 🟡 高优先级 2：thinking 慢问题（等用户发 log）

**用户原话**：「我先本地跑一个thread，你到时候看看langgraph.log。我们再决定，可能并不是过度think的问题，是反复决策的问题？或者我们能否用cache的方法呢」

**等待用户跑一次 thread 并发 langgraph.log**。在那之前**不要动 thinking 相关代码**。

可能的根因（看到 log 再确定）：
- (a) 真深度思考过度（reasoning_content 真的很长）
- (b) 反复决策来回弹（lead 在 task 派遣前思考很多次再决策）
- (c) Context 初始化慢（system prompt 太长、skills 太多）
- (d) Skill cache warm-up 慢（`_enabled_skills_cache` warm 路径）

**ultrathink 后才能给方案**——不要在没看 log 时就上"复用 intent guardrail 强制关 thinking" 之类的固定方案。

### 🟢 低优先级：Prompt 慢路径优化（如果 (c) 是 thinking 慢的根因）

`prompt.py` 已经把 skill 加载做成 background warm-up（`_refresh_enabled_skills_cache_worker`），但**首次启动还是要等**。可能优化方向：startup 预热、skill cache 持久化、skill 渐进披露而不是全注入。**等 log 确诊后再考虑**。

## 建议接手路径

### 第 1 步：读 CLAUDE.md 摸清现状

```bash
# 项目级
cat /home/wangqiuyang/noldus-insight/CLAUDE.md

# Backend 级（DeerFlow 细节）
cat /home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md
```

### 第 2 步：确认仓库干净 + 基线测试

```bash
cd /home/wangqiuyang/noldus-insight
git status                              # 应该 = "dev 与 origin/dev 一致 / 干净"
git log --oneline -5                    # HEAD 应该是 2b5100e8

cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_client_live.py --tb=no -q | tail -3
# 预期: 2677 passed, 17 skipped, 1 failed (那个预存在 router 测试)
```

### 第 3 步：根据用户当前的关注点选择路径

**如果用户给了 langgraph.log**：
1. 不要假设，直接读 log 找事实
2. 看 messages 序列，识别 thinking 内容占多少 token / 决策反复几次 / system prompt 占多少 token
3. 出具诊断报告 + 提 2-3 个可选方案让用户挑

**如果用户要求开始 ReadCacheMiddleware**：
1. 创建 worktree：`git worktree add .claude/worktrees/read-cache-middleware -b worktree-read-cache-middleware dev`
2. 进 worktree：`cd /home/wangqiuyang/noldus-insight/.claude/worktrees/read-cache-middleware`
3. **先**复制 `packages/agent/config.yaml` 到 worktree（gitignored，CLAUDE.md 教训"worktree + uv editable install 盲区"，详见 [feedback_worktree_uv_editable_install_pitfall.md](feedback_worktree_uv_editable_install_pitfall.md)）：
   ```bash
   cp /home/wangqiuyang/noldus-insight/packages/agent/config.yaml packages/agent/config.yaml
   ```
4. 用 TDD：先写测试 `tests/test_read_cache_middleware.py`（描述 cache miss / hit / mtime invalidation / 范围限制 / force_reread 逃生口 5 个场景），跑测试看红
5. 实现中间件 → 跑测试看绿
6. 全量 `PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_client_live.py --tb=no -q` 确保无 regression
7. commit + PR

## 风险与注意事项

### ⚠️ 易混淆点

1. **"sync 没合的剩余东西"≠"上游有的没合"**：上游 11 个 protected 文件大量 diff 是 Noldus 业务方向，**故意不合**，不要再做"补合 prompt.py 的某段"这种工作（除非用户明确要求）。
2. **"测试 fail 不一定是你引入的"**：先在 dev 干净状态跑测试看 baseline，再判断改动是否引入 regression。
3. **worktree 跑测试需要复制 gitignored 的 config.yaml**：CLAUDE.md 第 X 条「worktree + uv editable install 盲区」血泪教训。

### ⚠️ 不建议的方向

- ❌ 直接覆盖 `prompt.py` / `agent.py` / `executor.py` 整文件（CLAUDE.md 同步规则）
- ❌ 在没看 langgraph.log 时就提"thinking 慢"的修复方案
- ❌ 给 ReadCacheMiddleware 设计"拒绝重读"（强制 b 选项），用户明确说不要破坏 agent 决策权
- ❌ 改 memory `aupdate_memory` 回 async（已选 sync 委托避免 event loop bug）
- ❌ 改 `SubagentStatus.is_terminal` 回 enum identity 比较（importlib.reload 会破）

## 下一位 Agent 的第一步建议

**如果用户继续此线程**：等用户给指示——他要么发 langgraph.log（你看 log 诊断），要么说"开始做 ReadCacheMiddleware"（你按上面"建议接手路径第 3 步"执行）。

**如果是全新会话**：

```bash
# 1. 读 handoff
cat /home/wangqiuyang/noldus-insight/docs/handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md

# 2. 读两个 CLAUDE.md
cat /home/wangqiuyang/noldus-insight/CLAUDE.md
cat /home/wangqiuyang/noldus-insight/packages/agent/backend/CLAUDE.md

# 3. 确认基线
cd /home/wangqiuyang/noldus-insight && git log --oneline -1  # 应该是 2b5100e8

# 4. 问用户接下来想做什么：thinking 诊断还是 ReadCacheMiddleware
```

**用户当前的 mental model**：sync 已完成，**重点关注 agent 体验问题（thinking 慢 + read loop）**。他不再关心 sync 本身。

---

**会话产物**：
- PR #22 merge（ev19 worktree 的 patched_reasoning + memory user_id 穿透 fix）
- PR #23 merge（deerflow-upstream-sync worktree 的 3 个 commit）
- 仓库 worktree 全部清理
- 本 handoff 文档
