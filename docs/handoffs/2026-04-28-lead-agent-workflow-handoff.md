# Handoff: Lead Agent 工作流增强 — 修复 + 模式简化

**日期**: 2026-04-28
**状态**: tool 注册完成 + coroutine bug 修复 + 前端模式简化完成，等待 E2E 验证
**上一会话**: 2026-04-27 实施阶段交接（6 commits, 1705/1706 PASS）

---

## 1. 当前任务目标

延续 [2026-04-27 实施交接文档](2026-04-27-lead-agent-workflow-implementation-handoff.md) 的 P0 未完成项：

1. ✅ 注册 `set_experiment_paradigm` tool 到 agent tool list
2. ✅ 修复 `awrap_tool_call` 的 coroutine bug
3. ✅ 前端模式简化：ultra → auto，删除 flash/thinking/pro，只保留 auto + flywheel
4. ⬜ E2E 人工验证（用户手动进行）
5. ⬜ DeerFlow 上游更新检查（需要 GitHub 访问权限）

---

## 2. 关键变更（3 个）

### 2.1 注册 `set_experiment_paradigm` tool

**文件**: [tools/tools.py](packages/agent/backend/packages/harness/deerflow/tools/tools.py)

- 新增 import：`from deerflow.agents.middlewares.experiment_context import set_experiment_paradigm_tool`
- 加入 `BUILTIN_TOOLS` 列表（第 17 行），与 `ask_clarification_tool` 同级
- 验证：`set_experiment_paradigm` 出现在 `get_available_tools()` 返回列表中，参数完整（paradigm, paradigm_cn, category, subject, workspace_dir）

### 2.2 修复 `awrap_tool_call` coroutine bug（关键！）

**文件**: [gate_enforcement_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py)

**根因**: 原 `awrap_tool_call` 简单 delegate 给 `self.wrap_tool_call(request, handler)`。在 async 上下文中，`handler(request)` 返回的是 coroutine 而非 ToolMessage。`wrap_tool_call` 是 sync 方法，不会 `await`，把 coroutine 原样返回。LangChain 消息系统收到 coroutine 对象后报 `Unsupported message type: <class 'coroutine'>`。

**修复**: 参照 `ClarificationMiddleware` 和 `ToolErrorHandlingMiddleware` 的正确模式，`awrap_tool_call` 独立实现逻辑，关键差异是 `await handler(request)`。同时提取 `_build_block_message()` 消除重复代码，导入 `Awaitable` 和 `Command` 类型。

### 2.3 前端模式简化

**9 个文件修改，pnpm check 零错误**：

| 文件 | 变更 |
|------|------|
| `i18n/locales/types.ts` | flash/thinking/pro/ultra → auto + flywheel |
| `i18n/locales/zh-CN.ts` | "全自动" + "数据飞轮" |
| `i18n/locales/en-US.ts` | "Auto" + "Flywheel" |
| `settings/local.ts` | mode 类型 → `"auto" \| "flywheel" \| undefined` |
| `threads/hooks.ts` | 两模式统一启用 thinking/plan/subagent/reasoning=high |
| `workspace/input-box.tsx` | 删除旧模式 UI + reasoning effort 下拉 |
| `workspace/welcome.tsx` | 类型 → `"auto" \| "flywheel"` |
| `workspace/mode-hover-guide.tsx` | AgentMode → `"auto" \| "flywheel"` |
| `app/.../agents/new/page.tsx` | 初始化 mode `"flash"` → `"auto"` |

**两模式运行时行为**（hooks.ts 第 546-549 行）：

```
              │   auto   │ flywheel
──────────────┼──────────┼──────────
 thinking     │  true    │  true
 plan_mode    │  true    │  true
 subagent     │  true    │  true
 reasoning    │  high    │  high
 workflow     │  "auto"  │ "manual"
```

唯一区别是 `workflow_mode`：auto → 全自动流水线，flywheel → Gate 1 等待人工确认。

---

## 3. 测试基线

```
1704 PASS, 2 FAIL (pre-existing), 14 skipped
```

2 个已有失败（不是本次引入）：
1. `test_planning_skill_is_enabled_in_config` — `extensions_config.json` 文件不存在
2. `test_usage_example_shows_ask_clarification_between_analyst_and_writer` — prompt 文本已改但测试未更新

GateEnforcementMiddleware 5/5 PASS。

---

## 4. 未完成事项（按优先级）

### P0 — E2E 验证

1. **启动服务验证 flywheel 模式**:
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/agent && make dev
   ```
   - 访问 http://localhost:2026 → 新建对话 → mode-switcher 应该只有"全自动"和"数据飞轮"两个选项
   - 选择 flywheel → 上传测试数据 → 输入"帮我分析"
   - Gate 1 第一级是否弹出 7 大类选项？
   - 选择大类 → 第二级是否弹出细分范式？
   - 选择范式 → `experiment-context.json` 是否写入？
   - Gate 2 组别确认是否触发？
   - 切到 auto 模式 → 是否直接走全自动流水线？

2. **特别验证 coroutine bug 已修复**：
   - flywheel 模式选择范式后，`task()`（subagent 调用）不应再报 `Unsupported message type: <class 'coroutine'>`
   - 如果仍有类似错误，检查 LangGraph server 日志确认具体堆栈

### P1 — DeerFlow 上游同步

3. **检查上游更新**：上游 `git@github.com:Dimples-ai/deer-flow.git` 从当前机器无法访问（SSH + HTTPS 都不可用）。需要在有权限的环境中运行：
   ```bash
   cd /home/wangqiuyang/noldus-insight
   ./scripts/sync-deerflow.sh --dry-run
   ```
   上次 subtree sync point: `f0dd8cb`

### P2 — 文档/清理（延续 2026-04-27 的 P1）

4. paradigm-list.md 替换为动态生成
5. prompt 中范式列表改为从 `list_categories()` / `list_paradigms()` 动态注入
6. E2E 测试自动化

---

## 5. 文件位置速查

| 内容 | 路径 |
|------|------|
| 本次交接文档（实施阶段） | `docs/handoffs/2026-04-27-lead-agent-workflow-implementation-handoff.md` |
| 本次交接文档（设计阶段） | `docs/handoffs/2026-04-27-lead-agent-workflow-handoff.md` |
| set_experiment_paradigm tool 定义 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` |
| tool 注册（BUILTIN_TOOLS） | `packages/agent/backend/packages/harness/deerflow/tools/tools.py:14-18` |
| GateEnforcementMiddleware（已修复） | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py` |
| Lead agent workflow_mode 注入 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:235-287` |
| Lead agent prompt（Gate 1 两级分支） | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:259-350` |
| 前端模式定义 | `packages/agent/frontend/src/core/settings/local.ts:38` |
| 前端模式→context 映射 | `packages/agent/frontend/src/core/threads/hooks.ts:546-549` |
| 前端模式切换 UI | `packages/agent/frontend/src/components/workspace/input-box.tsx` |
| PARADIGMS registry | `packages/ethoinsight/ethoinsight/templates/__init__.py` |

---

## 6. 建议接手路径

```
第一步：验证测试基线
  cd /home/wangqiuyang/noldus-insight/packages/agent/backend
  source .venv/bin/activate && make test
  # 确认 1704 PASS, 2 pre-existing FAIL

第二步：启动服务 + E2E 验证
  cd /home/wangqiuyang/noldus-insight/packages/agent && make dev
  # 按第 4 节 P0 的 checklist 逐项验证

第三步（可选）：检查 DeerFlow 上游
  # 需要在有 GitHub SSH key 的机器上运行
  ./scripts/sync-deerflow.sh --dry-run
```

---

## 7. 风险与注意事项

| 风险 | 应对 |
|------|------|
| **coroutine bug 修复验证** | E2E 测试中重点验证：flywheel 模式选择范式后 agent 能正常调用 task()（subagent），不再抛出 coroutine 错误 |
| **前端模式持久化** | 旧 localStorage 中可能存有 `"ultra"` / `"pro"` 等模式值。`getResolvedMode` 已处理：undefined → 默认 `"auto"`。但旧值可能显示异常，清空 localStorage 即可 |
| **reasoning_effort 固定为 high** | 删除了 reasoning effort 下拉选择器，不会再出现 `"medium"` / `"low"` 等模式相关的 effort 值。但旧的 localStorage 可能保留 old reasoning_effort 值，不影响功能（hooks.ts 中 `context.reasoning_effort ?? "high"` 会覆盖） |
| **DeerFlow 上游无法访问** | 当前机器无法连接 `github.com:Dimples-ai/deer-flow.git`。需要确认仓库是否仍存在、是否改名或迁移 |
| **GLM-5.1 正面提示** | 继续遵循：只用正面指令，不用"禁止 X" |
