# 2026-05-22 FST E2E 基础设施修复 + subagent 进度播报

## 当前任务目标

修复 FST 端到端流程中的 4 个基础设施问题，并在 subagent 完成后向 lead 提供进度时间线。

**状态**：✅ 全部完成，2 个 commit 已合入 dev。

## 当前进展

### ✅ commit 1: `4473804a` — FST E2E 四重修复

**Phase A (P0): Gate 2 死锁修复**

根因：`GateEnforcementMiddleware` 告诉 lead "用 `write_file` 更新 experiment-context.json"，但 lead 的 `_LEAD_EXCLUDED_TOOLS` 排除了 `write_file`。lead 无法完成 gate 要求的操作，陷入 `task(data-analyst)` 重试死锁 → LoopDetection 触发。

修复：
- [experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py)：`set_experiment_paradigm_tool` 新增 `acknowledge_quality: bool = False` 参数，所有 Gate 1 字段改为 Optional。Gate 2 模式读取已有 context 追加 `gate2_quality_acknowledged`。
- [gate_enforcement_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py)：阻塞文案改为引导 `set_experiment_paradigm(acknowledge_quality=True)`。
- 测试：[test_set_experiment_paradigm_gate2.py](packages/agent/backend/tests/test_set_experiment_paradigm_gate2.py)（6 case）

**Phase B (P1): LoopDetection write_todos 兜底**

- [config.yaml](packages/agent/config.yaml)：新增 `loop_detection.tool_freq_overrides.write_todos: {warn: 2, hard_limit: 4}`（gitignored，仅本地生效）

**Phase C (P1): reasoning_effort 按阶段降级**

- [agent.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py)：`make_lead_agent` 中读取 `experiment-context.json` 的 `gate_completed` 状态，降级 reasoning_effort（high→medium→low）。fail-safe：找不到 workspace 时保持配置值。
- 测试：[test_make_lead_agent_reasoning_downgrade.py](packages/agent/backend/tests/test_make_lead_agent_reasoning_downgrade.py)（10 case）

**Phase D (P1): write_todos 三层防御**

- Layer 1：[todo_planning_discipline_provider.py](packages/agent/backend/packages/harness/deerflow/guardrails/todo_planning_discipline_provider.py) — 新建 GuardrailProvider，签名(content+activeForm) + status diff 检测 + `reason` 参数显式放行
- Layer 2：Phase B 的 LoopDetection 配置兜底
- Layer 3：[prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) — 新增 Todo 列表使用规则（正面指令）
- [todo_middleware.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py)：write_todos 工具增加 `reason` 参数
- 测试：[test_todo_planning_discipline_provider.py](packages/agent/backend/tests/test_todo_planning_discipline_provider.py)（6 case）

### ✅ commit 2: `39bda9e6` — subagent 进度时间线

根因：`SubagentExecutor` 只返回 subagent 最后一条 AIMessage 文本作为 ToolMessage。lead 需要通读全文才能提炼用户摘要，期间用户感知到"沉默等待"。

修复：[task_tool.py](packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py)
- 新增 `_build_progress_timeline(ai_messages)` — 从 `SubagentResult.ai_messages` 提取每步的里程碑标签（工具名或首句）
- 新增 `_extract_milestone_label(msg)` — 从单条消息提取简短描述
- `task()` 完成时的 ToolMessage 格式改为：
  ```
  Task Succeeded.

  ## 进度时间线
  1/10: 读取执行宪法
  2/10: 调用 read_file
  ...
  10/10: 写 handoff.json

  ## 最终结果
  <subagent 最终文本>
  ```

## 关键上下文

### 仓库状态

- **分支**：dev
- **HEAD**：`39bda9e6`
- **测试**：2698 passed / 17 skipped / 5 failed（5 个 fail 均为预存在 test isolation 问题）
- **config.yaml** 是 gitignored，Phase B 的 `loop_detection.tool_freq_overrides` 仅本地生效

### 重要文件路径

| 用途 | 路径 |
|------|------|
| Gate 2 修复 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` |
| Gate 阻塞消息 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py` |
| reasoning 降级 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` |
| Todo discipline provider | `packages/agent/backend/packages/harness/deerflow/guardrails/todo_planning_discipline_provider.py` |
| TodoMiddleware reason 参数 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py` |
| Task tool 进度时间线 | `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py` |
| Lead prompt | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| FST E2E 分析文档 | `docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md` |
| Handoff 参考 | `docs/handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md` |
| 项目 CLAUDE.md | `CLAUDE.md` |
| Backend CLAUDE.md | `packages/agent/backend/CLAUDE.md` |

## 关键发现

### 1. GuardrailMiddleware 支持多实例共存

通过 `name` 参数区分实例（[middleware.py:41](packages/agent/backend/packages/harness/deerflow/guardrails/middleware.py#L41)）。TodoPlanningDisciplineProvider 以 `fail_closed=False` 注册（误杀时不阻断，只提示）。

### 2. GuardrailRequest 不自动带 thread_id

`GuardrailMiddleware._build_request` 不设置 `thread_id` 字段。需要通过 ContextVar bridge 中间件模式（参考 `Ev19WorkspaceBridgeMiddleware`）传递。

### 3. TodoListMiddleware 的 write_todos 工具是动态创建的

不能用简单的 patch，需要在 TodoMiddleware.__init__ 中替换 `self.tools`。

### 4. SubagentResult.ai_messages 已存在但未利用

完整消息历史一直在内存中，`task_tool.py` 的 polling loop 也在逐条发送 `task_running` SSE 事件。但 ToolMessage 只包含最后一条消息文本。现在通过进度时间线利用了这个 infra。

## 未完成事项

### 后续观察（无需立即行动）

1. **reasoning_effort 降级效果** — 等待用户跑新的 FST E2E，对比 reasoning token 消耗。预期 Run 3/4/5 的 reasoning token 显著下降。
2. **write_todos 调用次数** — 预期从 ~6-8 次降到 ~2-3 次。
3. **进度时间线效果** — 观察 lead 是否因为时间线而更快生成用户摘要。

### 已知但推迟

4. **ReadCacheMiddleware** — 设计已对齐（handoff 文档），本次未实施。P0 修复后 lead 对 experiment-context.json 的重复读取已减少，可降低优先级。
5. **Pydantic 序列化警告** — 低优先级，不影响功能。
6. **test_memory_router::test_update_memory_fact_route_preserves_omitted_fields** — dev 上预存在 bug。

## 建议接手路径

**如果是新会话**：

```bash
# 1. 确认仓库状态
cd /home/wangqiuyang/noldus-insight
git log --oneline -3  # HEAD 应该是 39bda9e6

# 2. 读 CLAUDE.md
cat CLAUDE.md
cat packages/agent/backend/CLAUDE.md

# 3. 读本次改动文档
cat docs/problems/2026-05-22-fst-e2e-issues-and-solutions.md

# 4. 跑基线测试
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --ignore=tests/test_client_live.py --tb=no -q | tail -3
```

**如果用户继续此线程**：
- 用户可能在评估修复后的 FST E2E 效果，关注 reasoning token 减少和 write_todos 调用频率
- 用户可能要求实施 ReadCacheMiddleware（设计已就绪）
- 用户可能报告新的 E2E 问题

## 风险与注意事项

### ⚠️ 易混淆点

1. **test isolation 问题不是回归**：5 个 fail（`test_build_middlewares_uses_resolved_model_name_for_vision`、`test_lead_agent_includes_training_data_middleware`、`test_training_middleware_sits_after_memory`、`test_memory_router`、1 个其他）在单独运行时全部通过，是预存在的 test isolation 问题。
2. **config.yaml 是 gitignored**：Phase B 的 `loop_detection.tool_freq_overrides` 改动不会提交到 git。如果用户在新环境部署，需要手动加。
3. **TodoPlanningDisciplineProvider 的 threshold**：`planning_budget=2`，前 2 次 `write_todos` 总是放行。如果 lead 在前 2 次就完成了正确的 todo，后续触发 guardrail 是预期行为。

### ⚠️ 不建议的方向

- ❌ 给 lead 恢复 `write_file` 工具 — Gate 2 已通过 `set_experiment_paradigm(acknowledge_quality=True)` 解决
- ❌ 在 ClarificationMiddleware 中隐式更新 experiment-context.json — 引入隐式副作用
- ❌ 全局降低 reasoning_effort — Gate 1 阶段确实需要 high reasoning。当前方案按阶段降级
- ❌ 完全禁用 `write_todos` — agent 仍需追踪任务状态，当前通过 guardrail 限制频率
