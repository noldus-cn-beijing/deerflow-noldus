# Step 0 地基假设坐实报告：A1 后端事件分轨地基

> 对应 spec：[2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md](../specs/2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md)「Step 0（不可跳）：地基假设坐实」。
>
> 结论先行：**两条地基假设均成立**，选定落点见下。本报告在实施前落盘（验收标准 1）。

## 假设 1：custom 轨可达性 ✅ 成立

**问题**：在 lead 运行中调 LangGraph `get_stream_writer()({...})`，worker 是否正确 serialize 并以 `custom` SSE 事件名推送？

**证据（静态勘察 + 既有生产用法）**：

1. `runtime/runs/worker.py:42` — `"custom"` 在 `_VALID_LG_MODES` 内，是合法 astream stream_mode。
2. `runtime/runs/worker.py:309-334` — worker 把每个 chunk 经 `_lg_mode_to_sse_event(mode)` 映射成 SSE 事件名后 `bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))`。
3. `runtime/runs/worker.py:551-559` — `_lg_mode_to_sse_event` 是**1:1 恒等映射**（`return mode`），故 `custom` 模式 → `custom` SSE 事件名，原样透传。
4. **既有生产用法坐实**（不是理论可行，是已在跑）：
   - `agents/middlewares/llm_error_handling_middleware.py:283-295` — 中间件内 `from langgraph.config import get_stream_writer` + `writer({...})` 发 `llm_retry` 事件。
   - `agents/middlewares/safety_finish_reason_middleware.py:181-185` — 同模式发 `safety_termination` 事件。
   - `tools/builtins/task_tool.py:544-631` — `task` 工具内发 `task_started` / `task_running` / `task_completed` / `task_failed` / `task_timed_out` / `task_cancelled`。

**结论**：`get_stream_writer()` 写入的 dict 经 worker 恒等映射为 `custom` SSE 事件推送到客户端。A1 在中间件 / `task` 工具里复用同一写法即可，与 `llm_retry` / `task_started` 同构。

## 假设 2：派遣边界可观测性 ✅ 成立

**问题**：lead 派遣 subagent 的进/出在代码层是否有确定性可挂的钩子？能否拿到 subagent 类型 + 意图模式？

**证据**：

1. **派遣边界 = `task` 工具本身**（`tools/builtins/task_tool.py`，确定性，不是 LLM 自报）：
   - 进入点 `task_tool()`（line 354 起）：拿到 `subagent_type`、`description`、`tool_call_id`。
   - line 533 `executor.execute_async(...)` → 后台执行。
   - line 544 `writer = get_stream_writer()`，line 546 发 `task_started`。
   - 轮询 `result.status`（line 549-579）：完成发 `task_completed`（587），失败发 `task_failed`（596）/ `task_timed_out`（610/631）/ `task_cancelled`（603）。
   - **状态来自真实 executor 结果**（`get_background_task_result`），不是 LLM 自述——满足 spec「grounded，叙事不撒谎」。
2. **意图可确定性提取**：`guardrails/intent_classification_provider.py:63-88` `_latest_declared_intent(messages)` 已是现成 helper（lead 在第一个非 read_file tool call 前必须输出 `[intent] <INTENT>` 行，由 guardrail 强制）。`guardrails/path_sequence_provider.py:60` 同一正则 `_INTENT_LINE_RE`。
3. **意图→阶段映射已有 SSOT**：`guardrails/path_registry.py` 的 `PATHS`（8 条 INTENT → 有序 dispatch/ask step）+ `VIZ_INTENT_KEYWORDS`。**A1 复用此 SSOT 派生阶段集，不新建第二份阶段字典**（守 single-source-of-truth）。

**结论**：派遣进/出在 `task` 工具内有确定性观测点，且 subagent_type + 意图均可机械拿到。

## 选定落点（spec 模块 2「优先不重复造观测」）

spec 模块 2 给了两个落点候选，要求「若 executor/task_tool 已有可挂的边界钩子，复用之」。坐实后选定：

| 事件 | 落点 | 理由 |
|---|---|---|
| `stage_plan`（意图确定后发一次） | **新 `StageNarrationMiddleware`**（after_model，读 messages） | 意图是 lead 的决策产物，由中间件在 model 输出后观察 `[intent]` 行触发；非流水线意图不发。映射逻辑抽到纯模块 `stage_narration.py`。 |
| `stage_update`（每阶段进/出） | **复用 `task` 工具既有派遣观测点** | `task_tool.py` 已有 writer + subagent_type + 真实 status + description，是唯一「知道 subagent 真失败」的代码路径（验收 4 grounded 要求）。在既有 `task_started`/`task_completed`/`task_failed` 旁同源发 `stage_update`，不重复造观测。映射走同一 `stage_narration.py` 纯模块。 |

**架构原则兑现**：
- 真实状态机驱动（不靠 LLM 自报）：`stage_plan` 来自 guardrail 强制的 `[intent]` 行；`stage_update` 状态来自 executor 真实 `result.status`。与 PR#213 / `SealGateMiddleware` 同构（确定性门 > prompt 打地鼠）。
- SSOT：阶段名 + 意图→阶段映射只存 `stage_narration.py` 一处（消费 `path_registry.PATHS`），前端不维护第二份。
- 不动 messages / reasoning 轨、不动 worker、不动 ThinkTagMiddleware、不引入 ag-ui。

## 范围确认（A1 内）

- ✅ 新 `stage_narration.py`（纯映射模块）+ `StageNarrationMiddleware`（挂 lead 链）+ `task_tool.py` surgical 增 `stage_update` 发射。
- ❌ 不动前端（A2）、不动 messages/reasoning 轨、不引入 ag-ui。
