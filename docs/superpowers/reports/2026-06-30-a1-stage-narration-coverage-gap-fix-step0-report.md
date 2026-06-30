# Step 0 派发点坐实报告：A1 stage narration 覆盖缺口补齐

> 对应 spec：[2026-06-30-a1-stage-narration-coverage-gap-fix-spec.md](../specs/2026-06-30-a1-stage-narration-coverage-gap-fix-spec.md)「三、Step 0（坐实派发点）」。
>
> 结论先行：**「识别范式」阶段有确定性、grounded 的派发观测点**（`StageNarrationMiddleware.wrap_tool_call`，已挂在链上），**`prep_metric_plan` 返回 `status=ok` 是「识别完成、即将派 code-executor」的可靠信号**。缺口 2（knowledge-assistant）是纯 SSOT 登记。本报告在实施前落盘（验收标准 1）。

## 问题 1：「识别范式」阶段需要非 subagent 的真实派发点

「识别范式」由 **lead 自调工具**完成（`identify_ev19_template` / `inspect_uploaded_file` / `prep_metric_plan` / HITL 反问），**不派 subagent**，故 A1 既有派遣观测点（`task_tool.py:emit_dispatch_enter/exit`，仅 subagent 边界）永远收不到它。

### 候选评估

spec 二·修法 1 给了两类候选：(a) 工具调用边界（工具执行包装 / 中间件 / executor），(b) 复用 SealGate/中间件链（after_model 钩子识别识别类工具）。

**勘察 `AgentMiddleware` 钩子面**（`.venv/bin/python` introspect `langchain.agents.middleware.AgentMiddleware`）：

```
['after_agent', 'after_model', 'before_agent', 'before_model',
 'wrap_model_call', 'wrap_tool_call', ...async 对偶]
```

**关键发现：`wrap_tool_call(request, handler)` 是逐工具执行的精确观测点**（`langchain.agents.middleware.AgentMiddleware`）：

- `request.tool_call["name"]` → 当前要执行的工具名（确定性，机器侧）。
- `handler(request)` → 该工具**真实返回的 `ToolMessage`**（含 `prep_metric_plan` 的 `status`）。
- 既有的 `after_model` 钩子要在「下一轮 model 输出后」才间接看到 `ToolMessage`，需要回溯配对 tool_call_id，更绕、更易错；`wrap_tool_call` 在执行当场拿到 name + result，**一对一**。

### 选定落点：`StageNarrationMiddleware` 覆盖 `wrap_tool_call` / `awrap_tool_call`

`StageNarrationMiddleware`（`agents/middlewares/stage_narration_middleware.py`）已挂在 lead 链上（`agent.py:339`），构造期已注入 `writer`。给它加两个工具观测规则（映射走 `stage_narration.py` SSOT，与 A1 既有 `emit_dispatch_enter/exit` 同源）：

| 事件 | 触发（wrap_tool_call 内） | grounded 性 |
|---|---|---|
| `识别范式` **active** | `tool_call["name"]` ∈ {`identify_ev19_template`, `inspect_uploaded_file`} → 调 `handler` **之前**发 | 真实工具调用（机器侧 name），非 LLM 自报 |
| `识别范式` **completed** | `tool_call["name"] == "prep_metric_plan"` → 调 `handler` **之后**，解析返回 `ToolMessage.content`，`status == "ok"` 才发 | 真实工具结果（prep 真正 resolve 成功），失败/ambiguous/unsupported → **不发 completed**（叙事不撒谎，spec 验收 4） |

**与既有架构同构**：和 A1 既有 `emit_dispatch_exit(succeeded=...)`（`task` 工具用真实 executor 结果决定发不发 completed）完全同构——completed 由真实信号决定，不靠 LLM 自报。与 `QualityWarningBroadcastMiddleware`（在 after_model 读 tool_calls + ToolMessage）也同构，但本方案用更精确的 `wrap_tool_call`。

### `prep_metric_plan` 返回 shape 坐实

`prep_metric_plan_tool`（`tools/builtins/prep_metric_plan_tool.py:368-378`）成功返回：
```python
{"status": "ok", "plan_path": "...", "analysis_config_id": "...", "plan_summary": {...}}
```
失败路径 `_error_result`（:381）一律 `{"status": "error", "error_code": ...}`；`identify_ev19_template` 的 `unsupported`/`ambiguous`/`unknown` 也不是 ok。所以 `status == "ok"` 是「识别+计划完成、plan_metrics.json 已落盘、即将派 code-executor」的**可靠信号**（grounded，不撒谎）。

`ToolMessage.content` 是 `str`（JSON 序列化的工具返回）。解析：`json.loads(content).get("status") == "ok"`；解析失败/非 dict → 当作非 ok（不发 completed，安全侧倒）。

### `wrap_tool_call` 是否能拿到 writer / 在 graph 上下文？

`StageNarrationMiddleware` 构造期已注入 `writer`（生产用惰性 `get_stream_writer` 解析，测试注入捕获 list）。`wrap_tool_call` 是实例方法，直接用 `self._writer`，**与 `after_model`（stage_plan 发射）同一 writer、同一吞异常策略**（`_safe_write`）。无需在 hook 里重新拿 context。

## 问题 2：`knowledge-assistant` 漏登记（纯 SSOT）

`_DISPATCH_STAGE`（`stage_narration.py:49-54`）只登记 `code-executor`/`data-analyst`/`chart-maker`/`report-writer`，无 `knowledge-assistant`。

**落点**：在 SSOT `stage_narration.py` 一处加 `STAGE_KNOWLEDGE = "查阅领域知识"` 常量 + `_DISPATCH_STAGE["knowledge-assistant"] = STAGE_KNOWLEDGE`。一旦登记，`task_tool.py` 既有 `emit_dispatch_enter/exit` **自动**为 knowledge-assistant 派遣发 stage_update（无需改 task_tool）。

**「无 stage_plan 的独立活动提示」语义确认**：QA 意图（`QA_FACT`/`QA_KNOWLEDGE`）不在 `_INTENT_STAGES`（非流水线），A1 不发 stage_plan，故 knowledge-assistant 的 stage_update 在前端是「独立活动提示」而非 stepper 节点——前端渲染属 A2 范围，本 spec 不动前端。

## 防 vacuous 已对齐

`test_stage_narration.py` 的 `TestNoVisceraLeakage.FORBIDDEN` 已含 `knowledge-assistant`、`identify_ev19_template`、`set_experiment_paradigm`、`ev19_template` 等。新增的 `STAGE_KNOWLEDGE = "查阅领域知识"` 不含任何内脏词（已在 FORBIDDEN 集合上断言）。

## 风险与守约

- **守 SSOT**：新阶段名只进 `stage_narration.py` 一处，前端不复刻（A1 已立原则）。
- **不改 A1 5 阶段集语义**：`_INTENT_STAGES` 不动；只在 `_DISPATCH_STAGE` 加一项 + 加派发观测点。
- **惰性 import / 不闭环**：`StageNarrationMiddleware` 已惰性 import `intent_classification_provider`；新增的工具名匹配是字面量集合，无新 import，不触发 harness import-cycle（`tests/test_gateway_import_no_cycle.py` 会守）。
- **不动前端**（A2 gate）。
