# Spec C：后端质量五小项（legend / memory JSON / thread_id / Pydantic context 噪声 / 进度条脱节）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26（2026-06-26 E2E 后扩充 §四/§五）
> 代码基线：dev HEAD `6ec47f78`
> 性质：🟢 低 · 后端/前端质量瑕疵打包（不影响主流程，但污染 log / 退化体验），独立可做
> 取证：dogfood thread `e9837b33` + E2E thread `bd7ca7f7`（`/tmp/noldus-e2e-runs/20260626-114801/`）的 gateway.log
> 范围：五个**互不相关**的低优先级问题，打包方便一次清理；实施 agent 可独立处理任一项

---

## 一、图表 legend warning（19 次，charts.py:295）

### 现象
gateway.log 出现 19 次：
```
ethoinsight/charts.py:295: UserWarning: No artists with labels found to put in legend.
```

### 根因（读码坐实）
`charts.py:295` `ax.legend(fontsize=8, loc="upper right")` 被调用，但**当前图没有带 label 的 artist**（trajectory 单组/无分组时，`ax.plot(...)` 没传 `label=`，却仍调 `legend()`）。matplotlib 对"无 label 还调 legend"发 UserWarning。

### 修复
仅当**确有 labeled artist** 时才调 `legend()`：
```python
# charts.py 绘图函数，调 legend 前判断
handles, labels = ax.get_legend_handles_labels()
if labels:
    ax.legend(fontsize=8, loc="upper right")
```
或在多组分支内调 legend、单组分支不调（`charts.py:295` 已在多组分支，确认 else 单组分支没误调）。**影响范围**：trajectory/heatmap 等 per_subject 图。

### 验证
`cd packages/ethoinsight && pytest tests/` 跑图表测试；生成一张单组图，确认无 UserWarning。**纯图表库改动，与 agent harness 无关**，不需裸导入。图不变（legend 本就该在有图例时才出）。

---

## 二、memory 更新 JSON 解析失败（updater.py:528）

### 现象
gateway.log（该 thread 2 次）：
```
deerflow.agents.memory.updater - WARNING - Failed to parse LLM response for memory update: No valid memory update JSON object found
deerflow.agents.memory.queue - WARNING - Memory update skipped/failed
```

### 根因（读码坐实）
`updater.py:326` `raise json.JSONDecodeError("No valid memory update JSON object found")` —— LLM（memory 更新用的模型）返回的内容**不是合法 JSON 对象**，`updater.py:528` catch 成 warning、跳过更新。**这个 thread 没写进记忆飞轮**。

### 修复（按 HarnessX 自检：先判结构 vs prompt）
- **结构层**：确认 memory 更新调用有没有用 JSON mode / structured output 强制约束（若 LLM 支持）。若没有，加 `response_format={"type":"json_object"}` 或等价约束 → **结构性根治**优于改 prompt。
- **prompt 层**：若已是 structured output 仍偶发，检查 prompt 是否明确要求"只输出 JSON、无前后缀文本"（deepseek 正面指令）。
- **容错**：偶发解析失败 catch 成 warning 跳过本次更新（现状）是合理降级，**但不应高频**。若高频，说明约束不够硬，优先上结构约束。

### 验证
单测：mock LLM 返回带前后缀文本的 JSON（如 ```json ... ``` 包裹）→ 确认解析能剥离/或 structured output 后不再发生。dogfood：跑几个 thread，grep log 确认 memory update 失败率下降。裸导入两生产入口（改 agents/memory）。

---

## 三、quality_warning_broadcast thread=unknown（middleware:87）

### 现象
gateway.log（该 thread 3 次）：
```
quality_warning_broadcast | thread=unknown | injected=0 warnings
```

### 根因（读码坐实）
`quality_warning_broadcast_middleware.py:87` `state.get("thread_id", "unknown")` —— **state 里没有 `thread_id` 字段**，fallback 成 "unknown"。功能没崩（injected=0，本次无 warning 要广播），但说明 **thread_id 没贯通到这个中间件的 state**——是潜在的上下文传递缺陷（若将来该中间件需要 thread_id 做实事，会拿到 unknown）。

### 修复
- 确认 thread_id 在该中间件能否从别处拿到（`runtime.context` / ContextVar，对照其它中间件怎么拿 thread_id——memory `feedback_toolruntime_context_thread_id_is_flat_not_nested`：`runtime.context.get("thread_id")` 是扁平的）。
- 若 state 本就不该有 thread_id，改成从 `runtime.context` 取：`runtime.context.get("thread_id", "unknown")`。
- **优先级最低**：当前无实际故障（只是 log 标记 unknown），但修了能让 log 可追踪 + 防将来该中间件依赖 thread_id 时踩坑。

### 验证
单测：构造带 thread_id 的 runtime context，调中间件，断言 log 里 thread=<真实id> 非 unknown。裸导入两生产入口。

---

## 四、Pydantic context 序列化噪声（E2E 新发现，污染 log）

### 现象
E2E thread `bd7ca7f7` gateway.log **大量刷屏**：
```
PydanticSerializationUnexpectedValue(Expected `none` - serialized value may not be as expected
  [field_name='context', input_value={'thread_id': 'bd7ca7f7-5...object at 0x...>}, input_type=dict])
```

### 根因方向（需定位）
某 Pydantic model 的 `context` 字段声明类型与实际传入不符——实际传了 `dict`（含 `thread_id` + 某对象），但字段类型期望 `none`/其它。Pydantic 序列化时发 `PydanticSerializationUnexpectedValue`。**非阻断**（序列化仍继续），但每条消息/事件刷一次，污染 log 难排查真问题。

### 修复
- grep `field_name='context'` 涉及的 model（很可能是 runtime/event/stream 相关的 Pydantic schema），确认 `context` 字段类型声明。
- 要么把字段类型改对（`dict | None` 或具体 TypedDict），要么传入前 normalize。**先定位是哪个 model 序列化时带 context**（事件流 / run 元数据 / ToolMessage？）。
- 验证：跑一个 thread，grep gateway.log 确认 `PydanticSerializationUnexpectedValue` 不再刷。

---

## 五、前端流程进度条与后端脱节（E2E 新发现，独立 UI bug）

### 现象
E2E 实测：前端流程指示器（AnalysisRail，spec#4，`analysis-rail.tsx`）显示"**列对齐 进行中 / 指标计算 未开始**"，但后端**实际早已算完 140 指标 + 113 图**。进度条没随真实进度推进。

### 根因方向（需定位）
AnalysisRail 是前端从消息流派生阶段状态（spec#4 "前端推导零后端改动"）。推导逻辑可能：① 漏识别某些阶段的完成信号 ② 阶段映射与实际 subagent 事件不对齐 ③ 派生时机滞后。**读 `analysis-rail.tsx` 的阶段推导逻辑，对照真实事件流（E2E 的 `sse-*.txt` / `specB-messages-raw.json`）看哪一步没推进。**

### 修复
- 定位推导逻辑漏判的阶段信号，补上映射（守 spec#4 的"前端推导"原则，别加后端字段除非必要）。
- 验证：E2E 跑一遍，进度条阶段随真实 subagent 完成推进，不停在"列对齐进行中"。

> 注意：这是 spec#4 AnalysisRail 的后续修复，与 §一~四后端项不同范围（前端）。可单独派前端 agent。

---

## 六、实施顺序（五项独立，按价值）

1. **二（memory 更新 JSON）** 价值最高——直接影响记忆飞轮数据质量（训练数据源）。
2. **一（legend warning）** 最简单——纯图表库一处判断，顺手清。
3. **三（thread=unknown）** 最低——无实际故障，可观测性改善。

**三项互不依赖**，可分别提交或一并清理。每项改后端 agents/ 的都要**裸导入两生产入口**（`import app.gateway` + `make_lead_agent`）；图表项只跑 ethoinsight pytest。

---

## 五、关键文件

- `packages/ethoinsight/ethoinsight/charts.py:295`（legend warning）
- `packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py:326/528`（JSON 解析）
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/quality_warning_broadcast_middleware.py:87`（thread=unknown）

---

*依据：gateway.log 三类 WARNING/UserWarning + 读码坐实各自代码位置。三项低优先级、互不相关、不影响主流程，打包便于一次清理。*
