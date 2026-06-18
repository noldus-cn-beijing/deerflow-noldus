# Spec: seal 漏调的工程化根治 —— SealGateMiddleware（结构性出口闸门）+ prompt 合一

> 日期：2026-06-16
> 类型：harness 中间件（主体）+ prompt 重构（辅助）
> 范围：data-analyst + chart-maker + report-writer 的 seal 漏调。**code-executor 不改**（已合一，正面范例）。
> 状态：待 review → 批准后实施。
> 前身：本 spec 取代纯 prompt 版 `2026-06-16-seal-produce-deliver-merge-spec.md`（那份只做了"辅助层"，缺"主防线"，定位错误）。

---

## 0. 定位：这次要的是结构性 100%，不是概率性 99%

用户明确：「改 prompt 可能保证 99% 不再发生，但我们需要真正让它工程化落地。」

seal 漏调（subagent「terminated without emitting handoff」）在本仓库已被 prompt/skill/schema "修复" ≥4 次、每次换个诱因复发（详见附录 A）。**纯 prompt 改本质是"求模型配合"，永远是概率性的。** 本 spec 改终止条件本身，让漏 seal 在结构上无法发生。

**两层方案，角色不能搞反：**

| 层 | 机制 | 角色 | 保证 |
|---|---|---|---|
| **L1 主防线** | SealGateMiddleware（after_model + jump_to=model） | 结构性闸门 | **100%**：模型物理上无法不调 seal 就退出 subagent |
| **L2 辅助** | prompt 产出/交付合一 | 让模型一次就过闸 | 概率性：减少被闸门拦的次数、省 turn、理顺误导结构 |

**关键区分：L1 不是兜底。** 兜底（用户已否决）= 模型结束后用磁盘文件帮它补 handoff（承认漏 seal 发生、事后擦屁股）。L1 闸门 = 模型结束**前**拦住、`jump_to=model` 强制导回、**不让它结束**（漏 seal 无法发生）。前者治标，后者治本。

---

## 1. 根因：ReAct 允许"无 tool_call = 结束"

subagent 是标准 ReAct agent（`langchain.agents.create_agent`，executor.py:964）。ReAct 终止逻辑：

```
模型生成消息 → 有 tool_calls? → 是：执行工具，回到模型
                              → 否：终止（这条消息即最终答案）
```

**"模型生成一条无 tool_calls 的纯文本" = 循环自然结束。** 这是 ReAct 内置语义。模型只要写一条"分析完成"的纯文本（不带 seal tool_call），agent 就正常退出——harness 此时才发现 handoff 没写，但循环已结束。

**这就是为什么 prompt 的"step 3 必须 seal"无法强制**：ReAct 没有"步骤顺序"概念，只认"这条消息有没有 tool_call"。模型对"我做完了"的判断完全自主。executor.py:1040 也已记录"不强制 tool_choice（探针证明会产空 args）"——强制工具调用此路不通。

**唯一能改终止条件的位置：`after_model` hook。** 它在每次模型输出后、ReAct 决定是否终止前介入，可 `jump_to='model'` 把控制权强制导回。

---

## 2. infra 现成 + 活的先例（不是新发明）

- **langchain 1.2.3 原生支持**（已实测）：`@hook_config(can_jump_to=["model"])` 装饰 `after_model`，官方语义"Jump back to the model node"。factory.py 出现 `jump_to` 42 次、`'model'` 11 次。
- **本仓库已有一模一样的活范例**：`ParadigmIdentificationGateMiddleware`（`agents/middlewares/paradigm_identification_gate_middleware.py`）——用完全相同的模式强制 lead 先调 `identify_ev19_template`：检测"该调没调的必需工具"→ 注入 reminder HumanMessage + `jump_to='model'` 导回 → `_MAX_REMINDERS=2` 上限防死循环 → fail-open。**SealGateMiddleware 是这个模式在 subagent 链上的复刻，零新机制。**
- **subagent 有自己的 middleware 链**（executor.py:887 `_build_middlewares`，已挂 Guardrail/LoopDetection），SealGate 可直接 append。

---

## 3. L1：SealGateMiddleware（主体）

### 3.1 行为（复刻 ParadigmIdentificationGate 模式）

`after_model` hook，对**需要 seal 的 subagent**（data-analyst / chart-maker / report-writer）生效：

```
after_model(state):
  1. 本 subagent 不在 {data-analyst, chart-maker, report-writer} → return None（放行；code-executor/bash/general/knowledge 不管）
  2. 最后一条 AIMessage 的 tool_calls 里有 seal_<name>_handoff → return None（正在调，放行）
  3. handoff 历史里已有 seal 的 ToolMessage（已调过）→ return None（放行）
  4. 最后一条 AIMessage 仍带其它 tool_calls（还在干活，没想结束）→ return None（放行，别打断正常工具循环）
  5. reminder 次数 ≥ _MAX_REMINDERS → return None（撞上限，放行；交给 seal-resume + FAILED 这层老网，避免死循环）
  6. 否则（模型想用纯文本结束、但 seal 没调）→ 注入 reminder + jump_to='model'：
     return {"messages": [HumanMessage(reminder, hide_from_ui=True)], "jump_to": "model"}
```

判据 4 关键：只在"模型**想结束**（最后 AIMessage 无任何 tool_call）"时拦截，不打断正常的多工具循环（如 data-analyst 还在 read_file 范式文档）。判据 2/3 区分"正在调 seal"和"已调过"，避免重复拦。

### 3.2 seal 工具名 / handoff 文件名（已有规律，复用）

- seal 工具名：`f"seal_{name.replace('-','_')}_handoff"`（executor.py:1046 已用）。
- handoff 文件名：`_AUTO_SEALABLE` / executor.py:92-94 已有映射（data_analyst→handoff_data_analyst.json 等）。判据 3 检测 ToolMessage.name == seal 工具名即可（不必读磁盘，更快更稳）。

### 3.3 reminder 文案（正向，CLAUDE.md §6）

```
<system_reminder>
你的分析已完成，但尚未发出 seal_<name>_handoff 工具调用。
你的分析结论只有通过这次工具调用才会产出落库——这是产出分析的唯一动作。
请现在调用 seal_<name>_handoff，把你的结论（key_findings 等结构化字段）作为工具参数发出。
</system_reminder>
```

正向措辞、不用"不要/禁止"（deepseek 反向激活）。文案点明"调工具=产出"（与 L2 prompt 同一心智模型）。

### 3.4 防死循环 + fail-open（硬要求）

- `_MAX_REMINDERS = 2`（同 ParadigmIdentificationGate）。撞上限 → 放行 → 交给既有 seal-resume（executor.py:1029）+ 5.7 FAILED 老网。**SealGate 把"1 次机会"变成"max_turns 内持续逼 + 2 次显式 reminder"，但仍有上限，不会无限循环。**
- recursion_limit（executor.py:76）是 langgraph 硬上限，jump_to 受它管，是死循环的最终物理保险。
- `after_model` 整体 try/except → 异常 return None（fail-open，绝不因 gate 故障卡死 subagent）。
- reminder 计数按 run 隔离（`runtime.run_id`，同 ParadigmIdentificationGate）。

### 3.5 挂载位置

`executor.py:_build_middlewares` 末尾 append（LoopDetection 之后）。一个 SealGateMiddleware 实例按 `self.config.name` 自识别该管哪个 seal——**一个 middleware 覆盖三个 subagent**，不需各写一份。

---

## 4. L2：prompt 产出/交付合一（辅助，减少闸门触发）

L1 保证不漏，L2 让模型尽量一次过闸（少被拦、省 turn），并消除误导结构。详见根因附录 B。

### 4.1 data-analyst（重度分离，主体）

- step 2.7/2.8 标题"一次性完成核心分析推理"→ 改为"用思考**推导分析素材**（识别离群 + leave-one-out + 选判据 + 比对参数），不在思考里撰写最终 key_findings/各字段文本"。
- step 3"封存 handoff（把已得结论填进参数）"→ 改为"**产出分析 = 发出 seal tool_call**：结论第一次成文就直接写在工具参数里，没有'先写一遍再填进来'"。
- 删除 7+ 处"seal 必达/别忘/写文本≠落库"哀求补丁（L88-90 后半/L143/L165 括号/L192/L199/L270-274）——它们是分离症状，L1 闸门接管后不需要。**保留**"严禁 write_file 写 handoff，必须走 seal 工具"（工具选择约束，非症状）。
- fast-fail 路径同样合一：触发 → 直接发 seal tool_call 输出 partial/failed 结论。

### 4.2 chart-maker（轻度）

step 11 改为"绘图脚本一执行完，**下一个动作就是** seal_chart_maker_handoff，chart_files 直接列 step 10 落盘的 png 路径"。强调 seal 紧贴绘图、不要先 present_files 再回头 seal。

### 4.3 report-writer（轻度）

step 4 改为"report.md 一 write_file 完成，**下一个动作就是** seal_report_writer_handoff"。强调写完报告下一步必须是 seal。

---

## 5. 验收

### 5.1 L1 单元测试（可断言，TDD）

`tests/test_seal_gate_middleware.py`（参考 `test_paradigm_identification_gate_middleware.py` 若有）：

- **红→绿核心**：构造 state = 最后一条 AIMessage 无 tool_call + 无 seal ToolMessage + subagent=data-analyst → `after_model` 返回 `{"messages":[reminder], "jump_to":"model"}`（红：当前无此 middleware，模型纯文本直接终止）。
- 放行用例：① 最后 AIMessage 带 seal tool_call → None ② 历史有 seal ToolMessage → None ③ 最后 AIMessage 带其它 tool_call（还在干活）→ None ④ subagent=code-executor → None ⑤ reminder 已达 2 次 → None。
- fail-open：state 缺 messages / 抛异常 → None。
- 计数隔离：不同 run_id 计数独立。

### 5.2 裸导入回归（CLAUDE.md 铁律）

新 middleware 在 `agents/middlewares/`，被 `_build_middlewares` import。改后必须：
```
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
```
两者 exit 0（防 import 环——middleware import 放 `_build_middlewares` 函数体内惰性 import，同现有 Guardrail/LoopDetection 的挂法）。

### 5.3 dogfood 行为验证（治本终验）

用本次失败的同一数据（`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`，control=7/treatment=21，列名 open/closed）重跑：

- data-analyst 即使遇到空 statistics（手算冲动场景）也**必然**最终发出 seal tool_call（被闸门导回逼出），不再 `terminated without emitting`。
- 跑三场景：正常 statistics / 空 statistics / n<3 → 三种都 seal 成功。
- chart-maker / report-writer 各跑一次确认 seal 紧随产出、不漏。
- 观测 gateway.log：确认 SealGate 的 reminder 注入日志（被触发的次数应随 L2 prompt 改善而下降——这是 L2 见效的探针，同 `fallback_trigger_rate_must_be_observable` 精神）。

### 5.4 full 回归

ethoinsight 全量 + backend 全量；subagent middleware 链改动后跑 `test_subagent*` / executor 相关。

---

## 6. 不做 / 范围外

- **不改 code-executor**（已合一，run_metric_plan 即产出即落盘，结构上不漏；今天多次 dogfood 无问题）。SealGate 判据 1 显式放行它。
- **不加 auto-seal 兜底**：不把 data-analyst 加进 `_AUTO_SEALABLE`。L1 闸门是"结束前拦"，与"结束后补"正交且更优。既有 seal-resume + 5.7 FAILED 作为撞上限后的老网保留，不动。
- **不强制 tool_choice / 不改 max_turns 语义 / 不改 schema**。
- **不碰 statistics 列对齐**（另一 spec，用户在修，正交——那个让数据自洽，本 spec 让 seal 不漏，两者叠加才让本次 dogfood 完全跑通）。
- **不引入 thinking 开关改动**。

---

## 7. 改动文件清单

```
新增  agents/middlewares/seal_gate_middleware.py        （L1 主体，复刻 ParadigmIdentificationGate）
改    subagents/executor.py:_build_middlewares          （append SealGateMiddleware，惰性 import）
改    subagents/builtins/data_analyst.py                （L2：step 2.7/2.8/3 合一 + 删 7+ 哀求补丁）
改    subagents/builtins/chart_maker.py                 （L2：step 11 seal 紧贴绘图）
改    subagents/builtins/report_writer.py               （L2：step 4 seal 紧贴 write_file）
新增  tests/test_seal_gate_middleware.py                （L1 单元测试，红→绿）
```

---

## 附录 A：历史 4 次 seal 修复都在打地鼠（为什么这次不同）

| 次 | 判的根因 | 改法 | 复发因 |
|---|---|---|---|
| 1 | prompt 矛盾（空参数降级） | 改 prompt | 只修空参数路 |
| 2 | 非空参数走 a–f 长审计 | 又改 prompt | 只修那条触发路 |
| 3 | skill 教 write_file 与 seal 矛盾 | 改 skill | 只修那个 skill |
| 4 | schema 拒 list zone 参数 | 改 schema | 只修那个数据形状 |
| 今 | 空 statistics → 手算耗尽预算 | **L1 闸门**（不再改诱因） | —— |

前 4 次都修"这一次的诱因"，诱因无穷。L1 不修任何诱因，改终止条件本身——诱因再多，模型也物理上出不去。

## 附录 B：L2 根因（产出/交付分离，已逐字核实）

data-analyst prompt 把"产出分析"（step 2.7/2.8 思考成文）与"交付分析"（step 3 调 seal 誊抄）拆成两步，而 `seal_data_analyst_handoff` 参数恰是 `key_findings/outlier_findings/...`（与思考产出内容相同）→ 模型 step 2 做完就认为完成、跳过冗余的 step 3 seal。7+ 处"seal 必达"补丁是结构病指纹。L2 把两步合一（思考只推导、结论直接作 seal 参数），让模型自然一次过 L1 闸门。code-executor 的 run_metric_plan（一个工具即产出即落盘）已证明合一模式不漏，是 L2 的设计依据。

---

## 8. 一句话总结

seal 反复漏的真根因是 ReAct 允许"无 tool_call=结束"，prompt 改无法强制终止顺序。工程化根治用 langchain 现成的 `after_model + jump_to='model'`（本仓库 ParadigmIdentificationGateMiddleware 已是活范例）建 SealGateMiddleware：模型想纯文本结束但 seal 没调时强制导回，让 seal 成为 subagent 的唯一物理出口（L1，结构性 100%）；prompt 产出/交付合一让模型一次过闸（L2，省 turn）。不加任何兜底，不改 code-executor。
