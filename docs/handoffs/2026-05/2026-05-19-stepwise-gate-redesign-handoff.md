# 2026-05-19 数据飞轮 stepwise gate 重做 Handoff

> **决议(2026-05-20 grill-me 复盘)**:**v0.1 不做 stepwise gate**。删 PR-2 实现 + 9 个测试。
>
> 复盘逻辑见文末「2026-05-20 决议」一节。本文档前半部分保留为历史草案,**不要再按照前半部分实施**。

---

## 触发场景 — 这次 dogfood 拍下来的真实流水

用户开**数据飞轮 mode**,跑 EPM 单被试 E2E,顺序是:

1. lead 派 task(code-executor) → code-executor 跑 5/5 EPM 指标 → ToolMessage 写盘 ✅
2. lead 一条 AIMessage 同时含:
   - 一段 narrative "指标计算完成。现在派遣 data-analyst 解读和 chart-maker 出图"
   - **并行 2 个 task tool_call**:`task(data-analyst)` + `task(chart-maker)`
3. **当前 PR-2 StepwiseGateMiddleware 在 `wrap_tool_call` 阶段拦截**:看到上一条是 task ToolMessage 而本条又是 task → 返回 `Command(goto=END, update={"messages": [_GATE_TOOL_MESSAGE_CONTENT 占位 ToolMessage]})`,两个 task tool_call 都被拦,**subagent 实际从未被调度**
4. 前端流式结束,但:
   - 用户看到 "🔬 指标已完成,正在请专家解读..." + "🛠 正在派遣 chart-maker..." 两张 SubtaskCard,**永久停在"运行中"动画**(B7),因为 SubagentExecutor 从未发出 task_started / task_completed 事件
   - 用户不知道该怎么办,以为还在跑
5. 用户重发"需要" → lead 看到 history 里 2 个 `[gate]` 占位 ToolMessage,**误判 subagent 失败**(B1),自己兜底总结 5/5 指标 → 又重新派一次 chart-maker → 这次因为 stepwise gate 上次已经写过"先一次 task" → 又被拦 / 或者直接放行

---

## 4 个相互纠缠的 bug

| # | 现象 | 根因 |
|---|---|---|
| **B0** | lead 一次性派 2 个 task,不给用户机会评审上一棒 | prompt 没强制"派下一棒前先报告完整 progress + 等用户指令";SubagentLimit 也允许并行 |
| **B1** | Gate 拦截 task 后,state 留下 "AIMessage 派 task + [gate] 占位 ToolMessage"。下一轮 lead 读到这些占位**误判 subagent 失败**,自己兜底总结 | StepwiseGateMiddleware 拦在 pre-tool-call (`wrap_tool_call`),subagent 没真跑出去,占位 ToolMessage 又被 LLM 当成失败信号 |
| **B2** | UI 不知道 "已暂停,等用户继续",流式结束就结束了 | gate 占位 ToolMessage 没带任何前端可识别的 marker;前端 hooks 不知道 graph 是因 stepwise 暂停的还是正常结束 |
| **B7** | 死亡 subagent 卡片永久转圈 | B1 的另一面:subagent 没真跑,SubagentExecutor 没发 task_completed / task_failed,前端 task 卡片永远是 in_progress |

**4 个 bug 同根** — 拦截点选错了。

---

## 已对齐的设计决策 (grill 出来的 + 用户拍板的)

### D1 — 暂停时机
**subagent 完成后,lead 拿到 task ToolMessage → 下一次 LLM 调用之前 END**。

理由:subagent 实打实跑过、卡片有终态 (修 B7),lead 这一轮不被唤醒就不会无脑派下一棒 (修 B0/B1)。

### D2 — Lead 在暂停轮**不发言**
不补 narrative,不额外起 LLM 调用。用户看到的是 **subagent 自己产出的"执行摘要 + [gate_signals]"块**(code-executor 已经在写了)。

用户进一步要求:**通用 case 下的"update 进度"是另一条独立问题** — 不是 stepwise 专属,是所有模式都该有的"lead 拿到 task ToolMessage 后必须用人话报告 1-2 个关键数字 + 下一步打算" → 这个已经通过 prompt 改动落地在 PR #14 候选 (A-update),不在本次 stepwise gate 重做 scope 内。

### D3 — Hook 点
**`before_model` hook**(lead 下一轮 LLM 调用之前),不是 `wrap_model_call` (后者返回类型不允许 Command)。

返回 `Command(goto=END, update={"messages": [<pause marker>]})`。

> ⚠️ 验证项:写实现时确认 LangGraph 0.7.65 + langchain agent base 的 `before_model` 是否真接受 `Command(goto=END)` 返回 (CLAUDE.md 第 12 条提"复用 deerflow 现成功能优先",ClarificationMiddleware 在 `wrap_tool_call` 用过 Command;before_model 是否同等支持要测)。

### D4 — Pause marker 形态
**Backend 仅发结构化 pause event,前端渲染进度卡**(不动 LLM)。

形式:在 messages 里 append 一条特殊 ToolMessage(或专门 marker AIMessage),`additional_kwargs` 含:
```json
{
  "deerflow_pause_event": "stepwise_gate",
  "subagent_type": "code-executor",
  "gate_signals": "<subagent 自己产出的 [gate_signals] 原始块字符串>"
}
```
前端识别 `deerflow_pause_event` 字段渲染 "✅ <subagent> 完成 + 关键指标 + '请阅完后发下一条消息继续'" 卡片 (修 B2)。

字段集**最简版**(D4 子选项 1):只含 subagent 名 + gate_signals 原文。**不**包含 next_step_hint 推荐 — 让用户自己决定下一棒。

### D5 — 拦截范围
**任意 task ToolMessage 都触发暂停**(D5 子选项 1)。

不维护白名单,简单。未来加 general-purpose subagent 时若不想暂停,再加 opt-out。

### D6 — 并行派遣处理
**允许并行**(lead 一次派 2-3 个 task),不强制串行 — "严格 1 个 = workflow,失去 orchestration 弹性"。

但暂停时机要等**所有并行 task 都完成**:`before_model` hook 扫描上一条 lead AIMessage(含 task tool_calls)的所有 task,确认每一个都有对应 ToolMessage,才 END。否则放行 LLM 继续(让 lead 自己等剩下的 subagent 完成)。

### D7 — 模式开关
**仅 `workflow_mode == "manual"` (数据飞轮 mode) 触发**,auto 模式完全不挂这个中间件。

跟当前 PR-2 行为一致 — 全自动模式 lead 该连续派就连续派。

### D8 — 首棒 task 不暂停
HumanMessage → AI(task) → ToolMessage → **暂停**(这是第 2 棒之前)。
HumanMessage → AI(task) → 不暂停(这是首棒,直接派出去)。

判定方法:`before_model` 时扫 messages 从最近 HumanMessage 起,如果**已有 task ToolMessage**(说明至少一棒跑完了),就该暂停。

---

## 还没 grill 完的空白点

### G1 — `before_model` 返回 Command 的实际行为
LangGraph 0.7.65 + langchain agent 0.x 的 base class `before_model` 签名是 `dict[str, Any] | None`。**但 LangGraph node 任何 callable 都可以 return Command(goto=END)** — 验证项,写代码前用一个 demo middleware 跑一遍确认。

如果 before_model 不接受 Command:Plan B 是用 `wrap_tool_call` 但**仅拦截 lead 派的 ask_clarification 或新增的 `pause_for_review` 工具**,让 lead 在派下一棒前自己主动 call 这个工具(deerflow 已有的 ClarificationMiddleware 模式)。

### G2 — Pause marker 是 ToolMessage 还是 AIMessage 还是新 message type?
- ToolMessage:配对到不存在的 tool_call_id 可能违反 OpenAI / Moonshot 严格配对;但 ClarificationMiddleware 用的就是 ToolMessage 配 ask_clarification tool_call_id,看那个能不能复用
- AIMessage:更干净,带特殊 additional_kwargs 渲染 — 但 LLM 看历史 message 时会把它当成 lead 自己说过的话,可能影响后续推理
- 新 message type:不可行,LangChain message types 是固定的

**倾向**:复用 ClarificationMiddleware 模式 — 把 pause-marker 写成一条 ToolMessage,`tool_call_id` 指向上一条 task tool_call 之一 (语义上"这个 task 已完成,等用户响应")。`additional_kwargs` 带 `deerflow_pause_event` flag。

### G3 — 删 vs 改造 StepwiseGateMiddleware?
- **删** + 新建 `StepwiseGateAfterCompletionMiddleware`:语义跟 PR-2 截然相反,新名字更清晰
- **改造**:同一个类,改成 `before_model` 实现 + 改文档

**倾向**:改造,因为前端 hooks.ts L688 的 `workflow_mode: context.mode === "flywheel" ? "manual" : "auto"` 桥接 + lead_agent/agent.py 的中间件挂载点都不用动,只换内部实现。

### G4 — UI 怎么从"流式结束"识别这是 stepwise 暂停而非正常完成?
前端 hooks 收到 `end` 事件时,看最后一条 message 是否含 `additional_kwargs.deerflow_pause_event === "stepwise_gate"` → 是的话:
- 不清空 streaming state(避免"输入框等待中"切回"空闲"的视觉)
- 在 message group 末尾渲染"📌 已暂停 · 请阅读上方结果后发下一条消息继续"提示条
- 不要禁用输入框(用户随时可发下一条)

### G5 — 用户发下一条之后,gate state 怎么 reset?
trivial:HumanMessage 进来 → 不再有"上一条是 task ToolMessage"的情况了 → before_model 自然放行。无需显式 reset。

### G6 — 如果 subagent 失败了(real 失败,不是 gate 占位),还暂停吗?
**应该暂停**。subagent task_failed 也是个 ToolMessage,内容是错误信息。用户需要审阅错误决定怎么办 (重派 / 改 prompt / 放弃)。

但 pause-card 应该显式标失败状态 (`additional_kwargs.subagent_status: "failed"`),前端用红色 / ⚠️ 渲染。

### G7 — auto mode 下 lead 连续派 5 棒,中间没暂停 — 这次 dogfood 想要的进度透明怎么办?
**A-update 已经覆盖了**:lead 收到 task ToolMessage 后必须先一行 progress 报告。auto 模式下 lead 不被打断,但每棒之间都有 1 行 narrative,用户能看见。

这是 stepwise 跟 auto-with-progress 的核心区别:
- **auto + A-update**:连续派,每棒之间有 1 行 lead narrative,用户被动看
- **manual + stepwise**:每棒之间 END,用户主动审阅 + 留 feedback + 发下一条

### G8 — 用户能不能"跳过暂停一气呵成"?
不在 v0.1 scope。如果要做,加一个用户输入 quick action ("继续到底"按钮),触发一条 hidden HumanMessage "继续后续所有 subagent 一气跑完不要再暂停",lead prompt 看到这条进 quick-mode (但这又是个 LLM 自觉,不可靠)。

---

## 删除项

- `StepwiseGateMiddleware._should_pause` 当前的 wrap_tool_call 拦截逻辑 — 全删
- `_GATE_TOOL_MESSAGE_CONTENT` 占位文案 — 替换成"已暂停,请审阅 + 留反馈 + 发下一条消息"专用文案,但**不再当 tool_call 拦截结果出现**

## 不动的

- 前端 `hooks.ts:688` 的 `workflow_mode: context.mode === "flywheel" ? "manual" : "auto"` 桥接 (PR-1 修过)
- `lead_agent/agent.py` 的 middleware 挂载逻辑 (条件挂载,改造类内部即可)
- `ClarificationMiddleware` (管 ask_clarification,语义跟本次 stepwise 正交,别动)
- `Ev19TemplateGuardrailProvider` (管 ev19_template 字段,正交,别动)

---

## 实施步骤草案 (等 grill 完 G1-G6 再细化为 plan)

1. **Spike (1h)**:写一个最小 demo middleware,验证 `before_model` return `Command(goto=END)` 在 LangGraph 0.7.65 真能 stop graph
   - 否 → Plan B 走 ask_clarification 模式
2. **改 StepwiseGateMiddleware (2-3h)**:删 wrap_tool_call,加 before_model。pause-marker 写成带 `deerflow_pause_event` 的 ToolMessage。9 个现有 test 全重写
3. **前端识别 pause-marker (1h)**:`hooks.ts` 流处理识别 marker,设 thread state 字段 `is_paused_for_review`;`message-group.tsx` 或新组件渲染"已暂停"卡片
4. **lead prompt 微调 (15min)**:manual mode 下加一行"流程会在每棒完成后暂停,你不用主动汇报或派下一棒,用户会发下一条消息让你继续" — 防止 lead 因为找不到下一步该做啥而胡乱 ask_clarification
5. **9 个 stepwise 单测 + 4 个 hooks.ts 测 (1.5h)**
6. **dogfood 验证 6 条清单**(同 PR-3 handoff,但加上 stepwise 专项)

---

## 参考

- 当前 PR-2 实现:[stepwise_gate_middleware.py](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/stepwise_gate_middleware.py)
- ClarificationMiddleware 复用模式:[clarification_middleware.py](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/clarification_middleware.py)
- PR-3 完成的 frontend hooks.ts 历史消息 run_id 兜底 (B7 修复需要 thread state 完整):[hooks.ts:766-915](../../packages/agent/frontend/src/core/threads/hooks.ts)
- 2026-05-19 PR #14 候选 (V/B5/B3/A-update — 黑箱感快速止血):分支 `worktree-lead-visibility-fix`
- 范式体系迁移 (EV19 模板 + 学术范式双层) 是独立轨道,别跟 stepwise gate 混:[2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md)

---

## 2026-05-20 决议:v0.1 不做,删除 PR-2

经 grill-me 复盘 (本文 + 用户问答 + 第一性原理审视),结论:**v0.1 不引入 stepwise gate**。

### 4 条 bug 逐条重判

| # | 原认定 | 重判 |
|---|---|---|
| **B0** | lead 一次性派 2 个 task | **不是 bug** — orchestrator agent 并行派 2-3 个 subagent 是 deerflow 设计意图(`MAX_CONCURRENT_SUBAGENTS=3`,`SubagentLimitMiddleware` 仅在 >3 时 truncate)。原 handoff 把它当 bug 是被 PR-2 烂实现污染后看到的"病急乱投医"症状误判。|
| **B1** | gate 占位 ToolMessage 让 lead 误判 subagent 失败 | **由 PR-2 制造,删 PR-2 即消失**。背后的 general 问题(lead 看似失败就硬写假结果)由 lead prompt 第 252 行"任何 subagent 失败 → 必须 ask_clarification,绝不静默 bypass"覆盖,跟 stepwise 正交。|
| **B2** | UI 不知道暂停 | **不需要 stepwise 引入新机制**。v0.1 真有需要暂停的瞬间(范式不明 / subagent 失败 / 风险决策)由 lead 调 `ask_clarification` + ClarificationMiddleware + 前端 ChainOfThoughtStep + ClarificationOptions 现有链路完整覆盖。|
| **B7** | 死亡 subagent 卡片永久转圈 | **PR-2 副作用**。PR-2 在 `wrap_tool_call` 拦截 task → task_tool body 不跑 → `task_started/running/completed` SSE 事件全不发 → 前端 SubtaskCard 永远 in_progress。删 PR-2 → task_tool 正常跑 → 卡片正常 transition。|

### 第一性原理结论

**核心不变量**:数据飞轮 mode 的两个意义 — (a) 用户审阅 + 纠偏、(b) 收集训练数据。

- 飞轮(b)**已经在跑**,`TrainingDataMiddleware` 自动录制每棒 + 前端 `<FeedbackButtons>` 三按钮反馈 + `/feedback` API + SQLite。stepwise gate **不是飞轮采集的必要条件**。
- 审阅(a)在 v0.1 用户(EthoVision 研究员)的实际工作流里**几乎没有"第二棒之前要改方向"的真实需求** — 用户上传文件 → 选范式 → 等完整报告 → 看结论 → 重跑。中途插手是 v0.1 之后专家 review 阶段的事。

→ **stepwise gate 是 v0.1 之后的优化,不是 MVP 必需**。

### 实施清单(2026-05-20 完成)

| # | 动作 | 状态 |
|---|---|---|
| 1 | 删 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 里 `if workflow_mode_value == "manual": StepwiseGateMiddleware` 块(原 line 362-372,12 行) | ✅ |
| 2 | 删 `packages/agent/backend/packages/harness/deerflow/agents/middlewares/stepwise_gate_middleware.py` 整文件 | ✅ |
| 3 | 删 `packages/agent/backend/tests/test_stepwise_gate_middleware.py` 整文件 (9 个测试) | ✅ |
| 4 | 改 `packages/agent/frontend/src/core/threads/hooks.ts:683` 注释 — 去掉 `future StepwiseGateMiddleware` 字样,workflow_mode 桥接保留(GateEnforcementMiddleware 还在用) | ✅ |
| 5 | `make test` 验证:33 fail / 2666 pass / 15 skip,对比 baseline 33 fail / 2675 pass / 15 skip,**差异 = 删掉的 9 个 stepwise 测试,无新增 fail** | ✅ |

### 保留不动

- `GateEnforcementMiddleware`(Gate 1 范式反问,manual mode 触发)— 跟 stepwise 正交,继续工作
- `ClarificationMiddleware` + `ask_clarification` 工具 — v0.1 真要暂停时 lead 主动调用的现成机制
- `TrainingDataMiddleware` + `<FeedbackButtons>` + `/feedback` API + SQLite FeedbackRepository — 训练数据飞轮持续运转
- 前端 `workflow_mode: context.mode === "flywheel" ? "manual" : "auto"` 桥接 — GateEnforcement 还要用,不要删
- PR #14 的 A-update("lead 每棒必须先报告进度 + 下一步")prompt 改动 — auto mode 也受益,跟 stepwise 无关,保留

### V/B5/B3 临时止血改动

PR #14 候选里 V/B5/B3 三个是针对原 stepwise UI 黑箱感的临时补丁。stepwise 不做了,这些止血改动**也应回退或重审**(单独评估,看哪些在 auto mode 仍有价值)— 不在本次 stepwise 删除 scope 内,留待下一次 dogfood 复盘。

### v0.1 之后再做 stepwise 时的设计指针

如果未来真要做(专家 review 阶段),正确路线是:

1. **不用** `wrap_tool_call` 拦 task(会引入 B7:task_tool body 不跑 → SSE 事件不发 → 卡片永久 in_progress)
2. **用** `before_model` hook + LangChain agent 1.x 的 `{"jump_to": "end"}` + `@hook_config(can_jump_to=["end"])` 装饰器(原 handoff D3 的 `Command(goto=END)` 是错的 — Command 是 ToolNode 返回类型,before_model 不接受)
3. ThreadState 加 `is_paused_for_review: bool` 字段让前端识别,**不要** append marker ToolMessage(没合法 tool_call_id 可配,污染 LLM 历史)
4. UI 暂停卡片**完全复用 `<FeedbackButtons>`**,不引入新评价控件;quick action 按钮(继续/停止)v0.x 不做,用输入框够用
