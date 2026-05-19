# 2026-05-19 数据飞轮 stepwise gate 重做 Handoff

> **状态**:**实施未启动**。今天 dogfood 暴露 4 个相互纠缠的 bug (B0+B1+B2+B7),先用一拨 prompt-only 改动 (V/B5/B3/A-update,见 PR #14 候选) 把"用户感觉黑箱"先压住,本文档是**后续把数据飞轮 stepwise 模式做对**的设计草案 — 含已对齐的决策 + 还没 grill 完的空白点。

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
