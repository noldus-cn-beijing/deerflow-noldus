# Spec 7 (P7): 数据降级熔断器 —— middleware 拦降级 → 通知 + 有限自救 → HITL 确认 → 重试

> 日期：2026-06-17
> 类型：新机制（红线一的核心落地 —— 数据分析 agent 的可复现性底线）
> 触发源：用户 2026-06-17 产品级立场 —— "这是实验数据分析 agent，数据计算的可复现性和准确性必须保证。降级可以但静默不可以：降级要通知用户、尽量让模型自己 figure out（但有时间上限）、靠 middleware 拦住和用户确认细节再重试。数据分析的自然语言表述不可复现无所谓。"
> 状态：待 review → 批准后 worktree 实施。**这是 P2 的升级替代**——P2 的"fail-loud 报错"让位给本 spec 的"降级→熔断→HITL→重试"，更符合用户立场。P2 的"亮 statistics_status 信号"作为本 spec 的信号源保留。
> **遵循工程实践**：[红线一 —— 降级可以，静默不可以；降级要通知 + 有限自救 + HITL 兜底](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线一失败必须响亮数据层失败必须-fail-loud-触发自愈hitl禁止静默降级)。

---

## 0. 产品立场（为什么需要这个机制）

EthoInsight 是**实验数据分析 agent**，将来是完整的实验数据 agent。对研究员，**数据计算的可复现性和准确性是底线**——一份统计崩溃却伪装成"仅描述性报告"的产出，会让研究员拿残废结果写论文。这比报错更有害。

但"数据崩了就直接 failed 摆烂"也不对——很多降级是可以接受的（换检验方法、排除异常 subject、用描述性兜底），**只要：(a) 通知用户 (b) 让模型先试着自己解决 (c) 自救超时就停下来问用户**。

划界：
- **必须可复现/准确**：数值计算（统计量、指标值、p 值、效应量）。
- **不必可复现**：判读的自然语言表述（措辞每次不同，无所谓）。

## 1. 现状问题（gateway.log 实证）
statistics ZeroDivision 崩 → 静默降级成 `statistics={}` → data-analyst 走描述性手算螺旋 → 残废报告。**三宗罪：没通知、没有限自救（反而无限手算螺旋）、没 HITL。**

## 2. 机制设计：DegradationCircuitBreakerMiddleware（复刻现成范式）

用户给的机制：**middleware 发现降级 → 先停住 → 和用户确认细节 → 再重试**。这不需要发明新东西，复刻两个现成件：
- `SealGateMiddleware` / `GateEnforcementMiddleware`：`after_model` hook + `@hook_config(can_jump_to=...)`，能在终止前拦截、改写流向。
- `ClarificationMiddleware` + `ask_clarification` 工具：中断执行、把问题呈现给用户、等回答（clarification_tool.py:29）。

### 2.1 降级信号源（P2 提供）
数据子步降级时，handoff/gate_signals 亮**机读降级信号**（P2 的 `statistics_status="crashed"` 是其中一种）。通用化为 `degradation_signals[]`，每条含：`{stage, kind: "crashed"|"method_fallback"|"subject_excluded"|..., reason, reproducible: bool, user_visible_summary}`。

> 区分：`crashed`（数据计算崩溃，可复现性受损，必须熔断）vs `absent_by_design`（单组本就无推断统计，正常缺席，**不熔断**）vs `method_fallback`（如正态→非参数，正常自动决策，**不熔断**，但通知）。只有**损害可复现性/准确性的降级**才触发熔断。

### 2.2 熔断器（新 middleware，复刻 SealGate）
`DegradationCircuitBreakerMiddleware`：
- **监听**：subagent 完成（或 lead 收到 subagent handoff）时，检查 `degradation_signals` 里有没有 `kind=crashed`（或其它需熔断的降级）。
- **有限自救**：第一次遇到 → 不立刻打断用户，而是给模型**一次有限的自救机会**（注入一条提示"统计层降级:<reason>，请检查是否可重试/换参数/重新派 code-executor，**单次自救，勿手算**"），并 `jump_to` 让模型重试。
- **自救上限**：用计数器（state 里记 `degradation_retry_count`），**超过 1-2 次自救仍降级 → 不再自救**，转 HITL。这堵住"无限重试/手算螺旋"（用户的"figure out 要有时间上限"）。
- **HITL 兜底**：自救超限 → 拦住，强制走 `ask_clarification`，把降级细节呈现给用户："统计计算出现 <问题>，可能是 <数据 X 异常 / 参数 Y 需调>。是否：① 检查数据后重试 ② 接受降级（仅描述性）③ 调整参数 <…>？" 用户回答后按选择重试或接受。

### 2.3 与 data-analyst 手算螺旋的关系
本机制从根上掐断手算螺旋：data-analyst 收到 `crashed` 信号时，**不进描述性手算路径**（那是降级且不通知），而是被熔断器拦住 → 自救一次 → 还崩就问用户。手算螺旋（反复验算 28 subject）再也不会发生，因为"statistics 崩了"不再被当成"本次没统计、自己描述性补上"。

## 3. 实施范围与分层
- **L1（结构主防线，新 middleware）**：DegradationCircuitBreakerMiddleware，监听降级信号 + 有限自救计数 + 超限转 HITL。复刻 SealGate 的 after_model+jump_to。
- **L2（信号源，依赖 P2）**：数据子步崩溃时亮 `degradation_signals`（P2 已做 statistics_status，本 spec 通用化）。
- **L3（prompt 辅助）**：data-analyst/lead prompt 说明"收到 degradation 信号时配合熔断器，勿自行降级糊弄"。L3 是辅助，主防线是 L1。

## 4. 测试
- `test_crashed_signal_triggers_self_heal_once`：注入 crashed 信号 → 断言第一次给自救机会（jump_to model + retry count=1），不立刻打断用户。
- `test_self_heal_exhausted_triggers_hitl`（**核心**）：连续 crashed 超过自救上限 → 断言转 ask_clarification（中断 + 问用户），不再无限重试。
- `test_method_fallback_not_circuit_broken`：正态→非参数这类正常 method_fallback → 通知但不熔断（不打断用户）。
- `test_absent_by_design_not_broken`：单组无统计 → 正常 partial，不熔断。
- `test_no_handcalc_spiral`：data-analyst 收到 crashed → 不进描述性手算路径（被熔断器接管）。
- **端到端（真实数据，红线三）**：用 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（含 Trial 19=0.0），P1 未修时触发 crashed → 断言熔断器拦住并走 HITL（而非静默残废报告）；P1 修后 statistics 正常 → 不熔断。

## 5. 风险边界
- **不要把正常降级也熔断**：method_fallback（参数→非参数检验）、absent_by_design（单组）是正常的，只通知不打断。只有损害可复现性/准确性的降级（crashed/数据异常）才熔断。判据严格按 `degradation_signals[].kind`。
- **自救上限要小**（1-2 次），否则又变相纵容螺旋。用户明确"figure out 时间不应长"。
- **改 middleware 链**：DegradationCircuitBreaker 的位置要对（在 subagent handoff 可见之后、lead 终止之前），复刻 SealGate 挂载点。改完裸导入两入口（导入环铁律）。
- **HITL 中断的恢复**：ask_clarification 中断后用户回答，要能正确恢复到"按用户选择重试/接受"——复用 ClarificationMiddleware 现成的中断-恢复，别自造。
- 与 P2 关系：P2 提供信号（statistics_status），P7 消费信号做熔断。P2 先落地（信号源），P7 再落地（熔断器）。或合并为一个 PR（信号+熔断一起），实施者定。

## 6. 为什么这是数据分析 agent 的关键基础设施
将来扩展到完整实验数据 agent，会有更多数据计算环节（更多范式、更多指标、更复杂统计）。**每个环节都可能降级。** 有了通用的"降级→熔断→HITL"机制，新环节只需亮 `degradation_signals`，就自动获得"不静默、通知用户、有限自救、HITL 兜底"的保护——而不是每个新环节重新踩"静默降级出残废结果"的坑。这是把"可复现性/准确性底线"做成基础设施，而非靠每处自觉。
