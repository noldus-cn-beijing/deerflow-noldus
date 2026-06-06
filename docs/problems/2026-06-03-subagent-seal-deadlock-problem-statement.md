# 问题说明：subagent 反复「terminated without emitting handoff」—— 给独立 agent 分析用

> **本文档用途**：把 FST dogfood 反复出现的 `data-analyst terminated without emitting handoff_data_analyst.json` 卡死问题，**自包含、证据完整地**写清楚，交给其他 agent **独立分析、各自提方案**。本文档**只陈述事实与候选方向，不预设结论**。请基于证据自行判断根因与最优解。
>
> **写作背景**：这个问题在 2026-05-29 ~ 2026-06-03 至少 6 个 thread 复发，已被多次误诊（详见 §6 "历史误诊记录"）。本轮通过两次完整 dogfood transcript 对比，把根因钉到了一个新位置。请勿假设过去的诊断正确——以本文档的一手证据为准。

---

## 0. 一句话问题

`data-analyst` 这个 subagent（行为学数据洞察角色）做完了全部分析推理，**却没有发出 `seal_data_analyst_handoff` 工具调用**，撞上 `max_turns=12` 的硬限被终止，导致下游拿不到 handoff 文件、整个 FST 分析流水线卡死。报错信息说"forgot to call the seal tool"，但这只是表象——需要解释的是**为什么一个有能力调工具的模型，反复在该调的时候没调**。

---

## 1. 系统背景（独立 agent 必读）

### 1.1 架构
- 这是一个基于 **DeerFlow（LangGraph agent 框架）** fork 的项目 EthoInsight，做行为学实验数据的 AI 分析。
- 分析流水线：`lead agent` → `code-executor`（按 plan 跑指标脚本）→ `data-analyst`（统计审核 + 洞察）→ `report-writer`（出报告）。
- 每个 subagent **必须在结束前调用各自的 `seal_*_handoff` 工具**，把结构化结论落库成 JSON 文件（`/mnt/user-data/workspace/handoff_<role>.json`），下游 subagent 靠读这个文件接力。

### 1.2 关键文件路径（均为绝对路径，可直接读）
- subagent 执行引擎：`/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/subagents/executor.py`
- data-analyst 定义 + system prompt：`/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- seal 工具定义：`/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`
- handoff schema（Pydantic 校验）：`/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
- 模型配置：`/home/wangqiuyang/noldus-insight/packages/agent/config.yaml`

### 1.3 模型
- lead 与 data-analyst 都用 **`deepseek-v4-pro`**，配置里 `supports_thinking: true`（是 reasoning 模型）。
- data-analyst 配置 `model="inherit"`（继承 lead 的模型），`max_turns=12`，`timeout_seconds=600`。

---

## 2. 触发场景（本轮的具体数据）

用户 dogfood 大鼠 FST（Porsolt forced swim test，XT190 模板），**n=1/组**（Drug treatment 组 1 只 vs Saline control 组 1 只）。

真实 EV19 导出文件**没有 Activity 列**，immobility 走 EthoVision 自带的 `mobility_state` 路径。这导致 catalog 注入的 12 个 pendulum/velocity 参数**一个都没参与实际计算**。

经核实真实 thread（`c882d226`，2026-06-03 10:50）的实际文件：
- 6 个 `m_*.json`（compute 输出）：每个 `parameters_used: {}`（空）✅
- `handoff_code_executor.json` 的 `metrics_summary`：6 个指标条目，**每个都是 `parameters_used: {}`** ✅
- `handoff_data_analyst.json`：**不存在**（data-analyst 死在封存前）❌

> **重要**：上游两层"幽灵参数"修复（compute 脚本不再回显全量参数 + code-executor 不再从 plan 回退）**已经生效**——`parameters_used` 确实是 `{}` 了。但卡死照旧。**所以根因不在"幽灵参数还在被报告"，而在别处。**

---

## 3. 决定性证据：两次 dogfood transcript 对比

同一份数据连跑两次（lead 自动 retry），**第一次失败、第二次成功**。这个对比是定位根因的关键。

### 3.1 第一次（失败）
data-analyst 的 thinking/正文里完整地：
1. 读了 handoff、plan、范式文档；
2. 正确识别 `metrics_summary` 的 `parameters_used` 全为 `{}`，并明确写道：*"So step 2.8 says 跳过 parameters_used 为空 {} 的 metric... there's nothing to audit"*；
3. **但随后又自己改变主意**，从 `plan_metrics.json` 的 `parameters_in_use` 把 12 个参数捞回来，决定"做个 Phase 1 降级参数审计"，并**反复起草、重写** 3 条 `parameter_audit_findings`（velocity_threshold、pendulum_periodicity_threshold、pendulum_analysis_window……每条都纠结 mismatch_kind 填哪个、suggestion 写什么）；
4. 把完整的 seal JSON **用自然语言在正文里草拟了两三遍**；
5. 最后写下：**"现在我已经掌握了全部数据，让我来完成分析和封存。"** —— **然后就停了，从未发出 `seal_data_analyst_handoff` 工具调用。** 撞 `max_turns=12` 终止。

### 3.2 第二次（成功）
lead 重派时，在任务描述末尾**多加了一句话**：

> 提醒：你必须在完成分析后调用 seal_data_analyst_handoff tool 才能落库 handoff JSON，不能只在 thinking 里写"封存"或"已完成"。

data-analyst 这次**同样啰嗦**（推理量甚至更多，同样从 plan 捞了参数纠结半天），但在结尾**真的发出了 `seal_data_analyst_handoff` 工具调用**，文件落库，流水线继续。

### 3.3 对比结论
| 维度 | 第一次（失败） | 第二次（成功） |
|---|---|---|
| 叙述/思考量 | 极多 | 更多 |
| 是否在 12 turn 内完成了分析推理 | 是 | 是 |
| 从 plan 捞幽灵参数纠结 | 有 | 有 |
| **最后一下做了什么** | **写"让我来封存"文本** | **真调 seal 工具** |
| 差别诱因 | —— | **prompt 多了"必须调 tool，不能只写文本"** |

**两次差别不在叙述长度，而在"最后有没有把光标交给工具调用"。** 一句"写文本 ≠ 调工具"的直白提醒就翻转了结果。

---

## 4. 机制事实（已读代码核实，附 file:line）

> 以下是机器层面的客观事实，请独立 agent 据此推理，不要只信本文档的归因。

### 4.1 turn 预算按「AIMessage 条数」算，且不看是否调工具
`executor.py` 主循环（约 888–911 行）：每出现一条 `AIMessage` 就 `result.ai_messages.append(...)`，到 `len(result.ai_messages) >= self.config.max_turns` 就 `break` 硬终止。

```python
if not is_duplicate:
    result.ai_messages.append(message_dict)
    # Hard limit: terminate early when AI message count reaches max_turns
    if len(result.ai_messages) >= self.config.max_turns:
        logger.warning(... "reached max_turns ... terminating early")
        break
```

**关键**：这段代码**完全不检查这条 AIMessage 里有没有 `tool_calls`**。一条纯叙述消息和一条调了工具的消息，各算 1 个 turn，等价消耗预算。

### 4.2 subagent 一律被强制关闭 thinking
`executor.py` `_create_agent`（约 642 行）：

```python
model = create_chat_model(name=model_name, thinking_enabled=False)
```

**所有** subagent（含洞察型的 data-analyst）都 `thinking_enabled=False`，无差别。但模型本身 `supports_thinking: true`。

> 待独立 agent 判断的开放问题：对一个 reasoning 模型关掉 thinking 通道，它的推理是"消失"还是"漏成普通正文 AIMessage（从而占 turn）"？这直接关系到 §4.1 的预算是否被外漏推理吃掉。

### 4.3 没有任何 `tool_choice` 强制
全 harness grep `tool_choice`，**唯一一处是注释**（executor.py:712），明确说明**不**强制 tool_choice，理由是"探针证明强制 tool_choice 会产空 args"（把"无 handoff"换成"空 handoff"，更糟）。即模型自由选择调不调工具。

### 4.4 seal 工具的参数 schema
`seal_handoff_tools.py:284-328` `seal_data_analyst_handoff`：**只有 `status: str` 是必填**，其余 `key_findings / outlier_findings / method_warnings / recommendations / gate_signals / quality_warnings / parameter_audit_findings ...` 全是可选（默认 None/[]）。`analysis_config_id` 由工具自动注入，模型不用填。schema 校验走 `DataAnalystHandoff`（handoff_schemas.py）。内容校验（Sprint 5.5）要求 `key_findings` 非空。

> 即：seal 工具本身**调用门槛很低**（一个必填字段）。不是"参数太复杂填不动"。

### 4.5 兜底机制 seal-resume（已存在，但这次没救回）
`executor.py:701-795` `_attempt_seal_resume`：检测到 handoff 文件缺失时，追加一句 HumanMessage（"你上面的分析已经完成。现在请调用 {seal_tool} 工具…这是最后一步"），**再跑一轮 astream**。

它救不回的原因（待独立 agent 确认）：这一轮**依然没有 tool_choice 强制**，模型可以**再次用叙述填满这一轮**而不调工具；且它仍受同一 recursion_limit/turn 计数约束。有前置守卫：history 里没有任何 AIMessage 时直接放弃（"无米下锅"）。

### 4.6 step 2.8 的 prompt 逻辑（叙述黑洞所在）
`data_analyst.py` step 2.8（约 113–144 行）：
- 第 114-115 行：从 `metrics_summary` 取，**"跳过 parameters_used 为空 `{}` 的 metric"**。
- **但** 第 126-136 行的"Phase 1 降级路径"说："per_subject 缺该 metric 条目，或跨 subject 标量值 < 2 → 记一条 info finding"，且"used_value 从 parameters_used[参数名] 取，绝不填 None"。
- **而且** 第 226-229 行另外明确指示 data-analyst **去 read `plan_metrics.json`**（里面有全量 `parameters_in_use`）。

→ 这给了模型一个"绕过 §4.6 第一句跳过指令"的材料（plan 里有非空参数）和理由（降级路径要它每个参数记 finding）。两次失败的叙述黑洞都发生在这里：模型从 plan 捞参数、纠结 mismatch_kind/suggestion，写一大坨。

### 4.7 executor.py 是受保护文件
`scripts/sync-deerflow.sh:62` 把 `subagents/executor.py` 列入 `PROTECTED_FILES`。**改它属于 Noldus 定制，未来上游同步要 surgical 守护**——动它有维护成本，请在方案里权衡。

---

## 5. 候选修复方向（不预设优劣，供独立 agent 评估 / 补充 / 反驳）

> 以下是本轮已想到的方向。**请独立判断它们各自治标还是治本、风险、相互关系，并欢迎提出本文档没列的方向。**

**方向①：prompt 硬区分「写文本 ≠ 调工具」**
把 §3.2 实证成功的那句话写进 data-analyst 默认 system prompt 最显眼处（step 3 + 开头），让第一次派遣就带着它。正面措辞（项目铁律：deepseek 用正面指令，禁用"不要 X"反向激活，见 CLAUDE.md §6）。
- 杠杆：直接命中 §3 的"最后没把光标交给工具"。低风险（只改 prompt 文本）。
- 疑问：是否够稳？它依赖模型遵守提醒，而"加提醒"在历史上多次失败过（见 §6）——但那些是"提醒调 seal"，本方向是"区分文本 vs 工具"，语义不同。

**方向②：堵 step 2.8 逻辑死字**
当 `parameters_used` 全为 `{}` 时，直接 0 finding、立即进 step 3 调 seal，**明令不得从 `plan_metrics.json` 捞参数补降级审计**。
- 杠杆：削减 §4.6 的叙述黑洞（占了失败叙述的一大半），给 turn 名额松绑。低风险（改 prompt + 可能调降级路径触发条件）。
- 疑问：降级审计在"真有 parameters_used 但判据不足"时仍需保留——如何只关掉"空 {} 也硬审计"这条，不误伤正常降级？

**方向③：提高 max_turns（12 → 16~18）**
给"想完还能调工具"的余量。
- 杠杆：缓解边界 case 撞墙。低风险（改一个常量）。
- 疑问：这是治标——若根因是"最后没调工具"或"推理外漏占 turn"，单加预算只是让它多想几次再撞墙。是否必要取决于①②④的效果。

**方向④：给洞察型 subagent 单独开 thinking（治本候选，碰承重墙）**
改 `executor.py:642`，让 data-analyst/report-writer 这类**需要洞察**的 subagent `thinking_enabled` 按配置开启，code-executor 这类"按 plan 跑脚本不需洞察"的保持关闭。
- 论据：data-analyst 职责就是深度推理洞察（prompt 第 217 行"主动提出洞察"）。若 §4.2 的判断成立（关 thinking → 推理外漏成占 turn 的正文），那么开 thinking 后推理回到 thinking 块 → **不占 turn 名额** → 预算自动宽裕 + 对话历史/handoff 干净 + 洞察质量更高。这可能让方向③变得不必要。
- 风险/必须验证：
  1. deepseek-v4-pro 开 thinking 后，thinking 块**是否确实不计入** `result.ai_messages` 的 turn 计数？（取决于 `PatchedReasoningChatOpenAI` 如何处理 reasoning 字段、是否产生独立 AIMessage）—— **这是方向④成立与否的关键，必须实测验证。**
  2. 开 thinking 显著增加延迟 + token，data-analyst 本就慢。
  3. `executor.py` 是受保护文件（§4.7），改动有同步维护成本。
  4. 开 thinking **不直接保证**模型最后一定调工具（§3 的"混淆"根因可能仍在）——可能仍需配合①。

**方向⑤（更深，本轮倾向不做，仅记录）：改 turn 计数语义**
让"纯叙述、零 tool_call 的 AIMessage"在接近上限时触发一次硬提示，或不计入硬限。
- 治本但碰 `executor.py` 核心循环承重墙，风险高。本轮证据显示 prompt 一招似乎够用，故列为后备。

---

## 6. 历史误诊记录（避免重蹈覆辙）

- **2026-05-29 ~ 06-02**：多次归因为"data-analyst 在 step 2.8 参数审计陷入死循环烧光预算"，并修了上游两层"幽灵参数"（compute 回显 + code-executor 从 plan 回退）。**本轮证实那两层修复确实生效（parameters_used 已为 {}），但卡死照旧**——说明"上游不再报告幽灵参数"不是充分解。
- **反复加"提醒调 seal"无效**：至少 6 个 thread 加各种提醒均失败。但注意：本轮 §3.2 成功的那句是"**区分写文本 vs 调工具**"，与单纯"记得调 seal"语义不同，不要混为一谈。
- **探针结论：强制 tool_choice 会产空 args**（executor.py:712 注释）——把"无 handoff"换成"空 handoff"，被认为更糟。所以历史上**刻意没用** tool_choice。任何用 tool_choice 的方案需直面这一点。

---

## 7. 给独立分析 agent 的请求

请基于以上证据，独立回答：

1. **根因判定**：你认为 data-analyst 反复不调 seal 工具的**根本原因**是什么？（混淆"文本 vs 工具"？推理外漏吃 turn 预算？step 2.8 叙述黑洞？turn 计数语义？以上组合？还是别的？）请说明你的证据链。
2. **最优解**：在 §5 的候选方向（及你自己补充的方向）里，你会选哪个/哪些组合？为什么？尤其请评估**方向④（开 thinking）是否是真正的治本招**，以及它与方向①②③的关系（互补？替代？）。
3. **关键验证**：方向④的成立依赖"thinking 块不计入 turn 计数"——你会如何设计一个最小实验来验证这一点（在动承重墙之前）？
4. **风险**：你的方案有没有引入新风险（如开 thinking 后延迟爆炸、tool_choice 产空 args、上游同步冲突）？如何缓解？

请直接给出结论与推理，不必客气。
