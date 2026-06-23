# Spec：ETHO-9 部分反问缺快捷按钮（options）—— path_registry 声明 requires_options + guardrail 强制，DeerFlow 原生

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：🟢 低 · 确定性数据修复 + 交互一致性。`ask_clarification` 的 `options` 是可选参数，lead 自由决定传不传 → 同类反问有时给 A/B 快捷按钮、有时只给文本框。**确定性修复**（像 ETHO-8 候选排序）：让 path_registry 声明「哪些反问点必须带 options」，guardrail 强制。
> 拆分说明：原 handoff 把 ETHO-7+9 合并；本次按用户拍板**拆开**——ETHO-9 是纯确定性数据修复（Step 加字段 + gate 强制 options），ETHO-7（决策点数量/顺序漂移，涉 intent 分类稳定性）是更深的问题，另 spec。粒度不同，独立可先合。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-9；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - **生产实证**（2026-06-23 dump）：memory `reference_harnessx_report_and_etho_spec_application`——**e6ea7946 #5 范式确认反问 `options=无`（纯文本框）**，而同 thread 的「是否出图/出报告」反问都 `options=有(2)`。坐实 options 由 lead 自由传，关键反问反而漏。
> - SSOT：决策点路径在 `guardrails/path_registry.py` 的 `PATHS`（声明式，三消费者数据驱动，CI 哨兵 `test_path_registry_ssot.py` 守一致）。
> - **框架**：DeerFlow（LangGraph）原生——扩 `Step` frozen dataclass + `GuardrailProvider` 协议，**不引入 HarnessX 机制**（memory `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`）。
> - 既有约束：PR#175 已把 ASKVIZ deny message 文案对齐到 prompt SSOT；本 spec 改 gate 逻辑（options 检查），与文案正交不冲突。
> - 受保护文件：`path_registry.py` / guardrail provider 是 deerflow 定制面，sync 时 surgical。

---

## 〇、框架契合声明（DeerFlow-first）

HarnessX §7.1「类型系统不生成正确程序，但让错误程序可被检测」是本 spec 的思想支点——但**落地全用 DeerFlow 原生**：`Step` 是已有 frozen dataclass（加一个可选字段），`GuardrailProvider` 是 DeerFlow 现有协议，`PATHS` 是现有 SSOT。**零新抽象、零 HarnessX 机制**。HarnessX 只贡献「该用类型化结构而非 prompt 规则约束 options」这个判断 + change manifest / 病理自检的写法。

---

## 一、根因（逐字节 + 生产实证）

### 1.1 现象与生产实证

prod e6ea7946：
- `#5 ask_clarification: options=无` —— 范式确认反问，纯文本框。
- 同批「是否出图」「是否出报告」反问：`options=有(2)` —— 有 A/B 快捷按钮。

用户在缺 options 的反问处只能手敲，体验不一致 + 自动化脚本无法假设固定选项。

### 1.2 真因：options 是可选参数，lead 自由传

[`clarification_tool.py:17`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/clarification_tool.py#L17)：
```python
def ask_clarification_tool(..., options: list[str] | None = None, ...):
    # docstring: "Optional list of choices (for approach_choice or suggestion types)."
```
options 可选，传不传由 lead LLM 决定。`prompt.py:333` 只在「是否出图」样板硬编码了 options 示例，其他反问点（尤其范式确认）无强制。**没有任何 guardrail 检查发出的 ask_clarification 带没带 options。**

### 1.3 现有 gate 不覆盖 options

[`IntentPostStepAskGateProvider.evaluate`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/intent_post_step_ask_gate_provider.py#L129) **只拦 `task()` 调用**（L131 `if request.tool_name != "task": allow`），检查「派下游前是否跳过 ask 步骤」（那是 ETHO-7 范畴）。它**不拦 `ask_clarification` 本身**，所以 options 缺失无人管。

### 1.4 为什么是确定性数据修复（非 prompt）

options 该不该带，是**每个反问点的固有属性**（范式确认/是否出图/是否出报告这种结构化选择题就该带；纯开放澄清可不带）。这是**可在 PATHS 声明的确定性事实**，不该靠 lead 每次 prompt 自由裁量（HarnessX：用类型化结构，不用 prompt 规则——prompt 规则会变 Telecom 累加 reminder）。

---

## 二、设计

### 2.1 修法：Step 加 requires_options 字段 + 新 guardrail 拦 ask_clarification

**步骤 1 —— 扩 `Step` dataclass**（[`path_registry.py:18-36`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/path_registry.py#L18)）：
```python
@dataclass(frozen=True)
class Step:
    kind: StepKind
    target: str
    condition: str | None = None
    prerequisites: tuple[str, ...] = ()
    requires_options: bool = False   # 新增：ask step 声明「此反问必须带快捷选项」
```
向后兼容（默认 False，dispatch step 不受影响）。

**步骤 2 —— PATHS 给结构化反问标 requires_options=True**：
```python
"E2E_FULL_ASKVIZ": [
    ...,
    Step("ask", "viz", requires_options=True),      # 是否出图 = 是/否选择题
    ...,
    Step("ask", "report", requires_options=True),   # 是否出报告 = 是/否选择题
],
"E2E_MIN": [..., Step("ask", "four_choice", requires_options=True)],
"CLARIFY": [Step("ask", "clarify")],   # 开放澄清，requires_options 保持 False
```
（范式确认反问目前不在 PATHS 的 ask step 里——它发生在 intent 分类之前。范式反问的 options 强制归 ETHO-7 spec 处理 intent 前的反问，或在 prompt 侧另议；本 spec 只覆盖 PATHS 已声明的 ask step。**诚实标注这个边界**。）

**步骤 3 —— 新增轻量 guardrail provider 拦 ask_clarification**：`AskClarificationOptionsProvider`（或扩现有 provider 加一条 ask_clarification 分支）：
```python
def evaluate(self, request):
    if request.tool_name != "ask_clarification":
        return GuardrailDecision(allow=True)
    intent = _extract_latest_intent(_lead_messages.get())
    steps = PATHS.get(intent)
    if not steps:
        return GuardrailDecision(allow=True)
    # 找当前正在问的 ask step（按已完成 gate 推断下一个待问 ask）
    pending_ask = _next_pending_ask_step(steps, workspace)   # 复用 evaluate 的 gate_completed 推断
    if pending_ask is None or not pending_ask.requires_options:
        return GuardrailDecision(allow=True)
    options = request.tool_input.get("options")
    if not options:
        return GuardrailDecision(allow=False, reasons=[GuardrailReason(
            code=f"ethoinsight.ask_{pending_ask.target}_missing_options",
            message=(f"按 {intent} 路径，ask({pending_ask.target}?) 是结构化选择题，"
                     f"请在 ask_clarification 的 options 参数里给出快捷选项（如 ['A. 是…', 'B. 否…']）后重发。"),
        )], policy_id="ask_clarification_options")
    return GuardrailDecision(allow=True)
```

### 2.2 不改的东西

- **不改** `ask_clarification_tool` 的签名（options 仍可选——CLARIFY 类开放澄清不需要 options）。
- **不改** `IntentPostStepAskGateProvider` 的 task() 拦截逻辑（ETHO-7 范畴）。
- **不加** 任何「请记得带 options」的 prompt 规则（Telecom 禁令——用结构 gate 不用 prompt）。
- **不碰** PR#175 改的 deny message 文案。

---

## 三、改动清单（change manifest）

### 3.1 `path_registry.py` —— Step 加 requires_options + PATHS 标注
- **编辑**：`Step` 加 `requires_options: bool = False`；PATHS 给 viz/report/four_choice 标 True。
- **预期改善**：结构化反问点的 options 要求在 SSOT 声明，可被 gate 强制。
- **可能回归**：CI 哨兵 `test_path_registry_ssot.py` 可能断言 Step 字段集 → 需同步更新哨兵。
- **病理风险**：catastrophic forgetting —— 改 Step dataclass 影响三个消费者（prompt 图/派遣 provider/ask gate）→ 跑全部消费者测试守 seesaw。

### 3.2 新 `AskClarificationOptionsProvider`（或扩现有 provider）
- **编辑**：拦 ask_clarification，按 pending ask step 的 requires_options 强制 options。
- **预期改善**：结构化反问漏 options 时 deny，lead 补 options 重发 → 体验一致。
- **可能回归**：误拦 CLARIFY 开放澄清（requires_options=False 不拦，需测试覆盖）。
- **病理风险**：reward hacking —— lead 会不会传空 options 糊弄？→ 检查 `if not options` 含空列表，deny 文案明确要求实质选项。

### 3.3 注册 provider
- 在 guardrail provider 注册处加入新 provider（grep 现有 provider 注册方式，照搬）。

---

## 四、测试（TDD 红→绿 + smoke）

> ⚠️ guardrail 属 harness 核心，改后裸导入 `app.gateway` + `make_lead_agent`（import 环铁律）。prompt/PATHS 契约测试用 importlib 读被测源（防主仓假绿）。

测试文件：`tests/test_ask_clarification_options_gate.py`（新增）。

1. **test_deny_when_required_options_missing**（核心，红→绿）：E2E_FULL_ASKVIZ 路径、viz ask step（requires_options=True），ask_clarification 不带 options → deny。改前红（无此 provider）、改后绿。
2. **test_allow_when_options_present**：同上但带 options(≥2) → allow。
3. **test_allow_open_clarify_without_options**（守边界）：CLARIFY 路径 clarify step（requires_options=False）不带 options → allow（开放澄清不强制）。
4. **test_empty_options_denied**（reward hacking 防护）：options=[] 空列表 → deny（不能用空糊弄）。
5. **test_step_requires_options_backward_compat**：现有 dispatch step / 未标注 ask step → requires_options 默认 False，行为不变。
6. **test_path_registry_ssot_sentinel_updated**：CI 哨兵认得新字段、PATHS 标注一致。
7. **smoke test_provider_instantiates_and_evaluates**：provider 实例化 + 喂合成 request 跑通不抛。

---

## 五、验收标准（确定性 gate 序列，第一个失败即停）

1. **manifest 完整**：§三每处改动有预期改善 + 可能回归 + 测试。
2. **smoke**：测试 7 绿。
3. **红→绿**：测试 1 改前红改后绿；2/3/4/5/6 绿。
4. **回归（seesaw）**：`test_path_registry_ssot` + `IntentPostStepAskGate` 邻域 + clarification 邻域全绿；backend 全量按 `feedback_pr_merge_must_run_full_suite_on_shared_logic` 跑 + grep `Step(` / `PATHS` 消费者。
5. **import 环**：裸导入两入口 0 退出。
6. **scope**：未改 ask_clarification 签名；未加 options 提醒 prompt；未碰 PR#175 文案。

---

## 六、风险与注意事项（三大病理自检）

1. **reward hacking**：lead 传空/占位 options 糊弄？→ `if not options` 拦空列表；deny 文案要求实质选项。测试 4 守。
2. **catastrophic forgetting**：Step 加字段影响三消费者（prompt 图/派遣 provider/ask gate）→ CI 哨兵 + 全消费者测试守 seesaw（测试 6）。PR#175 类 drift（改一处漏同步）的教训：改 Step 必查所有读 PATHS 的地方。
3. **under-exploration**：本 spec **敢动结构**（Step 加字段 + 新 gate），不是加 prompt「记得带 options」规则——正面避开 Telecom 累加 reminder 陷阱。
4. **边界诚实**：范式确认反问（intent 分类前）不在 PATHS ask step 内，本 spec 不覆盖它的 options 强制 → 归 ETHO-7 或 prompt 侧另议。**spec 明标此边界，不假装全覆盖。**
5. **受保护文件**：path_registry / guardrail provider sync 时 surgical 守护。
6. **provider 注册**：新 provider 要在 guardrail 注册链加入，否则不生效——验收必含「provider 实际被 GuardrailMiddleware 加载」的集成确认。

---

## milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-9（options 确定性）与 ETHO-8（A/B 顺序确定性）同属「交互非确定性」的确定性子修复系列；ETHO-7（决策点漂移）是更深的 intent 分类问题，另 spec。
