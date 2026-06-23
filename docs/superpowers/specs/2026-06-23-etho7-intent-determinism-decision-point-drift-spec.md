# Spec：ETHO-7 决策点数量/形态运行间不一致 —— intent 分类稳定化 + PATHS 强制 ask 编排，DeerFlow 原生

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：🔴 高 · 病根 · harness 控制（D7）。同输入 Run1 出 4 个决策点、Run2 只 2 个且跳过「是否出图」。根因：决策点数量/合并 **100% 由 lead LLM 自由裁量**——`[intent]` 分类每轮可能不同（E2E_FULL vs E2E_FULL_ASKVIZ），导致走不同 PATHS、反问点漂移。
> **是 ETHO-9/ETHO-5 的共同病根**（交互决策无确定性约束），但本 spec 聚焦「决策点数量/顺序确定性」这一层；options 强制（ETHO-9）、分组全量（ETHO-5）已各自独立成 spec。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-7；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - **生产实证**（2026-06-23 dump，memory `reference_harnessx_report_and_etho_spec_application`）：4 thread 反问决策点数量/形态各不同——3a41e483 范式合进一次+出图#29+出报告#43；e6ea7946 范式单独#5；47e8155a 无范式反问+出图#45+出报告#56。identify status 四态（unknown/ambiguous/ok/error）lead 据此走不同分支 → 反问点漂移。
> - SSOT：`guardrails/path_registry.py` 的 `PATHS`（已声明 8 intent 的有序 step 序列，含 ask 点）。
> - 既有结构：`IntentClassificationGuardrailProvider`（强制 lead 声明 `[intent]` 行）+ `IntentPostStepAskGateProvider`（读 PATHS 拦 task() 跳 ask 步骤）。
> - **框架**：DeerFlow（LangGraph）原生——`GuardrailProvider` + `path_registry` + LangChain middleware，**不引入 HarnessX 机制**（memory `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`）。
> - **HarnessX 思想**：§7.1「类型系统不生成正确程序但让错误程序可检测」+ §6.6 Telecom 累加 reminder 致崩禁令。
> - 受保护文件：path_registry / 两个 intent provider / lead prompt 均 deerflow 定制面，sync surgical。
> - ⚠️ **PR#174 已把「是否出图」决策点改成 lead 反问**（`58588dc6`）；本 spec 不与之重复——PR#174 是「画图默认全画、chart_budget 仅 lead 透传」，本 spec 是「决策点该不该出现/顺序稳定」，正交。先读 PR#174 确认 viz 决策点现状。

---

## 〇、框架契合声明 + 病根定性

**这是病根 spec，但不是"重写交互系统"**。DeerFlow 已有完整的决策点 SSOT 骨架（`PATHS` + 两个 intent provider）；ETHO-7 的真因是**这套骨架的"入口"（intent 分类）本身不稳定**，导致下游 PATHS 选错、决策点漂移。修法是**稳定入口 + 让 gate 对 ask 编排也强制**，全在现有 `GuardrailProvider` + `PATHS` 之上，零新抽象。

HarnessX §7.1 类型系统类比是支点：**PATHS 是"类型"，它不生成正确交互，但让"偏离声明路径的交互"可被 gate 检测**。问题不在 PATHS 本身（它已声明对），在 intent 分类把任务映射到了错的 PATHS。

---

## 一、根因（逐字节 + 生产实证）

### 1.1 现象与生产实证

同一份 EPM 数据：Run1 出 4 个决策点（范式确认 + 列对齐 + 是否出图 + 是否出报告），Run2 只 2 个且跳过「是否出图」。

prod 4 thread（决策点形态各异，坐实"运行间不一致"）：
| thread | 范式反问 | 出图反问 | 出报告反问 | intent 推断 |
|---|---|---|---|---|
| 3a41e483 | #8/#9 合进一次 | #29 | #43 | E2E_FULL_ASKVIZ |
| e6ea7946 | #5 单独 | （无独立） | — | 范式 unknown→先澄清 |
| 47e8155a | 无 | #45 | #56 | E2E_FULL_ASKVIZ（status=error 走别路） |

### 1.2 真因：intent 分类不稳定 → 选错 PATHS → 决策点漂移

- `[intent]` 由 lead LLM 在 AIMessage 输出（`IntentClassificationGuardrailProvider` 只强制"必须声明"，**不强制"分类正确/稳定"**）。
- E2E_FULL（不问出图，直接全跑）vs E2E_FULL_ASKVIZ（问是否出图）的选择，凭 lead 对"用户模糊程度"的理解，每轮不同（`prompt.py:307` 一带的 intent 分类描述）。
- identify status 四态（unknown/ambiguous/ok/error）让 lead 走不同前置分支（范式先澄清 vs 直接进 PATHS），叠加 intent 选择，决策点数量/顺序随机。

### 1.3 现有 gate 只拦"跳步"，不拦"选错路径/合并漂移"

`IntentPostStepAskGateProvider` 拦的是"在声明的 PATHS 内跳过某 ask 步骤"（ETHO-9 同 provider）。但如果 lead **声明了错的 intent**（E2E_FULL 而非 E2E_FULL_ASKVIZ），gate 按错 PATHS 评估，"是否出图"根本不在路径里 → 不算跳步 → 不拦。**漂移发生在 intent 选择层，现有 gate 在 intent 之下，管不到。**

### 1.4 为什么是结构问题（HarnessX under-exploration 警告）

历史倾向是"加 prompt 规则讲清什么时候用 E2E_FULL vs ASKVIZ"——但这是 prompt 邻域打转（under-exploration），且累加规则会 Telecom 式互相冲突。真修在**让 intent 分类更确定**（减少自由裁量）+ **让 PATHS 对 ask 编排也强制**（已有 task 拦截，扩到 ask 顺序）。

---

## 二、设计

> ⚠️ 这是病根 spec，改动面比其他大。**分两个可独立交付的子改动**，按风险递增：

### 2.1 子改动 A（低风险，先做）：intent 分类确定化——E2E_FULL vs ASKVIZ 的判据从 LLM 自由裁量改为可声明规则

现状 E2E_FULL（不问出图）vs E2E_FULL_ASKVIZ（问出图）由 lead 凭"模糊程度"选。**收窄为确定性默认**：

- **默认 E2E_FULL_ASKVIZ**（问是否出图）——除非用户**明确说了**出不出图（"给我图"/"不要图"）。即"不确定就问"，而非"不确定 lead 自己定"。
- 落地：`IntentClassificationGuardrailProvider` 或一个新的轻量 check —— 当 lead 声明 `[intent] E2E_FULL`（跳过出图反问）但**对话里没有用户明确的出图意向**时，deny + 提示"用户未明确出图意向，应声明 E2E_FULL_ASKVIZ 先问"。
- 这把"该不该出现出图决策点"从 LLM 裁量变成**可检测规则**（HarnessX：类型化结构让错误可检测）。

### 2.2 子改动 B（中风险）：PATHS ask 步骤顺序强制——扩 IntentPostStepAskGate 检测 ask 乱序/合并

现状 gate 只在"派 dispatch 前查 ask 是否完成"。扩展：**ask 步骤之间的相对顺序**也按 PATHS 强制——lead 不能把 PATHS 里分开的两个 ask 点（如 viz 在 report 前）合并成一个、或乱序。

- 落地：在 `evaluate` 里，当检测到某 ask gate 的"前置 ask"未完成就先完成了"后置 ask"，deny。
- 复用现有 `gate_completed` + ASK_GATE_MAP，不新增机制。

### 2.3 不做的（守边界，避免 over-engineering）

- **不做** intent 分类的"完全确定性化"（把 LLM 理解全换成规则）——那会过度约束、且很多任务的 intent 确实需要 LLM 语义判断。只确定化**最易漂移的一处**（E2E_FULL vs ASKVIZ）。
- **不碰** identify status 四态逻辑（unknown/ambiguous/ok/error 是范式识别的正确产出，归 ETHO-4 设计裁决 + 范式反问铁律）。范式反问的存在本身是对的（HITL 铁律 L476），漂移的是"范式确认之后"的决策点。
- **不加** prompt 规则讲"什么时候合并反问"（Telecom 禁令）。
- **不与 PR#174 重复**：PR#174 改 viz 决策点的 chart_budget 透传；本 spec 改"viz 决策点该不该出现"，正交。

---

## 三、改动清单（change manifest）

### 3.1 intent 分类确定化（子改动 A）
- **文件**：`guardrails/intent_classification_provider.py`（或新 check）+ `agents/lead_agent/prompt.py:307` intent 描述微调。
- **编辑**：声明 E2E_FULL（跳出图反问）需对话有用户明确出图意向，否则 deny 要求改 ASKVIZ。
- **预期改善**：「是否出图」决策点不再随机消失（Run2 跳过它的现象消除）。
- **可能回归**：用户明确说"全做包括图"时不该被强制问 → deny 条件要识别"明确出图意向"，测试覆盖。
- **病理**：under-exploration——这是结构改动（gate 规则）非 prompt 累加。

### 3.2 PATHS ask 顺序强制（子改动 B）
- **文件**：`guardrails/intent_post_step_ask_gate_provider.py`。
- **编辑**：扩 evaluate 检测 ask 步骤乱序/合并。
- **预期改善**：决策点顺序稳定（viz 恒在 report 前）。
- **可能回归**：合法的"用户一次性回答多个问题"场景不该误拦 → 测试覆盖批量应答。

### 3.3 不改
- identify status 逻辑、范式反问、PR#174 的 chart_budget。

---

## 四、测试（TDD 红→绿 + smoke）

> ⚠️ guardrail 核心，裸导入两入口；prompt/PATHS 契约用 importlib。

测试文件：`tests/test_intent_determinism_etho7.py`（新增）。

1. **test_deny_e2e_full_without_explicit_viz_intent**（子改动 A 核心，红→绿）：lead 声明 E2E_FULL 但对话无明确出图意向 → deny 要求 ASKVIZ。
2. **test_allow_e2e_full_with_explicit_viz_intent**：用户明确"也要图" → E2E_FULL 放行。
3. **test_allow_e2e_full_askviz_default**：默认 ASKVIZ → 放行。
4. **test_ask_order_enforced**（子改动 B，红→绿）：PATHS 里 viz 在 report 前，lead 先完成 report 跳过 viz → deny。
5. **test_batch_answer_not_false_denied**（守边界）：用户一次回答多问 → 不误拦。
6. **smoke**：两 provider 实例化 + 喂合成 request 跑通不抛。
7. **test_identify_status_logic_unchanged**（守 scope）：范式 status 四态逻辑不变。

---

## 五、验收标准（确定性 gate 序列）

1. manifest 完整（§三）。
2. smoke（测试 6）绿。
3. 红→绿：测试 1/4 改前红改后绿；2/3/5/7 绿。
4. 回归（seesaw）：intent provider + PATHS SSOT + ask gate 邻域全绿；backend 全量 + grep intent/PATHS 消费者。
5. import 环：裸导入两入口 0 退出。
6. scope：未碰 identify status、未碰 PR#174 chart_budget、未加反问合并 prompt 规则。
7. **子改动可独立合**：A、B 各自红→绿可单独 PR（病根 spec 分步降风险）。

---

## 六、风险与注意事项（三大病理自检）

1. **reward hacking**：deny 后 lead 会不会假声明"用户要图"骗过 A？→ deny 检查对话**实际**有无出图意向文本，不信 lead 自述。
2. **catastrophic forgetting**：改 intent 分类影响所有走 PATHS 的路径 → 全 intent 路径测试守 seesaw；PR#175 类 drift 教训（改 intent 描述要同步所有消费者）。
3. **under-exploration**：本 spec **敢动 intent 分类结构 + ask 顺序 gate**，非加 prompt 规则——避开 Telecom。
4. **边界诚实**：这是病根但**不能一次根治所有交互非确定性**（HarnessX §7.3：符号空间无收敛保证）。只确定化最易漂移的 E2E_FULL/ASKVIZ + ask 顺序；intent 的语义判断部分仍靠 LLM。**spec 明标：这是降漂移非消除漂移。**
5. **与 ETHO-9 同 provider**：子改动 B 和 ETHO-9 都改 `IntentPostStepAskGateProvider` → 两 spec 若并行实施需协调（建议 ETHO-9 先合，ETHO-7 子改动 B 基于其上）。
6. **受保护文件 sync surgical**。

---

## milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-7（决策点确定性病根）是 ETHO-8（A/B 序）、ETHO-9（options）确定性子修复的**共同病根层**；checkpoint = 「交互决策点从 lead 自由裁量降到 PATHS/gate 数据驱动约束」。标注为病根，后续可能需多轮迭代（HarnessX：病根 spec 往往非一次到位）。
