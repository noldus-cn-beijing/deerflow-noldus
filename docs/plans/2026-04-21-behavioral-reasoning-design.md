# 行为学判断能力设计 — v0.1 可执行蓝图

> **创建日期**: 2026-04-21
> **目标里程碑**: 2026 年 9 月 v0.1
> **对应 roadmap**: Phase 0 M0.2/M0.3（范式补全）+ Phase 2 M2.3（异常诊断扩展）
> **作者**: Claude Opus 4.7 (1M context) + Qiuyang（Noldus）
> **状态**: 设计阶段（brainstorming → 设计 → 执行计划）

---

## 1. 背景和目标

### 1.1 为什么要写这份文档

EthoInsight 在 2026-04 完成了 shoaling 范式的端到端打通和可见性优化（参见 [docs/handoffs/2026-04-21-subtask-visibility-done.md](../handoffs/2026-04-21-subtask-visibility-done.md)）。流水线层面的问题（可见性、语言一致性、handoff 契约）基本闭环。

下一阶段的挑战是**内容层**——agent 在行为学领域的**判断能力**够不够深、够不够稳、够不够可信。这件事 roadmap 里分散在多处（Phase 0 M0.2/M0.3 的范式补全、Phase 2 M2.3 的异常诊断扩展），但没有一份总文档说清：

- 判断能力的**边界**在哪里（能做什么、不能做什么）
- 判断的**可靠性**怎么保证（不瞎编、可追溯）
- 新增范式时应该**交付什么标准件**
- 什么时候该靠**代码**、什么时候该靠**模型**、什么时候该靠**微调**

这份文档回答这些问题。

### 1.2 范围与时间窗

- **时间**: 2026-04-21 → 2026-09（v0.1 交付）
- **范围**: v0.1 的判断能力 — 5 个范式的标准分析 + 洞察审核机制 + 异常诊断模式库
- **不在范围**: 跨范式推理、实验设计指导、文献综述生成（这些是 Phase 3/4 的事）

### 1.3 目标（可验证）

v0.1 交付时，以下陈述应同时成立：

1. 5 个范式（shoaling / epm / open_field / forced_swim / morris_water_maze）能端到端分析，输出质量不低于 shoaling 当前水平
2. data-analyst 的每条洞察结论都能追溯到**数据依据**（handoff JSON 里的具体指标值）或**文献依据**（quality-reviewer 的引用）
3. 至少 5 个行为学专家标注的 golden-case 数据跑通 E2E 回归，agent 的核心结论与专家标注匹配
4. 用户能通过 SubtaskCard 看到 agent 的完整思考过程（含互审），在关键分叉点有明确选项

---

## 2. 优化函数转向："accuracy + 可见性 > 速度"

### 2.1 与消费级 Agent 的根本差异

EthoInsight 的用户是行为学研究员，不是消费级用户。这导致我们的优化函数和主流 AI 产品不同：

| 维度 | 消费级（ChatGPT / Copilot） | EthoInsight |
|---|---|---|
| 主要优化 | 响应速度 + 流畅度 | 结果准确度 + 推理可见度 |
| 用户耐心 | 几秒就嫌慢 | 愿意等 1-3 分钟换可靠结果 |
| 错误成本 | 重试一下即可 | 错误结论会影响论文、实验设计 |
| 思考过程价值 | 多数用户不关心 | **本身就是价值**（研究员会从中学到领域知识） |

这不是"我们觉得准确度重要"——而是**用户真实使用场景下的价值排序**：一篇行为学 paper 从实验到发表 6-12 个月，agent 多花 1 分钟让结论更可信是 100% 值得的交易。

### 2.2 这对架构的影响

三个直接推论：

**推论 1：可以承担多 subagent 互审的开销**。增加 quality-reviewer 导致的额外 30-60 秒延迟和 2× token 消耗，在这个优化函数下是正确的交易。

**推论 2：思考过程必须全量暴露**。fix4→fix5 扩展 SubtaskCard 全量 CoT 的方向是对的，下一步是让 subagent 之间的**对话**也对用户可见——"data-analyst 说 X，quality-reviewer 基于文献反驳 Y，data-analyst 采纳后修正为 Z"这种过程本身就是信任建立的机制。

**推论 3：agent 可以主动打断流程问用户**。在消费级产品里"少问问题"是美德，但在研究场景下，研究员宁愿被问清楚也不愿拿到错误前提下的自信回答。ask_clarification 不是兜底工具，而是**一等公民**。

### 2.3 作为设计原则

下面所有章节的设计决策都以这三条推论为基准。当出现"准确度 vs 速度""显式询问 vs 自动决策"的权衡时，本文档的默认选择始终是前者。这不代表完全不管速度——而是当两者冲突时，优先保准确度。

---

## 3. 能力边界：模型 vs 工具 / 微调 vs base

### 3.1 为什么这章很重要

一个常见误区是"把所有能力都堆到 prompt 里，让模型想办法"。这在 v0 demo 阶段可以，在 v0.1 可靠性要求下会崩。同样重要但反向的误区是"把所有逻辑都写成代码，让模型只做格式化"——这会丢掉 LLM 真正的价值（推理、适配、自然语言解释）。

能力设计的核心问题不是"模型能做吗"，而是**"这个能力放在哪层实现，系统作为整体最稳"**。放错层的代价非常高：数值计算放 LLM → 幻觉；推理判断写死代码 → 僵化且不可扩展。

### 3.2 四层能力分层

EthoInsight 的能力分布在四层，从稳定性高到低、从灵活性低到高排序：

| 层 | 实现形式 | 何时用 | 典型例子 |
|---|---|---|---|
| **L1 代码/库** | Python 函数、硬编码常量 | 数学定义、物理不变量 | `compute_metrics()` 计算 NND、`mean + 2*SD` 阈值公式 |
| **L2 配置/YAML** | 外部 YAML、数据库 | 领域知识、可调参数 | `assess.py` 读取的阈值 YAML、各范式的正常范围 |
| **L3 工具调用 + RAG** | MCP 工具、noldus-kb 检索 | 事实来源、文献引用 | quality-reviewer 查 noldus-kb、knowledge-assistant 答问 |
| **L4 模型权重（base 或微调）** | Prompt 或微调后的神经网络 | 推理、解读、路由、自然语言生成 | lead agent 路由、data-analyst 撰写洞察、异常模式识别 |

**设计原则**：
- 能下沉到 L1/L2 的**绝不**放 L4（数值不能靠模型记住）
- L4 的输出必须能被 L1/L2/L3 **可追溯验证**（每条结论都能找到依据）
- L3（RAG）是 quality-reviewer 的核心价值——它让模型能引用**外部事实**，而不是依赖自己可能模糊的权重记忆

### 3.3 微调的定位：L4 内部的优化

微调不改变分层结构——它只让 **L4 本身更稳**。这是关键认知：**微调不能替代 L1 的代码、不能替代 L3 的 RAG，它只能让 L4 里原本靠 prompt 的事情变得更一致、更准确、更省 token**。

v0.1 微调（Phase 1，Qwen3-8B）该解决的和不该解决的：

**微调能显著提升的**：

1. **路由决策的直觉** — 当前 lead prompt 要写很长描述才能让 base model 在"raw data vs 已分析数据"场景下做对选择；微调后可以把这些场景**内化到权重**，prompt 更短、决策更一致
2. **行为学术语的准确性** — base model 容易混用 "shoaling index / shoal cohesion / group cohesion"，微调后术语使用更严谨
3. **数值解读的 calibration** — 对"mean_nnd = 37.23 mm 对斑马鱼正常、同值对小鼠开放场偏高"这种分布感，微调后比 prompt 硬教更稳
4. **ask_clarification 调用直觉** — base model 常常过度自信直接往下走；微调可以通过训练数据反复塑造"遇到 X 类输入先问用户"的习惯
5. **JSON 格式输出稳定性** — C6/1b605d35 已经遇到过转义问题，微调后可以显著减少这类格式翻车

**微调做不了 / 不该做的**：

1. **精确数值** — `mean_nnd = 70.02223` 必须由 `metrics.py` 算，永远不靠模型记忆
2. **事实源** — quality-reviewer 要用**文献**反驳 data-analyst，文献必须来自 noldus-kb 的 RAG 检索，不是微调权重。模型权重里的"记忆"是模糊的、会幻觉，不能当事实来源用
3. **硬规则** — assess.py 里"NND > mean + 2*SD → 离群"是数学定义，不是判断能力
4. **审核的独立性** — 如果 data-analyst 和 quality-reviewer 是同一个微调模型，它们的盲点也一样。quality-reviewer 的"独立性"靠的是**访问不同工具（noldus-kb）**，不是不同 prompt 或不同微调版本

### 3.4 架构为微调"留路"

这是一个反向的设计约束：**今天的架构要考虑明天微调时的数据采集便利性**。具体体现：

1. **Lead prompt 里的"场景→动作"建议要写得像训练数据** — 结构化、可枚举、有标注。这样 Phase 1 的 `scripts/generate_synthetic_data.py` 可以直接把场景描述拆成 QA 对
2. **互审过程要结构化保存，不只是给用户看** — "data-analyst 初版洞察 → quality-reviewer 指出问题 → data-analyst 修正版"这种对话如果结构化落盘，就是天然的 DPO 训练数据（修正版 > 初版）。互审日志应该同时满足"用户可读"和"机器可采集"两个需求
3. **阈值（L2）和解读风格（L4）必须分离** — 阈值写代码/YAML，永不进微调数据；解读的措辞、结构、论证风格可以进。否则微调时会把硬阈值学成模糊概率，灾难

> **具体微调执行计划**：见 [2026-04-21-finetuning-strategy-update.md](2026-04-21-finetuning-strategy-update.md)——包含蒸馏策略、SFT/DPO 时机、Fireworks 实操、行为学同事协作分步。

### 3.5 本章总结

能力设计的判定流程：

```
新能力需求
    ↓
是否有数学定义或物理不变量？
  是 → L1（代码/库）
  否 ↓
是否是领域知识且需要可调？
  是 → L2（YAML/配置）
  否 ↓
是否需要引用外部事实？
  是 → L3（RAG + 工具）
  否 ↓
属于推理/解读/路由/生成？
  → L4（模型）
    ├─ v0.1 阶段靠 prompt 实现
    └─ 设计时留好微调路，数据结构化落盘
```

后面所有章节的每个能力设计，都会明确标注属于哪一层、是否有微调路径、微调前后有什么差异。

---

## 4. 判断能力的两层模型

### 4.1 两种"判断"的分离

当我们说"agent 的判断能力"时，实际上在说两件完全不同的事，混为一谈会导致设计错位：

**Layer A — 数值输出**：
- 例子：`mean_nnd = 37.23`、`p-value = 0.023`、`Shapiro-Wilk W = 0.94`
- 本质：对输入数据的确定性计算
- 可靠性来源：代码正确性 + 测试覆盖
- 所在层：**L1（代码/库）**
- 当前状态：ethoinsight 库 131 tests passed，基本已解决

**Layer B — 洞察判断**：
- 例子：
  - "Subject 3 的 NND 异常，可能是群体外离散个体"
  - "低运动量 + 中心区回避提示抑郁样表型"
  - "排除 Subject 3 后组均值回归正常，原'显著差异'实际是个体变异"
- 本质：对数值结果的**领域解读**和**因果/相关推断**
- 可靠性来源：数据依据（Layer A 的值）+ 领域知识（L2/L3）+ 推理质量（L4）
- 所在层：**L4（模型），但必须引用 L1-L3**
- 当前状态：C6 打通了 outlier_findings 的结构化路径，shoaling 上证明可行；但其他范式未覆盖、审核机制未建、失控风险仍在

### 4.2 为什么必须分离

混淆两者会导致两类典型错误：

**错误一：把 Layer A 问题当 Layer B 处理**。让 LLM"判断" NND 是不是 37.23——这是计算，不该问模型。过去 demo 阶段确实有过这种尝试，结果是模型会"估算"一个接近的值然后自信输出，幻觉典型案例。

**错误二：把 Layer B 问题当 Layer A 处理**。试图把"什么是异常个体"写成一个硬规则（比如 `NND > mean + 2*SD`），但真实行为学里异常判断依赖多个指标的组合、依赖范式、依赖群体规模、依赖物种。写死规则要么覆盖不全，要么过度触发。正确做法是：**提供候选异常检测算法作为工具（L1），让 L4 模型根据上下文决定用哪种、怎么解读**。

### 4.3 Layer A 的可靠性保障（已基本就绪）

所需工作量不大，主要是**补齐范式模板**：

- 每个范式模板覆盖的指标必须有对应 `metrics.py` 函数 + 单元测试
- 每个 `assess.py` 阈值必须有 YAML 配置 + 单元测试（阈值是数学值，但值本身来自领域，所以是 L1 函数读 L2 配置）
- 统计决策树（Shapiro-Wilk → 参数/非参数检验）已在 `statistics.py` 实现，各范式复用

标准：Layer A 的每个输出必须能在代码里追溯到**精确的计算路径**，不允许任何值只存在于模型输出里。

### 4.4 Layer B 的可靠性保障（v0.1 重点）

Layer B 是 v0.1 阶段的主要工作。保障分三层：

**保障 1 — 数据依据追溯**：data-analyst 的每条洞察必须引用 Layer A 的具体值或 `handoff_code_executor.json` 里的具体字段。当前 C6 已通过 outlier_findings schema 强制这一点，需要扩展到所有洞察类型（不只是 outlier）。

**保障 2 — 文献依据追溯**：涉及领域判断的结论（"这个表型对应抑郁样"、"这个距离是焦虑典型值"）必须有文献引用，由 quality-reviewer 经 noldus-kb 检索提供。这是第 6 章要设计的核心机制。

**保障 3 — 独立审核**：quality-reviewer 对 data-analyst 的初版洞察做文献交叉核对，发现无支撑或反事实的结论时要求修正。这是 v0.1 新增的机制，详见第 6 章。

### 4.5 两层的微调策略

| 层 | 微调 v0.1 阶段 | 微调 v0.1+ 阶段 |
|---|---|---|
| Layer A | **永不微调**，保持代码 + 测试 | 不变 |
| Layer B（洞察生成） | base model + prompt，C6 格式契约已落 | 微调 Qwen3-8B 稳定格式、提升术语准确度、强化"必引用数据/文献"习惯 |
| Layer B（审核判断） | base model + prompt + noldus-kb RAG | 微调强化"批判视角"（质疑无支撑结论、找替代解释），但 RAG 仍是事实源 |

一个关键推论：**Layer B 的洞察生成和审核判断，在微调阶段应该是两个独立的微调 checkpoint**（或通过不同 system prompt 激活的同模型两种模式）。否则两者盲点一样，互审机制失效。这个约束在第 6 章 quality-reviewer 设计里会进一步展开。

### 4.6 这一章决定了什么

- **Layer A 的重点是工程完整度**：每新增一个范式就按模板补齐 metrics/assess/tests，不是设计问题而是执行问题（第 7 章给模板）
- **Layer B 的重点是机制设计**：引用约束 + 独立审核是 v0.1 的核心创新点（第 5 章给 lead agent 扩展，第 6 章给 reviewer 设计）
- **两层都要为微调留路**：Layer A 的数据 + Layer B 的互审对话是未来 SFT/DPO 的天然数据源（第 3.4 节的架构反向约束在这里具体化）

---

## 5. 扩展 lead agent 的能力菜单（不规定路由）

### 5.1 设计原则：给素材，不给规则

一个容易陷入的误区是把 agent 架构设计成一个带分支的 switch：
- "如果输入是 .txt → 走 code-executor"
- "如果用户说'快速' → 跳过 reviewer"

这种写法把判断逻辑从 agent 剥离到代码里，剥夺了 agent 的价值。**agent 的价值在于能处理我们没预见的场景组合**——用户上传 raw data 但只想快速浏览不细分析、用户同时上传 raw 和已分析结果要求交叉验证、用户中途改主意想加 fact-check……这些场景不可能穷举成规则。

正确做法是**扩展 lead agent 的能力菜单**：
- 给它新工具（quality-reviewer subagent、data-analyst 审核模式）
- 给它判断素材（在 prompt 里描述典型场景和推荐动作，但明确说"你自己判断，这只是参考"）
- 给它主动询问的能力（ask_clarification 升级为一等公民）

最终的路径选择是 lead 自己的 reasoning，不是代码里的 if/else。

### 5.2 新增能力菜单（v0.1）

在当前 lead agent 的工具集基础上，v0.1 新增/强化三样：

| 能力 | 类型 | 状态 |
|---|---|---|
| quality-reviewer subagent | 新增 subagent | 第 6 章详细设计 |
| data-analyst 审核模式 | 现有 subagent 加能力 | 同一 subagent 通过不同 prompt 参数切换"生成模式 / 审核模式" |
| ask_clarification 主动使用 | 现有工具、提升地位 | prompt 层面强化使用时机 |

data-analyst "审核模式"指的是：当用户上传的是**已经分析好的结果**（表格、结论、图表）时，data-analyst 不再从 raw data 算指标，而是直接审核现有结论的合理性。这本质是 prompt 分支，不是新 subagent。

### 5.3 Lead prompt 里的"场景-动作"素材

给 lead agent 的 prompt 里增加一段**场景参考表**。这段文字的措辞必须同时满足两个要求：

1. **明示选择权归 lead**——用"通常"、"建议"、"可以考虑"等措辞，不用"必须"、"应该"
2. **结构化、可枚举**——为 Phase 1 微调数据采集做准备（第 3.4 节架构反向约束）

拟议内容（draft，真正写入 prompt 时要反复打磨）：

```
<场景参考>
以下是行为学研究员常见的请求类型和对应的推荐动作。
这些只是参考——遇到不匹配或混合场景时，你自己判断，必要时问用户。

场景 A：用户上传 EthoVision 导出的轨迹文件（.txt，含 Header/Track-Arena 标记）
  通常：这是 raw data，完整分析流程（code-executor → data-analyst → report-writer）合适
  如果用户明确说"快速看一下"：可以跳过 report-writer

场景 B：用户上传已经统计好的结果（.xlsx/.csv 含 mean/sd 列、或 .docx 含表格）
  通常：用户想要的是审核或解读，不是重新分析
  建议：询问用户意图（"审查结论准确性？还是补充解读？还是对照数据库交叉验证？"）
  然后：data-analyst 审核模式 ± quality-reviewer

场景 C：用户纯文字提问，不带数据
  通常：走 knowledge-assistant
  但：如果问题涉及之前分析过的 thread 内容，可以直接基于已有 handoff JSON 回答

场景 D：用户请求"核查/fact-check/对照文献"
  通常：在当前分支末尾追加 quality-reviewer
  注意：quality-reviewer 依赖 noldus-kb，如果不可用会退化为"只做逻辑自洽检查"

场景 E：输入形态模糊（文件名看不出、内容混合、用户表述含糊）
  推荐：用 ask_clarification 问用户
  不要：基于猜测走某条路径
</场景参考>
```

这段内容：
- **对 base model 阶段的 lead**：提供决策素材，减少瞎猜
- **对未来微调阶段**：这段本身就是训练数据的骨架，场景 A-E 各自扩展成 10-50 条 QA 就是一个 batch

### 5.4 ask_clarification 的一等公民化

当前 `ask_clarification` 工具存在但使用频率低，主要因为：
1. Lead prompt 里没有明示"什么时候该问"
2. 模型天然倾向"自己搞定"而非"麻烦用户"

v0.1 的处理是在 lead prompt 里加一段**显式触发条件**：

```
<何时使用 ask_clarification>
遇到以下情况，优先问用户而不是猜测：

1. 输入形态不明确（raw data 还是已分析结果、哪种范式）
2. 用户意图有多种合理解读（"帮我看看这个数据"可能是分析、审核、解读）
3. 分析路径有重要选项需要用户决定（要不要 fact-check、用参数还是非参检验）
4. 发现可能影响结果正确性的前提缺失（没说分组、没说对照）

问的方式：提供 2-4 个明确选项，不问开放问题。
</何时使用 ask_clarification>
```

这段同样是"素材不是规则"——lead 自己判断某次是否真的触发，但有了这段明示，使用频率会显著提高。

### 5.5 互审不是 subagent 嵌套，是 lead 编排

一个架构约束必须写明，避免后续实现时走弯路：

**DeerFlow 框架禁止 subagent 调用 task 工具**（即禁止 subagent 嵌套调 subagent）。证据：
- [config.py:25](../../packages/agent/backend/packages/harness/deerflow/subagents/config.py#L25) 默认 `disallowed_tools=["task"]`
- [general_purpose.py:47](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/general_purpose.py#L47) 注释 `# Prevent nesting`
- 所有内置 subagent（data_analyst / code_executor / report_writer / knowledge_assistant）都显式设置 `disallowed_tools` 包含 `"task"`

所以 **quality-reviewer 不能被 data-analyst 调用**。互审的完整流程必须由 lead agent 编排：

```
lead agent
  ├── 调用 data-analyst（生成模式）→ 初版洞察
  ├── [决定是否审核——参考场景 + 用户选项 + 自己判断]
  ├── 调用 quality-reviewer → 反馈
  └── 调用 data-analyst（修正模式，输入初版 + 反馈）→ 终版洞察
```

这个顺序安排**本身就是 lead agent 的 reasoning 内容**，lead prompt 里提供"互审的推荐流程"作为素材，但 lead 保留变通权（比如用户明确拒绝 fact-check 就跳过）。

### 5.6 本章能力分层归属

| 新增能力 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| quality-reviewer subagent | — | 阈值/规则配置 | noldus-kb RAG | prompt/微调后权重 |
| data-analyst 审核模式 | — | — | — | prompt 分支 |
| ask_clarification 强化 | — | — | — | lead prompt 触发素材 |
| 场景参考表 | — | — | — | lead prompt 素材 |

v0.1 微调（Phase 1）应该优先覆盖场景参考表里的 A-E 场景，以及 ask_clarification 触发时机。这些都是 lead agent 路由判断的核心能力，微调后可以大幅缩短 prompt、提升决策一致性（对应第 3.3 节"微调能显著提升的"列表）。

---

## 6. Quality-Reviewer Subagent 设计

### 6.1 职责定义

quality-reviewer 的**单一职责**：对 data-analyst 的初版洞察做批判性审核，输出结构化反馈。

它**不做**的事：
- 不生成新的洞察（那是 data-analyst 的职责）
- 不重做数据分析（那是 code-executor 的职责）
- 不直接修改 data-analyst 的输出（它的反馈由 lead agent 传回 data-analyst，由后者决定是否采纳）

这个边界要写死。如果 reviewer 越界自己生成洞察，就退化为"第二个 data-analyst"，审核机制失效。

### 6.2 输入与输出契约

**输入**：
- `handoff_data_analyst.json`（data-analyst 初版洞察，含 outlier_findings + 其他结构化字段）
- `handoff_code_executor.json`（上游数据依据，包含 per_subject、metrics、statistics）
- 范式上下文（paradigm 名称、用户意图、分组信息）

**输出**：`handoff_quality_reviewer.json`，新增 schema：

```python
class ReviewFinding(BaseModel):
    claim: str                    # 被审核的原结论（引用 data-analyst 原文）
    verdict: Literal[
        "supported",              # 有数据/文献支撑，通过
        "insufficient_evidence",  # 数据不足以支撑这么强的结论
        "contradicted",           # 有文献或数据反证
        "alternative_explanation" # 存在更合理的替代解释
    ]
    reasoning: str                # 审核理由（为什么这么判）
    evidence: list[Evidence]      # 证据列表（数据引用 + 文献引用）
    suggested_revision: str | None # 建议修改的表述（可选）

class Evidence(BaseModel):
    source: Literal["data", "literature", "logic"]
    reference: str                # 数据字段路径 / 文献 citation / 逻辑推理描述
    snippet: str                  # 关键内容节选

class QualityReviewerHandoff(BaseModel):
    reviewed_claims: list[ReviewFinding]
    overall_confidence: Literal["high", "medium", "low"]
    unreviewable: list[str]       # 无法审核的声明（数据缺失/文献无覆盖）
    metadata: ReviewerMetadata    # 用了哪些 MCP 查询、耗时等
```

这个 schema 的设计满足 Layer B 三重保障（§4.4）的前两条（数据依据 + 文献依据），且结构化到可以直接作为 DPO 训练数据使用（§3.4）。

### 6.3 审核维度（checklist）

reviewer 对 data-analyst 的每条声明按以下维度检查：

| 维度 | 审核问题 | 失败触发的 verdict |
|---|---|---|
| **数据支撑** | 声明的具体数值在 handoff_code_executor.json 里能找到吗？ | `insufficient_evidence` |
| **推断强度** | 声明的因果/相关性强度与数据匹配吗？（p=0.04 不是"显著效应"，是"边缘显著"） | `insufficient_evidence` |
| **领域共识** | 声明的行为学解读与文献共识一致吗？ | `contradicted` 或 `alternative_explanation` |
| **替代解释** | 有没有更简单或更符合 Occam 剃刀的解释被忽略？ | `alternative_explanation` |
| **样本问题** | 样本量是否足够支撑声明？异常值处理是否合理？ | `insufficient_evidence` |

这些维度写进 reviewer 的 prompt，作为**审核清单**。每个维度对应一段 prompt 指引，配合 noldus-kb 检索使用。

### 6.3a 一个具体例子

让上面的维度表落到场景里。设 data-analyst 给出初版洞察：

> "treatment 组的 mean_nnd 显著高于 control 组（37.23 vs 24.10, p=0.03），提示药物削弱了群体合群行为。"

reviewer 走 checklist：

- **数据支撑** — 查 `handoff_code_executor.json`，`statistics.mean_nnd.p_value = 0.03`，`group_means.treatment = 37.23, control = 24.10`，✅ 数据对得上
- **推断强度** — "显著"措辞在 p=0.03 可接受；但"削弱合群行为"是因果措辞，p=0.03 的**相关性**不足以支撑因果 → `insufficient_evidence`
- **领域共识** — 查 noldus-kb："zebrafish shoaling NND increase drug-induced"，检索到一篇近期文献报告"~20% zebrafish are exploratory phenotype with baseline high NND"，提示可能是个体差异而非药物效应 → `alternative_explanation`
- **替代解释** — 结合 per_subject 数据发现 Subject 3 `mean_nnd = 70.02` 显著拉高 treatment 组均值，排除后 treatment 均值 27.5，组间无显著差异 → `alternative_explanation`
- **样本问题** — control n=2, treatment n=3，总 n=5 偏低 → `insufficient_evidence`

reviewer 输出的 `ReviewFinding`：

```json
{
  "claim": "treatment 组的 mean_nnd 显著高于 control 组... 提示药物削弱了群体合群行为",
  "verdict": "alternative_explanation",
  "reasoning": "p=0.03 支撑相关性但不支撑因果；per_subject 数据显示 Subject 3 是离群个体，排除后组间差异消失；文献提示 ~20% zebrafish 基线 NND 偏高属于探索型表型变异",
  "evidence": [
    {"source": "data", "reference": "handoff_code_executor.json:per_subject.Subject 3.mean_nnd", "snippet": "70.02"},
    {"source": "literature", "reference": "Miller et al., 2024, Zebrafish Behavior", "snippet": "~20% individuals exhibit exploratory phenotype with elevated baseline NND"},
    {"source": "logic", "reference": "outlier sensitivity analysis", "snippet": "excluding Subject 3: treatment mean = 27.5, no significant difference"}
  ],
  "suggested_revision": "treatment 组 mean_nnd 高于 control（37.23 vs 24.10, p=0.03），但这一差异主要由 Subject 3 的离群高值驱动。Subject 3 可能属于文献报告的探索型表型个体。建议：(a) 增加样本量、(b) 对照 Subject 3 的探索型表型是否预先可识别、(c) 谨慎表述为'treatment 组中观察到 1 例 NND 偏高个体'而非'药物效应'"
}
```

lead agent 把这份反馈连同原洞察一起传回 data-analyst（修正模式），data-analyst 综合两者产出终版——这个终版和原版的对比，就是天然的 DPO 训练数据（§3.4）。

### 6.4 主路径 vs 降级路径

因为 noldus-kb 当前禁用（`extensions_config.json` 里 `"enabled": false`，等 `180.184.84.124:7001` 恢复），v0.1 必须设计两套运行模式：

**主路径**（noldus-kb 可用）：
- reviewer 调用 noldus-kb MCP 检索文献
- 每条审核反馈带文献引用（Evidence.source = "literature"）
- `overall_confidence` 可以基于文献覆盖度给出
- 这是设计的完整形态

**降级路径**（noldus-kb 不可用）：
- reviewer 只能做"数据自洽"+"逻辑批判"
- 只检查：数据支撑、推断强度、样本问题、内部一致性
- 不做：领域共识检查、替代解释（除非能从 skill reference 里找到）
- 自动标记 `overall_confidence = "medium"`（因为没文献背书）
- Unreviewable 列表会包含所有需要领域知识判断的声明

判定规则：reviewer 启动时检查 noldus-kb 连通性，不可用自动走降级路径，并在输出里明确标注 `metadata.mode = "degraded"`。这样下游和用户都能清楚看到"这次审核没查文献"。

### 6.5 审核范围的边界

不是所有洞察都值得审核。**过度审核会拖慢流程、稀释重要发现**。reviewer 只审核**高影响声明**：

会审核：
- 统计显著性声明（p-value、effect size 相关）
- 因果/机制推断（"提示焦虑样表型"、"药物削弱合群性"）
- 异常诊断（outlier_findings 里的每条）
- 组间比较结论
- 推广性声明（"这与文献一致"、"典型响应"）

不审核：
- 单纯的数值描述（"treatment 组均值 = 37.23"）
- 方法学陈述（"使用 Shapiro-Wilk 检验"）
- 用户原话引用
- 图表说明

这个边界由 reviewer 的 prompt 明确规定，且要带**"如果 data-analyst 的整个输出里没有高影响声明，reviewer 应该直接返回空 reviewed_claims 并标注 skip_reason"** 的兜底——避免 reviewer 为了"显得在工作"硬造反馈。

### 6.6 Subagent 注册与配置

`quality-reviewer` 作为新的内置 subagent 注册：

- 位置：[`packages/agent/backend/packages/harness/deerflow/subagents/builtins/quality_reviewer.py`](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/quality_reviewer.py)
- `disallowed_tools = ["task", "ask_clarification", "present_files"]`（不嵌套、不打断用户、不直接输出文件）
- `tools` 继承所有工具以能访问 noldus-kb MCP，但通过 disallowed 过滤掉写文件类工具（write_file / str_replace）——reviewer 只读数据 + 查文献 + 写单一 handoff JSON
- 模型配置：v0.1 阶段用与 data-analyst 相同的 base model；Phase 1 微调时独立 checkpoint（§4.5）

### 6.7 微调可分离性

按第 3.3 节的框架评估 quality-reviewer 的微调路径：

| 能力项 | v0.1 实现 | v0.1+ 微调能提升的 |
|---|---|---|
| 识别高影响声明（§6.5 范围判断） | prompt | 显著提升（一致性） |
| 按 checklist 逐维度检查 | prompt | 中等提升（格式稳定度） |
| 生成结构化 ReviewFinding JSON | schema 约束 + prompt | 显著提升（格式稳定度） |
| 文献检索与引用 | **RAG（noldus-kb）** | **不微调**，永远靠 RAG |
| 批判性视角（主动找替代解释） | prompt | 显著提升（这是微调最能发力的点） |
| 判定 `overall_confidence` | prompt | 中等提升（calibration） |

关键保护：**文献引用永远走 RAG**。微调模型的"记忆"不能当文献源用——这点在第 3.3 节已经强调，这里再点一次因为很容易被"反正微调后它知道文献"误导。

### 6.8 质疑与开放问题

落地前需要和你对齐的三个开放问题：

1. **reviewer 模型怎么选**：v0.1 用与 data-analyst 同模型（claude-sonnet-4-6）最简单，但违反"独立性"原则（§3.3）。替代方案：用不同 model 配置（比如 GLM-5.1 vs claude-sonnet-4-6 互审），或同模型但用不同 system prompt 激活"批判模式"。第一次实装建议同模型起步，观察互审质量后再决定
2. **审核循环是否无限**：如果 data-analyst 修正后 reviewer 还不满意，要不要第二轮？当前设计默认**一轮**（reviewer → data-analyst 修正 → 终稿），多轮由 lead agent 自己判断是否触发（给素材不给规则，§5.1）。但要小心用户体验：超过两轮用户会焦虑，建议在 lead prompt 里默认单轮，用户明确要求"再严格一点"才多轮
3. **降级路径的用户感知**：noldus-kb 不可用时，是否要在 lead agent 的回复里明确告知用户"本次审核未接入文献库"？我倾向**是的**，但这意味着 SubtaskCard 需要展示 reviewer metadata，前端要多一个组件。这个 UI 改动是否进 v0.1？

---

## 7. 范式补全标准模板

### 7.1 为什么需要标准模板

v0.1 9 月硬指标要求 5 个范式完整可用。当前只有 shoaling 一个——意味着剩下 3-5 个月要交付 4 个新范式（epm、open_field、forced_swim、morris_water_maze）。

如果每个范式都从零设计，4 × 2 周 = 8 周纯工程时间，远超预算。解决办法是**把 shoaling 抽象成标准模板**，新范式只填空，不重新发明。

这章做两件事：
1. 定义"范式完整"的标准（必须交付哪些文件 + 必须通过哪些测试）
2. 定义"范式接入 agent"的标准（必须怎么注册、prompt 怎么改）

### 7.2 范式交付的"八件套"

每个范式补全后，必须交付以下 8 个组件。把它做成 checklist 是为了：新增范式时照着勾选，每一项的完成标准明确可验。

| # | 组件 | 位置 | 内容 | 测试 |
|---|---|---|---|---|
| 1 | 模板 | [`ethoinsight/templates/<paradigm>.py`](../../packages/ethoinsight/ethoinsight/templates/) | 参数 + 工作流 + handoff，参照 shoaling.py 217 行结构 | 能独立跑通 demo 数据 |
| 2 | 指标函数 | [`ethoinsight/metrics.py`](../../packages/ethoinsight/ethoinsight/metrics.py) 扩展 | 范式特定指标（如 epm 的 closed_arm_time_ratio） | 单元测试覆盖每个指标 |
| 3 | 阈值 | [`ethoinsight/assess.py`](../../packages/ethoinsight/ethoinsight/assess.py) 的 `_DEFAULT_THRESHOLDS[<paradigm>]` + 可选 YAML | 正常/异常判定阈值，注明文献来源 | 阈值边界测试 |
| 4 | 图表配置 | [`ethoinsight/charts.py`](../../packages/ethoinsight/ethoinsight/charts.py) | 默认图表类型 + 坐标轴/标签/颜色 | 视觉回归（snapshot） |
| 5 | 分析规划提示 | `skills/custom/ethoinsight-planning/references/<paradigm>.md` | 告诉 agent 这个范式怎么规划分析 | 无（prompt 文件） |
| 6 | 范式识别 | lead prompt 新增场景标注 | "如果文件含 'EPM' 关键词或 arm/center 字段 → paradigm = epm" | 无（prompt） |
| 7 | E2E demo 数据 | `demo-data/<paradigm>/` | 至少 1 组 demo 轨迹 | 作为 8 的输入 |
| 8 | E2E 测试 | 回归测试套（§9） | 从 demo 数据到 handoff JSON 的端到端 | golden-case assertion |

**"范式完整可用"的定义**：8 件套全部交付、对应测试全绿、lead agent 能识别并走通完整流程产出合理结论。

### 7.3 模板骨架（从 shoaling.py 抽象）

shoaling.py 的结构里，以下部分**完全可复用**，新范式改值不改结构：

```python
# 参数块（每个范式必须有）
FILE_PATTERN = ...                # 文件通配
PARADIGM = "<name>"               # 范式标识
GROUPS = {...}                    # 分组配置
METRICS_TO_COMPUTE = [...]        # 本范式标准指标列表
CHART_TYPES = [...]               # 本范式默认图表
OUTPUT_DIR = ...
HANDOFF_PATH = ...

# 工作流块（固定 6 步，跨范式相同）
# Step 1: parse.parse_batch()
# Step 2: metrics.compute_paradigm_metrics()
# Step 3: stats.compare_groups()  (可选)
# Step 4: charts.generate_*()
# Step 5: assess.assess_results()
# Step 6: _write_handoff(code_executor_handoff)
```

范式特异的部分：
- `METRICS_TO_COMPUTE` 列表（shoaling 有 4 个、epm 预计 6 个、open_field 预计 5 个）
- `CHART_TYPES` 选择（shoaling 用 box_plot、epm 建议加 heatmap 展示 arm 时间分布）
- 可选的范式特定预处理（epm 需要识别开闭臂标签）

**行动项**：落地时应该把 shoaling.py 里的通用骨架抽到 `templates/_base.py`，提供 `generate_paradigm_template(paradigm_name, metrics, charts)` 函数。shoaling.py 改为调用这个基类。未来新增范式 3-5 分钟搞定 80% 代码。

### 7.4 指标函数的"契约"

每个新增的指标函数（metrics.py 扩展）必须满足：

**输入契约**：接受 `ParsedData` dict 和 `groups` 配置（可选），不接受其他参数。如果需要配置（阈值、时间窗），从 assess.py 的 YAML 读，不做参数。

**输出契约**：
```python
{
    "<metric_name>": {
        "per_subject": {subject: value, ...},
        "group_means": {group: mean, ...},
        "group_stds": {group: std, ...},
        "unit": "cm" | "ratio" | "count" | "s",
        "n_valid": int,
        "n_invalid": int,  # 缺失/异常值
    }
}
```

per_subject 是**必须**的（Layer B 的洞察审核依赖它，§6.3a 的 Subject 3 例子就靠这个）。C6 已经为 shoaling 验证过这条路径。

**命名契约**：`<metric>_<unit>` 或 `<behavior>_<measure>`，如 `distance_moved_cm`、`closed_arm_time_ratio`。和文献里的常见命名一致，便于 noldus-kb 检索时关键词匹配（为 Phase 2 quality-reviewer 的文献检索铺路）。

### 7.5 阈值的"来源"义务

assess.py 里每个阈值必须有**来源注释**：

```python
_DEFAULT_THRESHOLDS = {
    "epm": {
        "open_arm_time_ratio": {
            "normal_range": (0.15, 0.25),    # Source: Walf & Frye 2007, Nat Protoc
            "high_anxiety": {"below": 0.10}, # Source: Pellow et al. 1985
            "unit": "ratio",
        },
    },
}
```

这不只是文档洁癖——quality-reviewer 审核"这个值是焦虑典型"这种声明时，需要能追溯阈值来源。没有来源的阈值在 reviewer 眼里是"无支撑断言"，会被标记 `contradicted` 或 `insufficient_evidence`。

**阈值冲突怎么办**：文献之间经常有矛盾（大鼠 vs 小鼠、5 分钟 vs 10 分钟测试）。阈值 YAML 可以分物种/分实验时长分层。不要自己裁决——让阈值配置里体现这种条件性，data-analyst 根据上下文选用。

### 7.6 范式的优先级与节奏

v0.1 5 个范式的优先级排序（基于用户需求频率 × 实装难度）：

| 顺序 | 范式 | 预估工期 | 难点 | 依据 |
|---|---|---|---|---|
| 1（已完成） | shoaling | — | — | 基线 |
| 2 | **EPM**（高架十字迷宫） | 1.5 周 | 开闭臂区域识别 | 焦虑研究最常用、文献阈值最明确 |
| 3 | **Open Field** | 1 周 | 中心/外周区划分 | 指标计算简单、阈值文献充分 |
| 4 | **Forced Swim**（强迫游泳） | 2 周 | 不动时间识别需要速度阈值 | 抑郁模型经典范式、但行为状态检测复杂 |
| 5 | Morris Water Maze | 2-3 周 | 学习曲线拟合、目标象限统计 | 学习记忆范式核心、时间维度分析复杂 |

合计 6.5-7.5 周，在 5 月中-7 月中窗口内完成是现实的。

**关键建议**：2 和 3（EPM + Open Field）先做——它们结构最接近 shoaling，抽象 `templates/_base.py` 就是在做 2 的过程中完成，3 直接受益。4 和 5 留到 `_base.py` 稳定后再做，避免频繁返工。

### 7.7 能力分层归属

| 组件 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| 模板（`<paradigm>.py`） | ✅ 代码 | — | — | — |
| 指标函数 | ✅ 代码 | — | — | — |
| 阈值 | ✅ 函数 | ✅ YAML | — | — |
| 图表配置 | ✅ 代码 | — | — | — |
| 分析规划提示（skill reference） | — | — | — | ✅ prompt |
| 范式识别 | — | — | — | ✅ lead prompt |
| Demo 数据 | — | ✅ 文件 | — | — |
| E2E 测试 | ✅ 代码 | — | — | — |

范式补全的大部分工作在 L1/L2——**这是纯工程**，不是 agent 设计。微调的主要收益在 L4 的 skill reference 和 lead 识别（术语准确度、范式识别正确率）。

---

## 8. 异常诊断模式库

### 8.1 为什么这是独立一章

当前 data-analyst 能做离群诊断（C6 的 outlier_findings），但覆盖面窄：
- 只针对 shoaling 的 mean_nnd 单指标
- 只用"per_subject 值 + 反事实排除"一种检测思路
- 没有文献支撑的类型学（"这是什么类型的异常"）

v0.1 要让 data-analyst **系统地**识别异常，不能每个范式重新发明轮子。这章建立**异常模式库**——一个跨范式可复用的分类体系 + 每类异常的检测逻辑 + 对应的洞察措辞。

### 8.2 四种经典异常模式

基于行为学实验常见问题归纳：

| 模式 | 本质 | 典型触发 | 对结论的影响 |
|---|---|---|---|
| **A. 个体表型变异** | 群体内自然存在的行为型差异 | 某个体指标远离群体均值但无技术问题 | 可能被误解为药物效应或病理 |
| **B. 混杂因素** | 非实验变量影响了指标 | 动物体重差异、测试时辰不同、雌雄混合 | 实验组差异归因错误 |
| **C. 统计离群** | 数据分布上的极端值（可能技术性） | 跟踪丢失导致距离过高、卡顿导致速度过低 | 拉偏组均值、改变显著性 |
| **D. 设备/采集故障** | 硬件或软件问题 | 摄像头遮挡、轨迹中断、坐标漂移 | 数据无效但看起来合理 |

这四类不是互斥的——Subject 3 的 NND=70.02 可以同时是 A（探索型表型）**或** C（统计离群）**或** D（跟丢导致距离虚高）。reviewer 的职责之一就是**在这几种解释之间辨别**（对应 §6.3 的 "alternative_explanation" verdict）。

### 8.3 每种模式的检测逻辑

**模式 A — 个体表型变异**

- **L1 检测工具**：`detect_phenotype_variation(per_subject, metric, k=1.5)` — 用 IQR 方法找出偏离但未极端的个体
- **L3 文献核查**：quality-reviewer 查 noldus-kb "metric + species + phenotype" 确认是否有已知亚群
- **区分标志**：个体在多个指标上一致偏离（比如高 NND + 高运动量 + 低 group cohesion）→ 表型；单指标偏离 → 更可能是 C
- **洞察措辞**：`"Subject X 在 [指标列表] 上一致偏离群体均值，可能属于文献报告的 <phenotype_name> 表型亚群"`

**模式 B — 混杂因素**

- **L1 检测工具**：`detect_confounders(data, candidates=["body_weight", "age", "sex", "test_time"])` — 检查是否有非分组变量与指标高度相关
- **L2 配置**：可疑混杂变量清单按范式/物种定义
- **区分标志**：组间差异在控制某个变量后消失/反转
- **洞察措辞**：`"treatment vs control 的 mean_nnd 差异（p=0.03）在控制体重后变为 p=0.12，提示差异可能由体重驱动而非药物"`

**模式 C — 统计离群**

- **L1 检测工具**：已有 — `values > mean + 2*SD` 或 IQR × 3
- **反事实排除**：已实现 — 排除后重算组均值（C6 的 shoaling 验证）
- **区分标志**：排除后组间差异消失，且该个体的指标值在生物学合理范围外（比如斑马鱼单秒速度 > 200 cm/s）
- **洞察措辞**：`"Subject X 的 <metric> 显著高于 mean+2SD；排除后组间差异从 p=0.03 变为 p=0.21，原显著性主要由该个体驱动"`

**模式 D — 设备/采集故障**

- **L1 检测工具**：`detect_tracking_artifacts(trajectory)` — 检查轨迹的瞬时跳变（> 物理合理速度）、长时间静止（> 5 秒不动但坐标有微小抖动）、边界粘连
- **区分标志**：轨迹可视化有明显 artifact；指标值在生物学不可能的范围
- **洞察措辞**：`"Subject X 的轨迹数据存在 <N> 次瞬时跳变（速度 > 200 cm/s），提示跟踪丢失；建议排除该样本或修复数据"`

### 8.4 检测逻辑的分层归属

每种模式的实现跨越三层：

| 层 | 作用 | 典型内容 |
|---|---|---|
| **L1** | 检测算法 | `detect_phenotype_variation`、`detect_confounders`、IQR 离群、轨迹跳变检测 |
| **L2** | 配置阈值 | "速度上限"、"静止时长阈值"、"可疑混杂变量清单" |
| **L4** | 解读与辨别 | 在 A/B/C/D 之间辨别、生成洞察措辞、判断严重程度 |

关键设计：**L1 的检测工具是互相独立的**，同一份数据可以同时被多种检测器跑过，结果都给 data-analyst 看。由 **L4** 综合判断"这个异常最可能是什么类型"。这体现了 §4.2 的原则——**候选算法作为工具给模型，不把判断写死**。

### 8.5 与 quality-reviewer 的协作

异常诊断特别容易出"看起来对但其实错"的结论。reviewer 在这里的职责尤其重要：

- data-analyst 说"Subject 3 是统计离群"（C）→ reviewer 查文献发现该物种有 20% 探索型表型 → 改标注为"可能是表型变异（A），建议增加样本"
- data-analyst 说"treatment 组抑郁样"（基于低运动量）→ reviewer 发现 treatment 组体重平均高 30% → 标注"存在混杂因素（B），需控制体重后重分析"

异常诊断 → reviewer 核查是 v0.1 最能体现"accuracy + 可见性 > 速度"价值的环节（§2）。

### 8.6 扩展性：新范式怎么加

新范式（如 Morris Water Maze）的异常模式通常不需要新增 A-D 之外的类别，而是：
1. 在 L1 加范式特定的检测器（如 MWM 的"象限偏好异常"）
2. 在 L2 补该范式的阈值/物种配置
3. L4 的 prompt（data-analyst 的异常诊断指引）加上该范式的常见异常类型

如果出现 A-D 无法覆盖的新异常类型（比如社交行为范式的"社交回避"），再扩展模式库。**不要预设一堆用不到的类别**——YAGNI。

### 8.7 微调可分离性

| 能力 | v0.1 实现 | v0.1+ 微调提升 |
|---|---|---|
| 检测算法（L1） | 代码 | **不微调** |
| 阈值配置（L2） | YAML | **不微调** |
| 在 A/B/C/D 之间辨别 | data-analyst prompt | 显著（这是最典型的领域判断） |
| 生成洞察措辞 | 结构化模板 + prompt | 中等（术语准确度） |
| 严重程度判定 | prompt | 中等（calibration） |

异常类型辨别是微调 Qwen3-8B 的**黄金训练场景**——每次用户确认或修正 data-analyst 的异常判定，都是一条高质量训练数据。

### 8.8 本章状态：待行为学同事优化

第 8 章的模式库（A/B/C/D 分类、每种的区分标志、洞察措辞模板）是**工程视角的初版草案**，不是最终版。正式实装前必须经行为学同事过一遍：

- A-D 四类够不够？会不会有遗漏的常见异常？
- 每类的区分标志是否符合行为学实际判定习惯？
- 洞察措辞是否符合研究员的阅读预期、文献引用习惯？
- 各范式（EPM/OFT/FST/MWM）的特异性异常需要补什么？

**行动项**：EPM 范式补全（M0.2）开工前，请行为学同事对本章做一次结构化 review，结果以 diff 形式合入本文档。之后每补一个新范式，同步补该范式的异常清单。

---

## 9. 验收体系

### 9.1 为什么这是必须的一章

前面 8 章定义了"做什么、怎么做"。但设计文档最容易犯的错误是——**没有验收标准，优化变成空谈**。

"agent 判断能力提升"如果不能被量化，下次会话来了就变成无根据的 vibe check："感觉比上次好了""感觉不如之前"。更糟的是：prompt/模型一改就可能悄悄退化，没人知道。fix4 → fix3 的退化就是这个情形的前车之鉴。

这章建立一套**可自动化、可重复、能发现退化**的验收体系。

### 9.2 验收的三个层次

从成本低到高、从粗到细排列：

**Layer 1 — 数值回归**（已有基础）
- 对象：Layer A（数值输出）
- 方法：pytest 单元测试 + ethoinsight 库的 demo 数据 assertion
- 状态：backend 1660 passed / ethoinsight 131 passed，基础扎实
- 作用：防止 metrics/assess/statistics 代码改动导致的数值偏移
- 新增需求：每补一个新范式，同步补该范式的数值回归测试（对应 §7.2 八件套 #8）

**Layer 2 — E2E 流水线回归**（部分存在）
- 对象：从 raw data 到 handoff JSON 的完整 pipeline
- 方法：固定 demo 数据 → 跑完整流程 → 断言 handoff_*.json 的关键字段
- 当前覆盖：shoaling E2E 跑通（手动验证），但没有自动化 E2E assertion
- 作用：防止 pipeline 结构改动导致的链路断裂（subagent 对接、handoff schema 变更）
- 新增需求：把 shoaling E2E 自动化（作为 v0.1 基线），然后 EPM/OFT 依葫芦画瓢

**Layer 3 — 判断质量 Golden-Case**（v0.1 新建）
- 对象：Layer B（洞察判断）
- 方法：行为学同事提供的**标注数据集** — 每个 case 包含 raw data + 专家期望的结论
- 每次系统改动后跑：agent 输出 vs 专家标注的匹配度
- 作用：防止"不崩但变笨"的退化（比如 fix4 的"退化到只说'存在异常'"）
- 这是 v0.1 最关键的新基建

### 9.3 Golden-Case 数据集的获取

你提出的方向——让行为学同事给具体实例数据——是可行的。但需要明确的是**要的是什么形式**，不能只给原始 .txt。每个 golden case 的理想结构：

```
golden-cases/
└── case-001-shoaling-control-vs-drugX/
    ├── raw-data/
    │   └── *.txt                              # EthoVision 导出的轨迹
    ├── expected-analysis.yaml                 # 专家期望的分析结果
    ├── metadata.yaml                          # case 背景（物种、范式、实验条件）
    └── notes.md                               # 行为学同事的观察笔记（可选）
```

`expected-analysis.yaml` 是核心，示例：

```yaml
paradigm: shoaling
species: zebrafish
groups:
  control: [Subject 1, Subject 2]
  drug_X: [Subject 3, Subject 4, Subject 5]

expected_findings:
  - type: outlier_detection
    subject: "Subject 3"
    metric: mean_nnd
    expected_value_range: [65, 75]
    reasoning: "该个体表现探索型表型特征，NND 基线偏高"
    severity: moderate

  - type: confound_note
    variable: subject_count_imbalance
    reasoning: "control n=2, treatment n=3，样本量不均且偏小"

  - type: statistical_conclusion
    claim: "组间差异受 Subject 3 驱动"
    required_keywords: ["Subject 3", "排除", "反事实"]
    forbidden_claims: ["药物效应显著"]
```

这份 YAML 是**机器可读的专家标注**。E2E 跑完后，自动比对：
- agent 输出有没有提到 Subject 3？（严格匹配）
- mean_nnd 的值是否在 expected_value_range 里？（数值 assertion）
- 结论里是否包含 required_keywords、不包含 forbidden_claims？（字符串匹配）

### 9.4 Golden-Case 的渐进构建

一次性收集 5-10 个完整 case 不现实。建议的渐进路径：

| 阶段 | 目标 | 来源 | 时间 |
|---|---|---|---|
| **Phase 0** | 1 个 shoaling 基线 case | 用 fix4/fix5 已跑过的 thread 数据（6f046cc7...）+ 补 expected-analysis.yaml | 1 周内 |
| **M0.2 期间** | +1 个 EPM case | 行为学同事提供 | 与 EPM 实装同步 |
| **M0.3 期间** | +1 个 OFT case | 行为学同事提供 | 与 OFT 实装同步 |
| **v0.1 交付前** | 累计 5+ case，覆盖 5 个范式 | 行为学同事 | 9 月前 |
| **v0.1+** | 扩到 15+ case，含边缘场景（小样本、缺失数据、设备故障数据） | 持续收集 | 持续 |

每个 case 的前置成本：行为学同事~2 小时标注 + 工程师~1 小时转 YAML + 写测试。**总投入可控**。

### 9.5 自动化运行与告警

Golden-case 回归**必须自动化**，否则形同虚设：

1. **本地运行**：`make test-golden` 跑所有 case，默认跳过（耗时 5-10 分钟/case）
2. **CI 触发条件**：改动涉及 `lead_agent/prompt.py`、`subagents/builtins/*.py`、`ethoinsight/assess.py`、`ethoinsight/metrics.py` 时自动跑
3. **结果报告**：通过/失败/降级（以前过现在不过）三态，失败时输出 agent 实际输出 vs 期望对照
4. **告警阈值**：任何之前通过的 case 失败 → 阻止 merge（和其他测试一样严格）

这一层的实现需要新增 `tests/test_golden_cases.py` + 一个 runner 脚本，工期约 3 天（不含 case 本身的收集）。

### 9.6 不是所有判断都适合 golden-case

Golden-case 擅长测"agent 有没有抓住关键结论"。但有些判断**本质上是开放的**，不适合用 golden-case 硬卡：

- 报告的行文风格、段落组织
- 次要洞察的措辞选择
- 图表的视觉细节

这些靠**人工抽检**——v0.1 发布前由行为学同事走一遍 demo，给出 rubric 评分（§9.7）。

### 9.7 补充：专家 rubric 抽检

对 golden-case 覆盖不到的"开放质量"维度，v0.1 交付前让 2-3 位行为学同事按固定 rubric 打分：

| 维度 | 1 分（差） | 5 分（好） |
|---|---|---|
| 结论准确性 | 关键结论错误 | 与专家判断一致 |
| 推理透明度 | 黑箱结论 | 每条结论都能追到数据 |
| 文献引用质量 | 编造或错误引用 | 引用准确且切题 |
| 措辞专业性 | 术语误用 | 符合行业用语习惯 |
| 异常识别 | 漏掉明显异常 | 准确识别并给出可操作建议 |

v0.1 验收标准：**每个维度平均分 ≥ 4/5**。低于标准的维度进入 v0.1+ 迭代清单。

### 9.8 本章与其他章节的呼应

- 对 §3 能力分层：Layer 1 测 L1、Layer 2 测 L1+L3 联动、Layer 3 测 L4 的输出质量
- 对 §4 Layer A/B 模型：Layer 1 是 Layer A 的验收、Layer 3 是 Layer B 的验收
- 对 §6 quality-reviewer：reviewer 自身的质量也要进 golden-case，某些 case 专门测"reviewer 能不能抓住 data-analyst 的无支撑声明"
- 对 §7 范式补全：每个新范式必须同时交付 golden-case（§7.2 八件套 #8）
- 对 §8 异常诊断：每种异常模式 A/B/C/D 在 golden-case 里应该各有至少 1 个覆盖

---

## 10. 时间节奏

### 10.1 执行计划的归属

本文档定义**设计**（做什么、为什么这么做），执行节奏由 [2026-04-21-finetuning-strategy-update.md §3](2026-04-21-finetuning-strategy-update.md) 统一规划。原因：微调计划和范式补全在时间上紧密交织（Step 2 EPM 实装 ↔ golden-case case-002；Step 4 SFT 训练 ↔ 合成数据审核），拆成两份时间表会相互引用混乱。

### 10.2 本文档各章在执行计划里的映射

| 本文档章节 | 对应执行 Step | 时间窗 |
|---|---|---|
| §7 范式补全（EPM）+ §5 场景参考表 v0.1 落 prompt | Step 2 | 5 月 |
| §6 quality-reviewer 降级版 + §7 范式补全（OFT） | Step 3 | 6 月 |
| §5 lead prompt 里场景素材微调数据化 + §6 reviewer 审核模板化 | Step 4 | 7 月 |
| §9 Golden-case runner + case-001 | Step 1 | 4 月末 |
| §9 Rubric 抽检 + v0.1 验收 | Step 5 | 8 月 |
| §6 Reviewer 主路径（noldus-kb 接入） | Step 6 或 noldus-kb 恢复后 | 外部依赖驱动 |
| §8 异常模式库正式版（行为学同事 review 后） | 跨 Step 2-5 持续 | 5-8 月 |

### 10.3 关键依赖链

三条必须按顺序的依赖，打破任一条会导致返工：

1. **Golden-case case-001 → EPM 实装 → case-002**：有 case-001 作样板，行为学同事才知道怎么标注 case-002；没有 case-002，EPM 的判断质量无法验收
2. **EPM/OFT 跑通 → quality-reviewer 原型 → SFT 数据采集**：reviewer 要在真实 E2E 上跑，才能产生可用于训练的"初版 vs 修正版"对
3. **SFT 数据 ≥ 3 个 golden-case 可用 → 微调启动**：硬闸门，没有验收手段不微调（§9.2 Layer 3）

### 10.4 并行路径

两线并行，减少总时长：

- **工程线**（§5-§8 的实装）和 **行为学线**（§9 golden-case 标注 + §8 模式库 review）独立推进
- **范式补全**（§7）和 **quality-reviewer**（§6）在 Step 3-4 并行，两者只在"reviewer 验证时用 EPM/OFT 的 E2E"节点交汇

### 10.5 检查点（每月底）

每月最后一周做一次总体检查：

- golden-case 数量是否达标？（5 月末 2 个、6 月末 3 个、7 月末 4 个、8 月末 5 个）
- 本月工程产出是否通过自动化回归？
- 行为学同事反馈有没有需要写进 §8 或 §9？
- 有没有需要推迟到 v0.1+ 的开放问题（记入 §11）？

---

## 11. v0.1 不做的事

明确"不做"和明确"做"同等重要——避免 scope creep 和决策反复。

### 11.1 本文档范围外的能力（延后到 v1.0 或后续）

| 能力 | 为什么不做 | 计划归属 |
|---|---|---|
| **跨范式证据链** | 需要多次实验的统一表示、跨 thread 的长期记忆，复杂度远超 v0.1 | roadmap Phase 4 |
| **实验设计指导** | 需要和 Noldus 产品团队深度合作（推荐哪种 EthoVision 设置）、文献综述能力 | roadmap Phase 3 |
| **文献综述生成** | 需要 noldus-kb 检索 + 结构化推理，v0.1 的 knowledge-assistant 偏简单问答 | roadmap Phase 2 M2.1/M2.2 |
| **多用户 / 本地部署** | 产品化问题，v0.1 仍是内部/演示用 | roadmap Phase 5 |
| **跨范式 agent 协作** | Phase 3/4 的愿景"agent A 分析 shoaling、agent B 分析 EPM、agent C 综合推断这组药物效应" | roadmap Phase 4 |

### 11.2 技术选项里不做的

| 选项 | 为什么不做 |
|---|---|
| **PPO 强化学习微调** | 调参复杂度对我们团队规模过高，DPO 已足够（见 finetuning-strategy §1） |
| **全参微调（非 LoRA）** | 成本不划算、不利于 A/B 切换 |
| **Qwen3-8B 以外的基座** | 04-13 文档已锁定，v0.1 不重新评估 |
| **自托管训练基础设施** | Fireworks 已够用、自托管是 v0.1+ 的优化题 |
| **Redis/向量数据库自建 RAG** | noldus-kb 的 MCP 形式已经是 RAG，v0.1 不自建 |
| **多轮互审** | v0.1 默认单轮 reviewer（§6.8 讨论），多轮作为 lead agent 的可选动作不强制 |

### 11.3 行为学判断能力里不做的

| 能力 | 为什么不做 |
|---|---|
| **异常模式 A-D 之外的新分类** | §8.6 YAGNI 原则，除非实战证明 A-D 不够用 |
| **行为表型的自动分类**（如 "探索型 vs 合群型 zebrafish"） | 需要文献共识 + 标注数据，v0.1 范围外（reviewer 引用文献即可） |
| **实时流式分析**（边采集边分析） | 架构不支持、需求不足 |
| **非 EthoVision 数据源** | v0.1 仅支持 EthoVision XT 导出，其他导出格式延后 |
| **统计高级方法**（GLMM、生存分析、Bayesian） | statistics.py 当前决策树覆盖 80% 实验设计，高级方法按需按年迭代 |

### 11.4 用户体验里不做的

| 改动 | 为什么不做 |
|---|---|
| **Reviewer metadata UI 展示**（§6.8 开放问题 3） | 可以用 SubtaskCard 现有文本展示替代，不加新前端组件 |
| **多语言 agent**（英文/日文/德文） | v0.1 只保中文；语言一致性已在 4/21 fix5 解决 |
| **自动报告多样式**（期刊 A vs 期刊 B 格式） | APA 单一格式足够 v0.1；期刊特异格式延后 |

### 11.5 一个总原则

每当有"顺便做掉"的诱惑时，问三个问题：

1. 这件事不做，v0.1 的 6 条硬指标（[finetuning-strategy §6](2026-04-21-finetuning-strategy-update.md)）哪条会失败？
2. 如果没有哪条会失败——这件事真的和 v0.1 有关吗？
3. 如果有关——它的时间预算挤占了哪条硬指标的资源？

大多数时候第 1 问的答案是"都不会失败"。那就记入本章，v0.1 后再说。

---

## 附录 A：术语表

| 术语 | 含义 | 第一次出现 |
|---|---|---|
| **Layer A / Layer B** | 数值输出 / 洞察判断，判断能力的两层模型 | §4.1 |
| **L1 / L2 / L3 / L4** | 能力分层：代码 / 配置 / RAG / 模型 | §3.2 |
| **Golden-case** | 行为学同事标注的期望结论数据集，用于自动化回归验收 | §9.2 |
| **Quality-reviewer** | v0.1 新增 subagent，审核 data-analyst 的洞察 | §6 |
| **DPO 种子** | 互审对话的 (初版, 修正版) 对，未来 DPO 训练的原始数据 | §3.4 |
| **降级路径** | noldus-kb 不可用时 reviewer 只做自洽检查的运行模式 | §6.4 |
| **场景参考表 A-E** | lead agent prompt 里给出的典型场景指引，A 到 E 五类 | §5.3 |
| **八件套** | 每个新范式必须交付的 8 个组件 | §7.2 |

## 附录 B：相关文档索引

- [roadmap.md](../roadmap.md) — 12 个月产品路线图
- [2026-04-21-finetuning-strategy-update.md](2026-04-21-finetuning-strategy-update.md) — 微调策略更新 + 分步执行
- [2026-04-13-fine-tuning-small-model-design.md](2026-04-13-fine-tuning-small-model-design.md) — 微调基础架构（04-13 定稿）
- [2026-04-15-fine-tuning-data-checklist.md](2026-04-15-fine-tuning-data-checklist.md) — 数据 checklist
- [2026-04-20-ethoinsight-pipeline-redesign.md](2026-04-20-ethoinsight-pipeline-redesign.md) — Handoff JSON 契约设计
- [2026-04-21-subtask-visibility-and-language.md](2026-04-21-subtask-visibility-and-language.md) — fix5 执行计划（未追踪）
- [../handoffs/2026-04-21-subtask-visibility-done.md](../handoffs/2026-04-21-subtask-visibility-done.md) — fix5 完成记录

---

**文档状态**: 设计完成，待执行阶段持续更新。
**下一步**: 按 [finetuning-strategy §7](2026-04-21-finetuning-strategy-update.md) 的三个 immediate actions 启动。









