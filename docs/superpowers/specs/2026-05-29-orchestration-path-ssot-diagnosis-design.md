# 编排路径 SSOT 化 — 诊断 + 设计骨架

**类型**：诊断 + 设计骨架版（v0.2 鲁棒性根治立项依据；非 v0.1 实施 spec）
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) + 2026-05-29 Dynamic Workflows 文章引发的架构深挖
**状态**：诊断已用代码证据核实（见 §2）；设计是骨架，未排期
**前置认知**：[[feedback_single_source_of_truth]]（SSOT 是项目最高原则）+ catalog SSOT 重构（编排路径应学的正面范例）

---

## 0. 一句话

> **我们把"算什么"（范式→指标→图表）工程化成了 catalog SSOT，但"按什么顺序算、何时停下问用户"（编排路径）还停留在 `lead_agent/prompt.py` 的自然语言箭头图里。我们这一个月反复修的鲁棒性 bug（seal 漏调、lead 不读 handoff、意图误判、ask 漏问），根因都是"完整路径只存在于 LLM 要读的自然语言中，运行时只零星打了几个补丁式校验点，三者之间没有 SSOT 桥接"。**

**关键澄清（防误解）**：本诊断针对**编排路径**（orchestration path），**不针对**范式/指标/图表知识。后者（catalog YAML + review-packages）**已经是 SSOT、已经工程化、是正面范例**。不要把本诊断理解成"要重做 catalog"——恰恰相反，编排路径应该**学 catalog 的做法**。

---

## 1. 两类知识，工程化状态相反（必读，否则会修错对象）

| 知识 | 内容 | 现在在哪 | SSOT? | 本诊断的立场 |
|---|---|---|---|---|
| **「算什么」** | EPM 要算哪些指标、出哪些图、置信度分级 | `catalog/<paradigm>.yaml` + `review-packages/` | ✅ 是 | **正面范例，不动** |
| **「怎么算」** | INTENT 选定后，subagent 按什么顺序派、在哪 ask 用户 | `lead_agent/prompt.py:286-294` 箭头图 | ❌ 否 | **本诊断的对象** |

lead 调 `catalog.resolve` CLI 生成 `plan_metrics.json`——这是「算什么」已工程化的铁证：指标清单从 YAML 确定性派生，不靠 LLM 背。**编排路径没有等价物**：没有 `path.resolve`，没有 `path.yaml`，路径只活在 prompt 的箭头图里，靠 LLM 每跳重读重建。

---

## 2. 诊断：编排路径"三层各说各话"（代码证据）

核实方法：把 `prompt.py` 箭头图 与 `guardrails/` 下的 provider 规则 head-to-head 摆开（2026-05-29 实测）。

### 2.1 第 1 层 — prompt 箭头图：唯一完整的路径描述，但是自然语言

`lead_agent/prompt.py:286-294`，8 个 INTENT 各自的完整派遣链：

```
E2E_FULL_ASKVIZ → code-executor → data-analyst → ask(出图?) → [yes]chart-maker → ask(report?)
E2E_FULL        → code-executor → data-analyst → chart-maker → ask(report?)
E2E_MIN         → code-executor → ask(four-choice)
CHART/REPORT/QA_FACT/QA_KNOWLEDGE/CLARIFY → 单跳
```

这是**整个系统唯一**完整描述"路径"的地方。它是 LLM 要读的自然语言，不是机器可执行的结构。

### 2.2 第 2 层 — provider 规则：只覆盖路径的零星几个点

| Provider（`guardrails/`） | 实际检查什么 | 覆盖了路径的多少 |
|---|---|---|
| `intent_classification_provider` | lead 有没有输出 `[intent] X`，X 在不在 8 个合法值（`_VALID_INTENTS`）里 | **0%**：只管"声明了没"，完全不管 X 之后该派谁 |
| `intent_post_step_ask_gate_provider` | **仅** `E2E_FULL_ASKVIZ` 时，data-analyst 完成后没 ask viz 就拦 chart-maker | **8 条路径里 1 条的 1 个 ask 点** |
| `task_handoff_authorization_provider` | 派 X 时 prompt 里有没有 X 的 `required_upstream_handoffs` 占位符 | 管**依赖**，不管**顺序/路径** |

### 2.3 第 3 层 — `required_upstream_handoffs`：唯一的结构化 SSOT，但它是「依赖」不是「路径」

`subagents/builtins/*.py`：
```
code-executor:      required_upstream_handoffs = []
data-analyst:       required_upstream_handoffs = ["code_executor"]
chart-maker:        required_upstream_handoffs = ["code_executor"]
report-writer:      required_upstream_handoffs = ["code_executor", "data_analyst"]
knowledge-assistant: required_upstream_handoffs = []
```

这是 DAG 的**边（依赖）**：chart-maker 需要 code_executor 的产出。但它**不表达路径**：它不知道 "E2E_FULL 下 chart-maker 之前必须先过 data-analyst"。

### 2.4 三个可指认的漂移/洞（每个都是真实的当前风险）

**洞 1：路径骨架只活在自然语言，provider 不校验派遣顺序。**
如果 lead 在 `E2E_FULL` 下**跳过 data-analyst 直接派 chart-maker**——没有 provider 会拦。`task_handoff_authorization` 只查 chart-maker 要的 `code_executor` handoff 在不在（在，code-executor 跑过了），于是**放行**。data-analyst 被静默跳过，整个统计审核环节消失，无人察觉。**这是当前存在的鲁棒性洞。**

**洞 2：ask 点保护是打补丁式的，8 个点只硬保护了 1 个。**
`ask_gate` provider 只拦 `E2E_FULL_ASKVIZ` 的 `ask(viz?)`。箭头图里还有 `ask(report?)`（3 条路径出现）、`E2E_MIN` 的 `ask(four-choice)`——**prompt 说要有，但没有任何 provider 保证它们发生**。lead 漏掉 `ask(report?)` 直接结束？没人拦。这解释了 ask 相关 bug 为何反复出现。

**洞 3：三层用不同命名/粒度，无法机器校验一致性。**
- prompt 用 `code-executor`（连字符）/ provider 数据用 `code_executor`（下划线）
- prompt 表达「路径」（选哪条）/ `required_upstream_handoffs` 表达「依赖」（谁需要谁）——不是一回事
- **没有任何 CI 哨兵**能验证三层一致。改 prompt 不会让 provider 红，改 provider 不会让 prompt 红。

### 2.5 统一诊断：这不是 N 个 bug，是同一个结构缺陷的 N 次发作

| 修过的 bug | 表面 | 真病根 |
|---|---|---|
| seal 漏调（5.7/5.8） | subagent 没调 seal | 长路径中段丢了"该收尾了"的状态 |
| lead 不读 handoff（5-14/5-18 复发） | lead 跳过 handoff 内容 | 唤醒点重入后没重建"上一步产出" |
| 意图误判（ASKVIZ） | 派错 subagent | 重入后重新解释箭头图，解释偏了 |
| Gate 双重提示 | banner + ToolMessage 重复问 | 两个中断机制各自重建状态 |

**根因统一表述**：每个唤醒点（ask/gate）现在是"中断到 END + 下一轮重入"模式（`clarification_middleware` 用 `Command(jump_to END)`，**未用 `langgraph.interrupt`**）。每次重入，lead 的"我走到路径哪一步了"靠重读 prompt 箭头图 + workspace 文件**重建**——重建是 LLM 的概率行为，所以会错。**happy path 能跑（重建大多对）；长路径不鲁棒（唤醒点越多，累积重建失败率越高）。**

---

## 3. 设计骨架：编排路径 SSOT 化（学 catalog 的做法）

**核心思想**：给编排路径一个 `catalog` 等价物——一份结构化 SSOT，prompt 从它**渲染**、provider 从它**生成校验**、CI 哨兵验证一致。**不发明新框架，复用 deerflow 现有原语**（provider 协议 + middleware + SubagentConfig）。

### 3.1 SSOT：一份「路径定义」（位置 + 形态待定）

候选形态（骨架，实施前定）：一份声明式的 path spec，每个 INTENT 一条，描述 step 序列 + 每步类型（dispatch / ask / gate）：

```yaml
# 示意，非最终 schema
E2E_FULL:
  steps:
    - dispatch: code-executor
    - dispatch: data-analyst        # ← 洞 1 修复：顺序进入 SSOT，可被校验
    - dispatch: chart-maker
    - ask: report?                  # ← 洞 2 修复：ask(report?) 进入 SSOT
E2E_FULL_ASKVIZ:
  steps:
    - dispatch: code-executor
    - dispatch: data-analyst
    - ask: viz?                     # ← 现有 ask_gate provider 已硬保护，迁移到此
    - dispatch: chart-maker (if viz=yes)
    - ask: report?
```

**位置候选**：① ethoinsight 包内 YAML（像 catalog，但编排是 harness 职责，可能不合适）；② deerflow 配置；③ 一个 harness 内的 path registry 模块。**实施前需定，且必须是 single source**——绝不 prompt 一份、SSOT 一份（否则只是把双存换个地方）。

### 3.2 从 SSOT 派生三个消费者（消除三层漂移）

| 消费者 | 现状（手写） | SSOT 化后（派生） |
|---|---|---|
| prompt 箭头图 | `prompt.py:286-294` 手写自然语言 | 从 path spec **渲染**成自然语言（lead 仍读自然语言，但它是 SSOT 的投影，不会漂移） |
| 派遣顺序 provider | 不存在（洞 1） | **新增** `PathSequenceProvider`：从 path spec 读"X 之前必须先完成 Y"，拦截乱序派遣 |
| ask 点 provider | `ask_gate` 只硬保护 1 个（洞 2） | 从 path spec **生成**所有 ask 点的拦截规则（统一，不再逐个打补丁） |

### 3.3 唤醒点：评估从「中断到 END + 重入」迁移到状态不丢的形态

**这是最难、也最值得想的一块。地基已于 2026-05-29 核实（结论如下）：**

| 核实项 | 结果 |
|---|---|
| deerflow 有 checkpointer 基础设施吗 | ✅ 有（`runtime/checkpointer/`，`langgraph.json` 注入 `make_checkpointer`，含自定义 `deerflow_saver`） |
| 全仓用过 LangGraph `interrupt()` 原语吗 | ❌ **完全没用**（grep 空）——我们对 interrupt 零经验 |
| 唤醒点现在怎么实现 | `ClarificationMiddleware` 用 `Command(goto=END)` 主动结束 graph，靠下一轮重入；`create_agent()` 调用未传 checkpointer，由 LangGraph Server 外层注入 |

**结论：根治可行但非免费。** checkpointer 地基在（resume 有支撑），但要把唤醒点从 `goto=END` 改造成 `interrupt()`，并验证 `deerflow_saver` + Gateway SSE 流能正确处理 interrupt/resume 往返。这是中等规模改造，不是配置开关。

**因此根治可分两阶段（喂给排期）：**
- **阶段 A（低成本，不碰 interrupt）**：只做 path SSOT（§3.1）+ 派遣顺序 provider（§3.2 洞 1）+ ask 点 provider 从 SSOT 生成（洞 2）。唤醒点仍走"结束+重入"，但重入时读**结构化路径进度**（experiment-context.json 的 gate_completed 已是雏形）而非重新解释 prompt 箭头图——重建从"解释自然语言"降级为"读状态机指针"，错误率大降。
- **阶段 B（根治，引入 interrupt）**：`ClarificationMiddleware` 改 `interrupt()`，状态留 checkpointer，resume 不丢。唤醒点状态从源头不再重建。依赖阶段 A 的 path SSOT 已就位。

旧表述（保留作背景）：
- 现状 `Command(jump_to END)` + 重入 → 状态靠重建
- deerflow 是否能用 LangGraph `interrupt`（状态留 checkpointer，resume 不丢）来实现 ask/gate？**这是事实问题，实施前必须核验 deerflow 的 create_agent + checkpointer 支不支持 interrupt/resume**
- 若支持：唤醒点状态不再重建，洞的根因从源头消失（不只是加校验）
- 若不支持：退而求其次，把"路径进度"显式写进 thread_data（experiment-context.json 已有 gate_completed 的雏形），重入时读它而非重新解释箭头图

### 3.4 CI 哨兵（关键，catalog 同款）

新增测试：验证 prompt 渲染出的箭头图 与 path spec 一致、provider 生成的规则与 path spec 一致。**改了 SSOT 三处自动同步；任何一处手改导致漂移 → CI 红**。这正是 catalog 的 `test_tuning_section_lists_catalog_tunable_params` 哨兵的同款思路。

---

## 4. 与剩余 sprint / roadmap 的关系（不替换，但有一处交集要调）

| Sprint | 关系 | 是否需调整 |
|---|---|---|
| 3（参数审计）/ 4（调参指南）/ 6（记忆）/ 8（feedback） | **正交**，解决别的问题 | 不调 |
| **5（DataQualityGuardrail）** | **同类！** 5 要再加一个 provider。若认编排路径 SSOT 化，5 的 quality gate 应考虑**从 SSOT 生成**而非再手写一个孤立 provider，否则又多一块补丁 | **🟡 建议调整实现方式**（不调目标） |
| 7（假设面板） | 正交（present_assumptions 是聚合查询） | 不调 |

**roadmap 建议**：新增一条（Sprint 9 或 v0.2 首项）"编排路径 SSOT 化"，标注：① 是这一个月所有鲁棒性 bug 的统一根因；② 与 Sprint 5 的实现方式有交集；③ 依赖 deerflow interrupt 可行性核验（§3.3）。

---

## 5. 不在本诊断范围 / 明确不做

- ❌ 重做 catalog / 动范式·指标·图表知识（它们已是 SSOT，是正面范例）
- ❌ 引入 Claude Code 的 Dynamic Workflow JS 引擎（那是 CLI 功能，非我们 infra；且字符串拼接模型丢失 handoff 契约）
- ❌ 抛弃 lead 的智能（lead 仍做"知情路由决策"——选哪条 INTENT 路径；SSOT 化的是"选定后的执行"，不是"选择"本身）
- ❌ v0.1 立即实施（这是 v0.2 鲁棒性根治；v0.1 继续走"护栏逼近"的路 B）

## 6. 给未来实施 agent 的核验清单（骨架→核验版升级）

1. **deerflow 的 create_agent + checkpointer 支不支持 LangGraph interrupt/resume**（§3.3，决定唤醒点能否根治）
2. path spec 放哪是 single source（§3.1，绝不再造双存）
3. 现有 8 个 INTENT 的完整路径（从 prompt 箭头图逐条提取，作为 SSOT 初始内容）
4. `intent_post_step_ask_gate_provider` 如何迁移成"从 SSOT 生成"（它是现有唯一的 ask 保护，是迁移样板）
5. prompt 渲染机制：箭头图段如何从 path spec 模板渲染（参考现有 prompt.py 的 capability 块渲染 `noldus_order`）
6. 与 Sprint 5 的协调（§4，避免再加孤立 provider）

---

## 7. 编排哲学：为什么是声明式而非动态编排（2026-05-29 对话沉淀）

> 本节回答两个反复出现、且最易把 agent 带偏的问题：①"动态编排才像真正的智能 agent，我们要不要上？" ②"既然不上动态，把 SSOT 写成 JS 代码、再写个能消费它的 middleware，是不是更鲁棒/高效？" 把论证钉死，避免下一个读到 Dynamic Workflows 文章的人重走弯路。愿景尺度的延伸（两模式假说）见 [roadmap v2 §"2026-05-29（下午）愿景架构对齐"](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md)。

### 7.1 为什么 Claude Code 做动态编排，而我们不该

**Claude Code 的任务空间是不可枚举的**——"修任意 bug / 写任意功能"，解法流程在执行中动态长出来，事前画不出，所以**只能**让运行时 LLM 即兴决定下一步。动态编排是它在开放任务空间下的**不得已**，不是"先进"。

**我们的任务空间可枚举**——成熟行为学范式，流程教科书级确定。而且**边界是 Noldus 自己用产品定义的**（不是猜外部开放空间）。把"不得已"误读成"先进"，是这个话题最大的认知陷阱。

**三把判据**（详见 roadmap v2 §A）：① 可枚举 vs 不可枚举；② 看重时间不容试错（科学实验"一次做对"）；③ 可复现是科学仪器的命（动态"每次可不同"与之直接对立）。三者叠加 → 从现在到将来都不需要动态编排。

### 7.2 lead 现在就在"动态编排"——SSOT 不是关掉它，是给它装护栏

常见误解：以为 SSOT 化 = 把 lead 变成轨道列车、丢掉智能。**真相**：`path_registry.py` 注释明说"路径的执行由 provider 读数据驱动，不在此运行"——lead 运行时**仍是自由 LLM**，自己判断意图、自己决定调哪个 subagent、自己组织反问。这就是用户想要的"动态编排"，它一直开着。

provider 做的不是"代替 lead 编排"，而是**护栏**：lead 自由开车，PATHS 是车道线——你想去哪自己开，但压实线（跳过 data-analyst 直接派 chart-maker、漏 ask）时拦你。司机没被换掉。

**智能在节点内（判断/解读/决策/沟通），约束在节点间（顺序/ask/可复现）。** 这两层正交——这就是"最智能"和"最鲁棒"同时成立的原因。dogfood thread 87edb29b 的 trace 是活样本：lead 自主推断 FST、结合 memory 选模板、从 0.56s 看出参数未校准（节点内智能），同时被 ask-gate provider 拦住跳步（节点间约束）。

### 7.3 为什么不能把 SSOT 写成 JS 代码（回答"做个能消费 JS 的 middleware 行不行"）

这是个精妙的技术问题，答案是**不能，且会摧毁 SSOT 的全部价值**。

**核心**：SSOT 的价值在"**一份数据被多个消费者多视角读取**"——`path_registry` 被三个消费者读：① prompt 渲染成中文箭头图；② 顺序 provider 校验跳没跳步；③ ask provider 校验漏没漏问。

**声明式数据**描述"路径是什么"，能被**读/渲染/校验/遍历**（CI 哨兵 `for intent, steps in PATHS.items()` 能跑，正因它是可遍历数据结构）。

**命令式 JS** 描述"怎么执行"，**只能被执行，不能被多视角读**：
- 渲染箭头图？得解析 AST = 重新发明一个解释器
- provider 校验"该按什么顺序"？JS 函数体描述的是"怎么做"不是"该是什么"，provider 无从比对
- CI 静态校验全部路径？JS 函数体遍历不了

"写个能消费 JS 的 middleware"——那个 middleware 要"读出路径结构"才能校验，等于把 JS 解析成 AST，**绕一大圈又得到一个数据结构，回到 PATHS 原点**。而且：把路径写成 JS = 渲染/顺序/ask 三处真相**重新分裂成三份**，正是这一个月所有 bug 的统一根因（§2.5），违背 `feedback_single_source_of_truth`。

**效率上也全面更差**：运行时 provider 读 dict 走 list 是纳秒级，真正耗时在 LLM/subagent（秒级），数据结构形态对总耗时影响为零；开发上数据版改一条路径三处自动同步、JS 版三处手改易漏；测试上数据版能静态校验、JS 版只能跑端到端。**"JS 更快/更鲁棒"是错觉，方向反了。**

这也是 §5"明确不做"里"❌ 引入 Dynamic Workflow JS 引擎"的完整论证。`path_registry.py` 顶部"没有 await/while/if 控制流""声明式数据非命令式代码"——看到控制流就是方向错了。

### 7.4 意图路由 ≠ 动态编排（防混淆）

"不知道用户想干嘛"是**路由**问题（有限 INTENT 清单里选 + CLARIFY 兜底，LLM 判断），不是**编排**问题（路径事前画不出）。场景再多（8→800）只是 SSOT 规模变大——规模越大越需要可枚举/可检查/可复现，越不能交给 LLM 即兴。详见 roadmap v2 §A 末段。

### 7.5 infra 层复审：有限路由 vs 图灵完备编排（2026-06-01 对话沉淀，新增第四判据）

> **本节性质**：2026-06-01 把"图灵完备 vs 声明式"这个出发点，从**业务层**（实验设计/分析要不要图灵完备）推进到 **infra 层**（agent 作为新软件范式，它的**调度循环/编排骨架**要不要图灵完备）。结论一致（有限路由），但新增一把判据 + 一手证据链。

**两层必须同一个答案**：业务层基元有限（见 roadmap §D，已被用户 + 行为学同事确认）。如果业务有限、但 infra 调度循环是图灵完备（运行时 LLM 即兴决定下一步派谁），就是用图灵完备引擎跑有限任务——徒增不可复现性，杀鸡用牛刀。**编排范式必须与业务范式对齐：都是有限路由。**

**第四把判据：可观测性 / replay**（前三把"可枚举 / 不容试错 / 可复现"见 roadmap §C）。这把判据是 infra 层独有的，且是压垮图灵完备编排的最后一根稻草：

- 生产级 harness 的"可观测性"要求每一步可记录、可 replay、可 debug。
- **图灵完备编排让这一项变绝症**：每次运行执行路径都不同 → 没有"标准路径"可比对 → bug（派错 subagent、漏 ask、漏 seal）无法定位，因为"它本就是即兴的，没有对错"。
- **反证（这一个月的实证）**：seal 漏调 / lead 不读 handoff / 意图误判 / ask 漏问——这些 bug 能被定位和修复，**正是因为编排路径"本该确定"，所以"偏离"才可见**。若编排本就图灵完备即兴，这些全部变成不可 debug 的"特性"。
- 所以：编排路径 SSOT 化（§3）的 infra 价值不只是"消除三层漂移"，更是**让运行时偏离可观测**——这是有限路由相对图灵完备编排的硬优势。

**有限路由在 infra 层的形态**（与业务层基元化同构）：
```
有限路由（选）：
  INTENT 清单（可枚举意图：配模板/做实验/查记录/分析/问问题）  ← 声明式数据
       ↓ LLM 在有限清单里判断 + 不确定走 CLARIFY               ← 节点内智能
  每条 INTENT 的派遣骨架（先派谁/产什么 handoff/何时 ask）     ← 声明式 SSOT（§3）
       ↓ guardrail provider 按 SSOT 校验运行时偏离             ← 节点间约束（可观测）

图灵完备编排（不选）：
  运行时 LLM 即兴生成编排脚本 → 不可枚举/不可复现/不可 replay/bug 不可定位
```

**四把判据的最终收口**（缺一不可，且在我们场景里只会越来越强）：
1. **可枚举**：任务空间由 Noldus 用 EV19 License 模块矩阵定义边界（业务）+ INTENT 可列举（infra）
2. **不容试错**：实验科学一次做对，将来接采集后毫秒级不可逆
3. **可复现**：科学仪器的命
4. **可观测/可 replay**（infra 新增）：有限路由让偏离可见可 debug；图灵完备编排让 bug 不可定位

**一句话**：不是图灵完备编排，是有限路由（≠ 枚举每条完整路径，是枚举有限 INTENT + 有限派遣基元，路径由组合涌现）。智能不在编排的图灵完备度，在节点内的判断质量。infra 的落地动作 = 把编排路径从 prompt 的自然语言箭头，提升为与 catalog 同级的声明式 SSOT（§3 / roadmap 编排路径 SSOT 阶段 A，P0 最前）。

**关联 harness 审计**：2026-06-01 用"生产级 harness 11 项能力"清单审计本系统——见 roadmap v2 §F。结论：我们是 deerflow harness 的领域定制层，11 项里 9 项由 deerflow 提供，本月所有 sprint 集中在 ⑥调度循环可靠性 / ⑧可观测性 / ⑨安全策略业务侧三项——正是"通用 harness 给地基、生产级可靠性自己补"的三项。有限路由 + 编排路径 SSOT 正是 ⑥+⑧ 的根治方向。
