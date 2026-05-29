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
