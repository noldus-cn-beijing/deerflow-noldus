# 设计文档：Handoff 契约化 —— 从「守卫拦截的自由输出」到「节点间声明式契约」

> 状态：设计文档（愿景级，待 review 后切 sprint）
> 日期：2026-06-04
> 关联：
> - 上游问题来源：[`2026-06-04-handoff-task-package-optimization-design.md`](2026-06-04-handoff-task-package-optimization-design.md)（task-package 优化）及其 [review](2026-06-04-handoff-task-package-optimization-design-review.md)
> - 愿景对齐：[`2026-05-29-orchestration-path-ssot-diagnosis-design.md`](2026-05-29-orchestration-path-ssot-diagnosis-design.md)（声明式编排 / 节点间约束）
> - 铁律：`feedback_single_source_of_truth`、`feedback_subagent_seal_deadlock_is_prompt_not_budget`、`feedback_handoff_metrics_field_divergence_mislabels_failed`、`feedback_version_boundary_v01_insight_v10_experiment_harness`

---

## 〇、本文档要回答的问题

不是「handoff 该加哪些字段」（那是 task-package spec 的层次），而是更根本的一层：

> **handoff 当前是什么？它该是什么？两者的差距，在 v1.0 声明式编排路上会造成什么结构性风险？v0.1 能先夯哪块承重墙？**

一句话主张：**handoff 当前是「一堆守卫（prompt + guardrail）拦着的、subagent 自由发挥的输出快照」。v1.0 的声明式愿景要求它是「节点间的声明式契约」——让不合规的 handoff 根本无法被表达，而不是靠守卫事后拦截。**

---

## 一、事实根基：字段三分裂不是「填错」，是「契约根本不存在」

本节的结论全部由 2026-06-04 一次完整核实链坐实（56 真实样本统计 + 落盘时间线 + 写入指纹 + git 时间线），不是推断。

### 1.1 三分裂的真实机制 = 旁路写入

code-executor 的 `handoff_code_executor.json`，顶层指标字段名在 56 个真实落盘样本里三分裂：`metrics`（27）/ `metrics_summary`（24）/ `metrics_results`（1）。

追落盘时间线 + 写入指纹，真相是**版本演进的疤，不是 LLM 填错参数**：

| 时期 | 写入方式 | 字段名 | 指纹 |
|------|---------|--------|------|
| 5-13~5-27 | LLM 直接 `write_file`（当时 prompt 就这么教，seal tool 还不存在） | `metrics` | **缺 `analysis_config_id`、缺 `.lineage/manifest.json`**（34 个旧样本 100% 缺） |
| 5-28（Sprint 0） | 强制走 `seal_code_executor_handoff` | `metrics_summary` | 有 analysis_config_id + manifest |
| 6-03（PR-3） | seal tool + **guardrail 硬拦 write_file** | `metrics_summary` | `handoff.write_file_forbidden` 拦截旁路 |

铁证：34 个旧 `metrics` 样本 **100% 缺 seal tool 专属的 `analysis_config_id` 字段、100% 缺 `.lineage/manifest.json`**——它们根本没经过 seal tool，是 LLM 按当年 prompt 手写 `write_file` 落盘的。

### 1.2 核心洞察：一个 prompt 改动就能制造 34 个不合规样本

5-13 的 prompt 教 LLM「`write_file` 写 handoff」→ 立刻产出 34 个字段名不合规的 handoff，**没有任何机制在落盘时拦住它**。Pydantic schema 当时不校验这条路径，guardrail 当时不拦 write_file。

**这就是「没有契约」的定义**：handoff 长什么样，取决于当时 prompt 怎么教、LLM 怎么发挥，而非一个不可违反的约定。

### 1.3 当前的「契约保障」是软约束拼出来的伪契约

6-03 后旁路确实被堵了，但堵法是三道**软约束**叠加，不是一个**结构性契约**：

| 防线 | 机制 | 弱点 |
|------|------|------|
| prompt「严禁 write_file 写 handoff」 | 劝说 | deepseek 可能不听（`feedback_deny_messages_must_direct` / `skill_describing_tool_output` 实证 prompt 拦不住模型） |
| guardrail 拦 write_file | 字符串模式匹配（路径含 `handoff_` 且 `.json`） | 换写法 / 换工具 / 新范式换字段名都可能绕过 |
| seal tool 的 Pydantic 校验 | 只校验**它收到的 args** | 管不住落盘后是否仍合规；`extra="allow"` 让任何未声明字段静默通过；新范式可另发明字段 |

**本质**：这套体系是「用守卫去拦截不合规行为」，而不是「让不合规行为无法表达」。在 v0.1（4 个 subagent、6 范式、人盯着）能凑合；v1.0 一旦范式增多、基元化后节点动态组合，**守卫必漏**——因为守卫的覆盖面追不上节点组合的爆炸。

---

## 二、为什么这在 v1.0 路上是结构性风险（而非 v0.1 的小 bug）

### 2.1 handoff 是「节点间约束」，而 v1.0 要求节点间是声明式契约

`2026-05-29-orchestration-path-ssot-diagnosis-design` 锁定的愿景判断：

> 全链路声明式（可枚举 / 不容试错 / 可复现）；**智能在节点内 · 约束在节点间**；有限路由 vs 图灵完备收口（第四判据=可观测性）。

**handoff 就是最关键的「节点间」。** 节点内（subagent 怎么算）可以是 LLM 的智能发挥；但节点间传递的东西，必须是**可枚举、可校验、可复现的契约**——否则「全链路声明式」在节点边界处断裂。

当前 handoff 是「守卫拦着的自由输出」，恰恰是这条愿景的反面：它的合规性靠运行时守卫保证，不靠结构保证。**v1.0 的声明式 harness 站在 handoff 这块承重墙上——墙是软约束拼的，v1.0 会在节点间约束上反复塌方。**

### 2.2 契约化一次性消解的，不止字段三分裂

如果 handoff 是真正的契约（字段、语义、不变量都被声明式钉死），本会话挖出的一连串问题**从源头消失**：

| 当前问题 | 契约化后 |
|---------|---------|
| 字段三分裂 | 字段被契约唯一钉死；新范式**扩展契约**而非另发明字段；去掉 `extra="allow"` 偷渡 |
| 校验器硬编码猜字段、fail-open 脆弱 | 校验照契约派生，不再硬编码 |
| 需要事后 evaluator 检查「handoff 对不对」 | 「对不对」在写入时由契约保证，**evaluator 可能根本不需要** |
| task_context 该谁填吵不清 | task_context 是契约一等公民，契约规定谁必填、谁供给 |
| seal deadlock（subagent 烧 turn 起草） | 契约把「该填什么、由谁填」前置声明，subagent 不再现场归纳自由文本 |

**这就是「按愿景改 handoff 收益很大」的具体含义**：收益不在修那 34 个旧样本（它们是死疤、没人读、SFT 不吃——见 §三），而在**把节点间约束从「守卫」升级为「契约」，为 v1.0 声明式编排夯实承重墙**。

### 2.3 与 SkillOpt → SFT 的协同（为什么时机也对）

CLAUDE.md 第 14 条（2026-06-04 启动）：SkillOpt → SFT，handoff 语义会进新生成的 SFT 轨迹。

- 契约化让 handoff 的「正确形状」成为**可声明的标准**，微调模型学的是契约，不是某个时期 prompt 的偶然产物。
- 契约即 eval 标准：「handoff 是否合契约」是确定性可判的，可直接作为 SFT/eval 的自动 reward 信号的一部分（对齐 `2026-06-03-agentic-rl-training-best-practices` 的思路）。

---

## 三、必须澄清的边界：哪些不是本文档要解决的

### 3.1 那 34 个旧样本：是死疤，不在本设计范围

本会话已查实（三层数据管道核实）：

- `TrainingDataMiddleware` 只录对话流（AI text + tool_call args），**不读 handoff 落盘文件**。
- seal tool 参数名是 `metrics_summary`，tool_call args 里永远是 `metrics_summary`——旧 `metrics` 字段只存在于落盘文件，**进不了任何录对话流的管道**。
- `extract_e2e_sessions.py` / SkillOpt 都不扫 workspace 的 handoff 文件。

**结论：34 个旧样本污染风险 = 0。** 它们不是本设计要「清理」的对象。本会话早期一度把 Phase 0 定性为「活 bug / SFT 污染」，**该定性已被后续核实降级**——当前主路径（5-28 后）只产 `metrics_summary`、guardrail（6-03）已堵旁路、SFT 不吃旧文件。task-package spec / review 里仍按早期「活 bug」定性，本文档以此处更新后的定性为准。

### 3.2 v0.1 / v1.0 边界：本文档只夯「handoff = 声明式契约」这一块墙

`feedback_version_boundary_v01_insight_v10_experiment_harness` 铁律：愿景层（实验本体 SSOT / 设计智能 / 基元化）是 v1.0，不进 v0.1 sprint；但 v0.1 的 infra 是 v1.0 的承重墙。

handoff 契约化正好卡这条线：

- **属于 v0.1 能落的承重墙**：把 handoff 升级为声明式契约（格式钉死、校验派生、task_context 是一等公民、堵死旁路的结构化保证）。
- **不展开为 v1.0 全套愿景**：不碰实验本体 SSOT、不做基元化、不做设计智能。契约**为这些留接口**，但不实现它们。

**红线**：本设计若膨胀到「定义整个实验本体」，就越界了。它只回答「节点间传的那个东西，怎么从自由输出变成契约」。

---

## 四、契约要保证的不变量（设计的核心）

无论最终用哪种形态承载（§五），契约必须保证以下不变量。这是验收标准，也是「契约 vs 当前伪契约」的分界：

1. **唯一字段真相（No Divergence）**：每个语义槽位（指标数据、统计、质量警告、task_context…）有且仅有一个字段名。新范式只能**扩展契约声明的槽位**，不能另发明顶层字段。→ 杜绝三分裂。
2. **非法状态不可表达（Make Illegal States Unrepresentable）**：`status=completed` 但无指标数据、`status=partial` 但无 pending 说明——这类组合应在类型/契约层就无法构造，而非运行时检查。
3. **唯一写入门（Single Write Path）**：handoff 只能经契约写入点产生（当前的 seal tool 是雏形）。旁路写入（write_file）在**结构上**不可能产出合规 handoff——而非靠 guardrail 字符串匹配拦。
4. **校验从契约派生（Derived Validation）**：`_validate_handoff_emitted` / evaluator 的检查规则从契约**生成**，不手写硬编码字段名。契约改，校验自动跟。
5. **消费者从契约派生（Derived Consumers）**：下游（data-analyst / chart-maker / report-writer / 前端展示）读 handoff 的取数路径从契约派生，契约改字段则消费者编译期/测试期暴露，不静默崩。
6. **可复现 lineage**：每个 handoff 带契约版本 + analysis_config_id + sha256（manifest 已有雏形），契约演进可追溯。

---

## 五、形态选择（本设计的关键决策点，留待 review 拍板）

契约用什么承载？两条路，trade-off 不同。本节摆清，不替决策者选。

### 方案 ①：强化版 Pydantic Schema（演进现有）

在现有 `handoff_schemas.py` 上：去掉 `extra="allow"`、改 strict、字段名收成唯一真相、加范式扩展机制（新范式注册子契约而非另发明字段）、校验器从 schema 派生。

- **优点**：增量演进、不推翻现有 seal/guardrail、风险低、v0.1 即可落地。
- **缺点**：Pydantic 是「运行时校验」，不变量 2（非法状态不可表达）只能部分做到；schema 仍是「一份 Python 定义」，前端展示元数据 / 校验 / evaluator 仍各自从它手写派生，离「单一真相源」差一层。

### 方案 ②：独立声明式契约层（SSOT 化）

把 handoff 契约抽成一份独立、语言无关的声明（YAML/JSON Schema，类似 review-packages 那种 SSOT）。Pydantic schema、校验器、evaluator、前端展示元数据**全部从它生成**。

- **优点**：真正单一真相源（`feedback_single_source_of_truth` 的正解）；契约语言无关，前后端共享；最贴合 v1.0 声明式愿景；为基元化留好接口（基元的输入输出也是契约）。
- **缺点**：投入大，是 v1.0 级地基工程；需要 codegen 工具链；过度提前可能违反 §3.2 的 v0.1/v1.0 边界。

### 倾向（非决策）

**v0.1 走 ① 的「向 ② 兼容」子集**：先用 ① 把不变量 1/3/4/6 夯实（字段唯一、唯一写入门、校验派生、lineage），但**把字段定义组织成「可被 ② 抽取为独立声明」的形态**（例如字段元数据集中、不散落在 docstring）。这样 v0.1 拿到承重墙，v1.0 做 ② 时是「抽取 + 生成」而非「重写」。

避免两个极端：纯 ① 会在 v1.0 重做；直接 ② 会越过 v0.1 边界、且在契约不变量还没在实践中验证时就上重工具链。

---

## 六、与现有 task-package spec 的关系

task-package spec（task_context / evaluator / Phase 0 字段统一）**不作废，是本契约化的子集 / 先导**：

- 它的 **Phase 0（字段统一到 metrics_summary）** = 本契约不变量 1 的第一步。建议把 Phase 0 重新定性为「契约化的第一块砖」，而非「修活 bug」（定性已降级，见 §3.1）。
- 它的 **task_context 被动数据结构** = 本契约不变量 2（task_context 作为契约一等公民）的雏形。
- 它的 **evaluator** = 在「还没有真正契约」时的过渡补丁。契约化后，evaluator 的多数检查应被「契约写入时保证」取代——本文档 §2.2 已指出 evaluator 可能根本不需要。**建议 review 时重新评估 evaluator 是否仍要独立实现，还是并入契约校验。**

---

## 七、下一步（本文档产出后）

1. **review 本文档**，拍板 §五 形态（① / ②/ ①向②兼容子集）。
2. 拍板后切 sprint：契约不变量 1/3/4/6（v0.1 承重墙）优先；不变量 2/5 跟进。
3. 与 task-package spec 合流：Phase 0 并入契约不变量 1；重估 evaluator 去留。
4. 红线复查：确保未膨胀到实验本体 / 基元化（§3.2）。

---

## 附录：本设计的事实来源（可复现）

- 字段三分裂统计：`find .deer-flow -name handoff_code_executor.json` 全量 56 样本 + import 真实 `_check_code_executor_content`。
- 旁路写入指纹：34 个旧样本 100% 缺 `analysis_config_id` + `.lineage/manifest.json`；git 时间线坐实 seal tool（5-28 aed27729）+ write_file guardrail（6-03 797e42b9）的引入时点。
- 数据管道核实：`TrainingDataMiddleware` / `extract_e2e_sessions.py` / SkillOpt plan 均不读 handoff 落盘文件 → 旧样本污染风险为零。
- red 锚点：`tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence`（`xfail(strict=True)`）。
- 相关 memory：`feedback_handoff_metrics_field_divergence_mislabels_failed`、`feedback_subagent_seal_deadlock_is_prompt_not_budget`。
