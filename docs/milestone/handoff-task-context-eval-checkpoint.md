# Handoff task_context 评审 + 写入端落地 + 框架适配性结论

**状态**：done（task_context 这条线收口，无续项）
**时间跨度**：2026-06-04
**dev HEAD**：`48ac5003`

## 做了什么

从评审一份「把 handoff 升级为任务状态包」的 spec 出发，通过三轮真实数据核实证明该框架只部分适配 EthoInsight 拓扑，同时挖出并修复了 2 个真实 bug。

**修的两个 bug**：
1. **handoff metrics 字段三分裂**（Phase 0）：56 个真实 handoff 里 28 个完整成功的分析因顶层字段叫 `metrics`/`metrics_results` 而非 `metrics_summary`，被校验器误判残缺 → 误标 FAILED → 触发无效 seal-resume。校验器改为认三字段等价集，非规范格式记 warning 可见。
2. **`pending_items` 恒空 bug**：seal 工具的 `_build_task_context` 从 `errors[]` 派生 pending_items，但真实 partial 场景（统计因 n=1/n=2 跳过）`errors[]` 恒空 → 死代码恒产空 list 且误导。删掉派生逻辑，诚实留空 + 注释说明真实 partial 语义 + TODO。

**落地的能力（写入端阶段一）**：seal 工具内置 `_build_task_context` 纯函数，从 output_files/errors/status 确定性组装 task_context（subagent 零手填，避开 seal deadlock 风险，向后兼容）。

**核心框架认知**（已存 memory）：业界「handoff = 给下游直接接力的任务状态包」框架只适配 EthoInsight 一半——我们是 lead 中转拓扑（subagent → lead → 下一个，下游不直接接手）+ 数据驱动 partial（= 统计因样本量跳过，非失败/未完成），导致 task_context 多数字段对下游冗余、对 lead 被 gate_signals 覆盖。读此类文章别照做，先核实拓扑与真实数据。

## 关键节点

| 日期 | 事件 | handoff / 文档 |
|------|------|----------------|
| 2026-06-04 | 评审原 spec → 三盲点 + Phase 0 活 bug 实证（56 样本）| [review doc](../superpowers/specs/2026-06-04-handoff-task-package-optimization-design-review.md) |
| 2026-06-04 | Phase 0 spec + 实施 + review 通过 | [phase0 spec](../superpowers/specs/2026-06-04-phase0-handoff-validator-hardening-spec.md) |
| 2026-06-04 | 契约化愿景文档（形态未拍板，留待痛点驱动）| [contract-vision](../superpowers/specs/2026-06-04-handoff-contract-vision-design.md) |
| 2026-06-04 | 写入端 task_context spec + 实施（b8d982b0）| [writeside spec](../superpowers/specs/2026-06-04-task-context-writeside-spec.md) |
| 2026-06-04 | 三轮核实证伪下游/lead 消费价值 + 收尾 spec + 实施（48ac5003）| [reeval spec](../superpowers/specs/2026-06-04-task-context-reeval-and-pending-items-fix-spec.md) |

## 当前状态

- **完成项**：
  - 两个真 bug 已修（字段三分裂 + pending_items 恒空），全量 3717 passed + 4 已知污染，review 通过，已 push。
  - task_context 写入端落地（b8d982b0），subagent 零感知，向后兼容。
  - task_context 正名：TaskContext docstring 写入价值评估 + 真实用户待定 + 架构判据（只装产出方独有、消费方无法自行推导的执行事实）。
  - 三条 memory 沉淀：字段三分裂根因 / 全量测试 4 污染 / task_context 框架不适配 EthoInsight 拓扑。
  - 取消读取端阶段二（下游不需要、gate_signals 已覆盖 lead 需求）。

- **遗留项（TODO，无痛点不做）**：
  - task_context 整体价值：待 v1.0「subagent 直接接力」拓扑或真实「脚本失败型 partial」出现后重评估。
  - pending_items 正确实现：若出现脚本失败型 partial，从 plan_metrics.json metrics[].id 减 metrics_summary 实际 id 差集（id 命名一致，可行）。
  - 契约化形态决策（Pydantic 强化 vs 独立 SSOT）：未拍板，等真有字段漂移痛点。
  - 4 个全量测试 isolation 污染：值得单独立项修。

- **下一 milestone**：无（本线收口）。如有 handoff 优化续项，应由真实痛点驱动（lead 决策出问题 / 字段漂移），而非框架预判。

## 相关文档

- [本会话 handoff](../handoffs/2026-06/2026-06-04-handoff-task-context-eval-and-fixes-handoff.md)
- [原 spec（已评审修订）](../superpowers/specs/2026-06-04-handoff-task-package-optimization-design.md)
- [契约化愿景（形态待决）](../superpowers/specs/2026-06-04-handoff-contract-vision-design.md)
