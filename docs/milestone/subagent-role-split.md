# Subagent role split + capability-exposure 重构

**状态**：done
**时间跨度**：2026-05-18 ~ 2026-05-19
**dev HEAD**：`cd5cceca`（PR #8 merge commit）

## 做了什么

解决 5/14 和 5/18 两次同类故障的根本性架构问题：lead agent 不读 handoff、code-executor 角色过载（既算指标又画图）、catalog 无 fallback。

核心重构：
1. **Subagent 从 4 个拆为 5 个**：code-executor（只算指标）、data-analyst、report-writer、knowledge-assistant、chart-maker（新建，专责画图）
2. **Capability-exposure 架构**：每个 SubagentConfig 宣告 `when_to_use` / `input_contract` / `output_contract` / `required_upstream_handoffs`，lead 按 contract 调度
3. **Lead prompt 瘦身 1243→395 行**：删除 ~340 行 noldus_rules，细节移入 `ethoinsight-lead-interaction` skill
4. **两个 GuardrailProvider**：IntentClassificationGuardrailProvider（意图路由）+ TaskHandoffAuthorizationProvider（handoff 强制消费）
5. **Catalog fallback**：`_common.yaml` + `resolve_charts()` 新增，plan.charts=[] 时自动加载通用图表

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 5/18 | 故障诊断 + grill-me 12 轮设计讨论 | [lead-handoff-consumption](../handoffs/2026-05/2026-05-18-lead-handoff-consumption-and-role-split-handoff.md) |
| 5/18 | grill-me 完成，spec 待写 | [subagent-role-split-spec](../handoffs/2026-05/2026-05-18-subagent-role-split-spec-handoff.md) |
| 5/19 | 22 Task 全部落地，PR #8 merge | [role-split-impl](../handoffs/2026-05/2026-05-19-subagent-role-split-impl-handoff.md) |
| 5/19 | bugfix 修复 | [role-split-bugfix](../handoffs/2026-05/2026-05-19-subagent-role-split-bugfix-handoff.md) |

## 当前状态

- 完成项：5 subagent + capability-exposure + 2 guardrail + lead 瘦身 + catalog fallback，2604 tests pass
- 遗留项：无
- 下一 milestone：[FST E2E dogfood](fst-e2e-7fixes-askviz-intent.md)（新架构上的首次端到端验证）

## 相关 handoff

- [5/18 lead handoff consumption + role split 设计讨论](../handoffs/2026-05/2026-05-18-lead-handoff-consumption-and-role-split-handoff.md) — 故障诊断 + 5 重根因
- [5/18 subagent role split spec](../handoffs/2026-05/2026-05-18-subagent-role-split-spec-handoff.md) — grill-me 12 轮终态架构
- [5/19 role split impl](../handoffs/2026-05/2026-05-19-subagent-role-split-impl-handoff.md) — 22 Task 实施详情
- [5/19 role split bugfix](../handoffs/2026-05/2026-05-19-subagent-role-split-bugfix-handoff.md) — bugfix
