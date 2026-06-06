# Milestone 索引

> 每个 milestone 是一个 feature track 的 checkpoint 总结。
> 想看全局进展从这里开始，想看操作细节下钻到对应 handoff。

## 活跃 feature track

| Feature | 状态 | Milestone | 最新 handoff |
|---------|------|-----------|-------------|
| **EV19 列语义对齐** | Sprint 1 已合 dev（自定义分析区列 HITL 对齐）· Sprint 2 结构聚合等专家 · [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) | [column-semantics-alignment.md](column-semantics-alignment.md) | [6/5 design v2](../design/2026-06-05-column-semantics-hitl-design-v2.md) |
| **Skill 优化 → SFT（SkillOpt 方法论）** | 计划已就绪 · 等待行为学专家 Golden Cases · [Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) | — | [6/4 实施计划](../plans/2026-06-04-skillopt-skill-optimization-plan.md) |
| **SOTA Agent v2** | Sprint 0/1/2a 已合 dev · Sprint 5.7 实施中 | [sota-agent-v2-sprint-0.md](sota-agent-v2-sprint-0.md) | [5/28 Sprint1+2a deploy 准备 + seal bug 真根因](../handoffs/2026-05/2026-05-28-sprint1-2a-deploy-prep-and-seal-bug-handoff.md) |
| **Subagent seal/handoff 鲁棒性** | 核心卡死已解 · 非空参数路 + 字段三分裂修复完 · data-analyst thinking spec 待实施 | [subagent-seal-handoff-robustness.md](subagent-seal-handoff-robustness.md) | [6/4 task-context eval+fixes](../handoffs/2026-06/2026-06-04-handoff-task-context-eval-and-fixes-handoff.md) |
| **Handoff task_context 评审** | done（task_context 线收口，框架认知沉淀） | [handoff-task-context-eval-checkpoint.md](handoff-task-context-eval-checkpoint.md) | [6/4 task-context eval+fixes](../handoffs/2026-06/2026-06-04-handoff-task-context-eval-and-fixes-handoff.md) |
| DeerFlow upstream sync | ongoing | [deerflow-sync-2026-05-25-all-5-pr-merged.md](deerflow-sync-2026-05-25-all-5-pr-merged.md) | [5/27 issue57 + sync PR #3229/#3218](../handoffs/2026-05/2026-05-27-issue57-fix-sync-3229-3218-handoff.md) |
| FST E2E dogfood + ASKVIZ | done | [fst-e2e-7fixes-askviz-intent.md](fst-e2e-7fixes-askviz-intent.md) | [5/25 FST E2E recursion + frontend chart final](../handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md) |
| Chart catalog P0-P3 | done | [chart-catalog-p0-p3.md](chart-catalog-p0-p3.md) | [5/25 chart catalog implementation](../handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md) |
| Subagent role split | done | [subagent-role-split.md](subagent-role-split.md) | [5/19 role split impl](../handoffs/2026-05/2026-05-19-subagent-role-split-impl-handoff.md) |

## 已完成的 feature track

| Feature | 完成日期 | Milestone |
|---------|---------|-----------|
| Handoff task_context 评审（框架适配性收口） | 2026-06-04 | [handoff-task-context-eval-checkpoint.md](handoff-task-context-eval-checkpoint.md) |
| DeerFlow upstream sync (e19bec14→f9b70713) | 2026-05-25 | [deerflow-sync-2026-05-25-all-5-pr-merged.md](deerflow-sync-2026-05-25-all-5-pr-merged.md) |
| FST E2E dogfood 7 fixes + ASKVIZ intent | 2026-05-21 → 5/25 | [fst-e2e-7fixes-askviz-intent.md](fst-e2e-7fixes-askviz-intent.md) |
| Chart catalog P0-P3 | 2026-05-25 | [chart-catalog-p0-p3.md](chart-catalog-p0-p3.md) |
| Subagent role split + handoff protocol | 2026-05-19 | [subagent-role-split.md](subagent-role-split.md) |
