# Milestone 索引

> 每个 milestone 是一个 feature track 的 checkpoint 总结。
> 想看全局进展从这里开始，想看操作细节下钻到对应 handoff。
> **本索引即本项目的 roadmap**（无独立 `docs/roadmap.md`）。

> ## ⚠️ 当前 v0.1 推进的真实阻塞 = 行为学同事的范式方法论
>
> 两条路卡在同一个上游（**不是工程卡点**，harness/基础设施层均可独立推进）：
>
> - **结构聚合**（[Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)）— 自定义分区粒度按范式聚合，**需逐范式确认聚合语义**（不都是 OR，可能加权/需区分臂身份）。
> - **Golden Cases**（[Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90)）— 微调 benchmark + 回归种子 + SFT 数据评分基准，**只有同事能定义对/错标准**。
>
> 精确待办（降低同事回合成本）见 → [blocked-on-expert-methodology.md](blocked-on-expert-methodology.md)
>
> v0.1 六范式（EPM/OFT/LDB/FST/Zero Maze/TST）的**识别 + 判读领域知识同事已交付**（`ethovision-paradigm-knowledge` skill 内对应文件已填实）；标准命名数据现在就能端到端跑。

## 活跃 feature track

| Feature | 状态 | Milestone | 最新 handoff |
|---------|------|-----------|-------------|
| **EV19 列语义对齐** | Sprint 1 已合 dev（自定义分析区列 HITL 对齐）· **Sprint 2 结构聚合 🔴 阻塞于同事方法论** · [Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) | [column-semantics-alignment.md](column-semantics-alignment.md) | [6/5 design v2](../design/2026-06-05-column-semantics-hitl-design-v2.md) |
| **Skill 优化 → SFT（SkillOpt 方法论）** | 计划已就绪 · **🔴 阻塞于同事 Golden Cases**（benchmark 数量为零）· [Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) | — | [6/4 实施计划](../plans/2026-06-04-skillopt-skill-optimization-plan.md) |
| **DeerFlow upstream sync（21 commit）** | ✅ 已合 dev（74e3e80c→f92a26d5，PR #113，受保护文件 surgical + full-follow 补回 2 处定制） | [deerflow-sync-2026-05-25-all-5-pr-merged.md](deerflow-sync-2026-05-25-all-5-pr-merged.md) | [6/9 sync21 review-fix + 409 merged](../handoffs/2026-06/2026-06-09-sync21-review-fix-and-409-merged-handoff.md) |
| **Gateway 多 worker 切 thread 409 修复** | ✅ 已合 dev（PR #112，前端优雅降级 A + 默认单 worker B；共享 StreamBridge C 留 backlog） | — | [6/9 sync21 review-fix + 409 merged](../handoffs/2026-06/2026-06-09-sync21-review-fix-and-409-merged-handoff.md) |
| **Subagent seal/handoff 鲁棒性** | 核心卡死已解 · 6/8 一批合 dev（seal auto-seal 兜底 + n=1 路由 + 判读语言对齐 + 字段三分裂）· 循环导入 fix 已修（3ffaf672） | [subagent-seal-handoff-robustness.md](subagent-seal-handoff-robustness.md) | [6/9 two-specs sync+409](../handoffs/2026-06/2026-06-09-two-specs-sync-and-409-fix-handoff.md) |
| **SOTA Agent v2** | Sprint 0/1/2a 已合 dev · Sprint 5.7 实施中 | [sota-agent-v2-sprint-0.md](sota-agent-v2-sprint-0.md) | [5/28 Sprint1+2a deploy 准备 + seal bug 真根因](../handoffs/2026-05/2026-05-28-sprint1-2a-deploy-prep-and-seal-bug-handoff.md) |
| **noldus-kb 知识库改造（RAG→grep+SQL+vector）** | 设计 + 调研完成 · 346 篇论文已分类/markdown 化/embedding · 下一步建 MCP 搜索工具（**独立轨道，与范式方法论正交**） | — | [6/10 noldus-kb redesign](../handoffs/2026-06/2026-06-10-noldus-kb-redesign-handoff.md) |
| **Handoff task_context 评审** | done（task_context 线收口，框架认知沉淀） | [handoff-task-context-eval-checkpoint.md](handoff-task-context-eval-checkpoint.md) | [6/4 task-context eval+fixes](../handoffs/2026-06/2026-06-04-handoff-task-context-eval-and-fixes-handoff.md) |
| FST E2E dogfood + ASKVIZ | done | [fst-e2e-7fixes-askviz-intent.md](fst-e2e-7fixes-askviz-intent.md) | [5/25 FST E2E recursion + frontend chart final](../handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md) |
| Chart catalog P0-P3 | done | [chart-catalog-p0-p3.md](chart-catalog-p0-p3.md) | [5/25 chart catalog implementation](../handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md) |
| Subagent role split | done | [subagent-role-split.md](subagent-role-split.md) | [5/19 role split impl](../handoffs/2026-05/2026-05-19-subagent-role-split-impl-handoff.md) |

## 已完成的 feature track

| Feature | 完成日期 | Milestone |
|---------|---------|-----------|
| DeerFlow sync 21 commit (74e3e80c→f92a26d5) + Gateway 409 修复 | 2026-06-09 | [6/9 sync21 + 409 handoff](../handoffs/2026-06/2026-06-09-sync21-review-fix-and-409-merged-handoff.md) |
| Handoff task_context 评审（框架适配性收口） | 2026-06-04 | [handoff-task-context-eval-checkpoint.md](handoff-task-context-eval-checkpoint.md) |
| DeerFlow upstream sync (e19bec14→f9b70713) | 2026-05-25 | [deerflow-sync-2026-05-25-all-5-pr-merged.md](deerflow-sync-2026-05-25-all-5-pr-merged.md) |
| FST E2E dogfood 7 fixes + ASKVIZ intent | 2026-05-21 → 5/25 | [fst-e2e-7fixes-askviz-intent.md](fst-e2e-7fixes-askviz-intent.md) |
| Chart catalog P0-P3 | 2026-05-25 | [chart-catalog-p0-p3.md](chart-catalog-p0-p3.md) |
| Subagent role split + handoff protocol | 2026-05-19 | [subagent-role-split.md](subagent-role-split.md) |
