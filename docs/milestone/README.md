# Milestone 索引

> 每个 milestone 是一个 feature track 的 checkpoint 总结。
> 想看全局进展从这里开始，想看操作细节下钻到对应 handoff。
> **本索引即本项目的 roadmap**（无独立 `docs/roadmap.md`）。

> ## ⚠️ 当前 v0.1 行为学分析方向的唯一硬阻塞 = 行为学同事的 Golden Cases
>
> - **Golden Cases**（[Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90)，OPEN）— 微调 benchmark + 回归种子 + SFT 数据评分基准，**只有同事能定义对/错标准**，当前数量为零。卡的是**微调路线**（SkillOpt → SFT → Qwen3-30B），不卡 v0.1 端到端分析。
>
> ✅ **结构聚合（原阻塞，Issue #98）已解除**（#98 于 2026-06-18 CLOSED）：同事逐范式聚合方法论已交付（PR #115）；读码坐实 N 列→1 概念的 OR 聚合机制**全链路早已实现并工作**，milestone 此前标的 blocked 滞后于代码。**功能性上 v0.1 六范式端到端分析现已可用**（识别+判读+聚合全齐、标准与复杂多分区数据都能跑）。剩余只是坐实+固化+补测（[Sprint 2 spec](../superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md)，非新增能力、不阻塞）。
>
> 精确待办（降低同事回合成本）见 → [blocked-on-expert-methodology.md](blocked-on-expert-methodology.md)
>
> v0.1 六范式（EPM/OFT/LDB/FST/Zero Maze/TST）的**识别 + 判读领域知识同事已交付**（`ethovision-paradigm-knowledge` skill 内对应文件已填实）。未支持的全是 v1.0 才做的范式（鱼类/学习记忆迷宫/PhenoTyper/昆虫）。

## 活跃 feature track

| Feature | 状态 | Milestone | 最新 handoff |
|---------|------|-----------|-------------|
| **EV19 列语义对齐** | Sprint 1 已合 dev（自定义分析区列 HITL 对齐）· **Sprint 2 结构聚合机制已在线**（#98 已 CLOSED、方法论已交付；剩余=坐实+固化+补测，非阻塞） · [Sprint 2 spec](../superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md) | [column-semantics-alignment.md](column-semantics-alignment.md) | [6/5 design v2](../design/2026-06-05-column-semantics-hitl-design-v2.md) |
| **文件路径可靠性承重墙** | 📋 立项（治"路径类 bug"家族：DB 外键+run 隔离+解析单点强制；DeerFlow 已有 resolve_virtual_path、缺的自己补）· 未派实施 | — | [6/26 路径可靠性 spec](../superpowers/specs/2026-06-26-file-path-reliability-loadbearing-convergence-spec.md) |
| **Experiment 跨范式对比** | 📋 立项（thread archive 归一为实验 + 快照提取 + synthesizer 跨范式综合；让项目更像 agent）· 工程骨架可建，**判读方法论待行为学同事** · 未派实施 | — | [6/26 experiment init 设计](../superpowers/specs/2026-06-26-experiment-cross-paradigm-comparison-init-design.md) |
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
