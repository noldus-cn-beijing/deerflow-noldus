# DeerFlow upstream sync (e19bec14 → f9b70713)

**状态**：done
**时间跨度**：2026-05-18 ~ 2026-05-25
**dev HEAD**：`37bcbba4`（2989 passed / 19 skipped / 0 failed）

## 做了什么

DeerFlow 上游 15 个新 commit 全部以 surgical merge 方式合入 Noldus dev 分支，保留所有 Noldus 定制（中文 prompt、5 ethoinsight subagent、4 工具注册、EV19+Intent guardrails、shared workspace、TrainingDataMiddleware、ThinkTagMiddleware、MCP 4096 截断、sandbox extra_env、loop_detection 中文化、verdict 三分类等）。

5 个 PR 分批合入，覆盖 sandbox async polling、MCP session pool、SafetyFinishReasonMiddleware、Langfuse 完整集成、Tier 4 persistence 整文件升级、loop_detection defer warning injection 等模块。每次合入后跑全量测试验证无回归。

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 5/18 | Round 4 完成，剩余 7 commit 待合 | [sync-round4-complete](../handoffs/2026-05/2026-05-18-deerflow-sync-round4-complete-handoff.md) |
| 5/18 | pending 7 commit 四类 surgical merge | [sync-pending](../handoffs/2026-05/2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md) |
| 5/19 | 预上线 deerflow cleanup | [pre-launch-cleanup](../handoffs/2026-05/2026-05-19-pre-launch-deerflow-cleanup-handoff.md) |
| 5/21 | sync 完成后继分析 | [sync-complete-next](../handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md) |
| 5/25 | PR-A + PR-B 合入，PR-C 待启动 | [pra-merged-prb-pending](../handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md) |
| 5/25 | 全部 5 PR 合入，dev HEAD `37bcbba4` | [all-5-pr-merged](../handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md) |
| 5/25 | sync 善后：PROTECTED_FILES 13→22，sync-state 机制就位 | [cleanup-pr-pending](../handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md) |

## 当前状态

- 完成项：上游 15 commit 全部合入，5 PR (#33 #34 #35 #36 #38) 已合，0 test regression
- 遗留项：sync 善后 PR（chore/sync-protected-files-and-sop）push 完待合；sync 基准 + SOP 教训更新
- 下一 milestone：无（本 track 完成）

## 相关 handoff

- [5/18 sync round4 complete](../handoffs/2026-05/2026-05-18-deerflow-sync-round4-complete-handoff.md) — Round 4 收尾
- [5/18 sync pending surgical merge](../handoffs/2026-05/2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md) — 待合 commit 分类
- [5/18 sync progress](../handoffs/2026-05/2026-05-18-deerflow-sync-progress.md) — 进度跟踪
- [5/19 pre-launch cleanup](../handoffs/2026-05/2026-05-19-pre-launch-deerflow-cleanup-handoff.md) — 上线前清理
- [5/21 sync complete and next](../handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md) — 完成后分析
- [5/25 PR-A merged PR-B pending](../handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md) — 分批合入状态
- [5/25 all 5 PR merged](../handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md) — 最终状态
- [5/25 cleanup PR pending](../handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md) — 善后 PR 待合
