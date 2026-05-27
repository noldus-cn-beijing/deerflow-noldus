# DeerFlow upstream sync (e19bec14 → 162fb214)

**状态**：ongoing
**时间跨度**：2026-05-18 ~ 2026-05-27
**dev HEAD**：`4df35f44`（全量 3000+ passed，blocking_io 4 passed / 1 skipped）

## 做了什么

DeerFlow 上游 23 个 commit 以 surgical merge 方式合入 Noldus dev 分支，保留所有 Noldus 定制。

### Round 1-5（2026-05-25 前，15 commit，5 PR）

sandbox async polling、MCP session pool、SafetyFinishReasonMiddleware、Langfuse 完整集成、Tier 4 persistence、loop_detection 等。

### Round 6（2026-05-27，8 commit 中取 2 PR）

| PR | 内容 | 类型 |
|----|------|------|
| #3229 | async_provider SQLite path offload + Blockbuster runtime gate + CI | surgical merge |
| #3218 | think tag backtick guard（`<think>` 反引号防护） | surgical merge |

跳过 6 个 PR：PR template、blocking IO inventory（诊断工具）、auth token、clipboard guard、todo ThreadState、MCP session pooling（评估为低风险但非紧急，留后续）。

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

| 5/27 | PR #3229 + #3218 合入，sync 基准 f9b70713→162fb214 | [issue57-fix-sync-3229-3218](../handoffs/2026-05/2026-05-27-issue57-fix-sync-3229-3218-handoff.md) |

## 当前状态

- 完成项：上游 23 commit 全合入，7 PR 已合，0 test regression
- 遗留项：sync state 文件待更新（`.deerflow-sync-state` 仍指 f9b70713）；sync 善后 PR 待合
- 下一 milestone：评估剩余 6 个低风险 PR（PR template / blocking IO inventory / auth token / clipboard / todo / MCP session pooling）并按需合入

## 相关 handoff

- [5/18 sync round4 complete](../handoffs/2026-05/2026-05-18-deerflow-sync-round4-complete-handoff.md) — Round 4 收尾
- [5/18 sync pending surgical merge](../handoffs/2026-05/2026-05-18-deerflow-sync-pending-surgical-merge-handoff.md) — 待合 commit 分类
- [5/18 sync progress](../handoffs/2026-05/2026-05-18-deerflow-sync-progress.md) — 进度跟踪
- [5/19 pre-launch cleanup](../handoffs/2026-05/2026-05-19-pre-launch-deerflow-cleanup-handoff.md) — 上线前清理
- [5/21 sync complete and next](../handoffs/2026-05/2026-05-21-deerflow-sync-complete-and-next-handoff.md) — 完成后分析
- [5/25 PR-A merged PR-B pending](../handoffs/2026-05/2026-05-25-deerflow-sync-pra-merged-prb-pending-prc-todo-handoff.md) — 分批合入状态
- [5/25 all 5 PR merged](../handoffs/2026-05/2026-05-25-deerflow-sync-all-prs-merged-handoff.md) — 最终状态
- [5/25 cleanup PR pending](../handoffs/2026-05/2026-05-25-deerflow-sync-cleanup-pr-pending-handoff.md) — 善后 PR 待合
