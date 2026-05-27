# FST E2E dogfood 7 fixes + ASKVIZ intent

**状态**：done
**时间跨度**：2026-05-20 ~ 2026-05-25
**dev HEAD**：`cd512536` → `eb2852ef`（后续修复在此基础上叠加）

## 做了什么

用户对 5 个真实范式数据做端到端 dogfood，本 track 覆盖 FST 完整循环的陪跑修复。核心成果：

1. **7 个 FST E2E bug 修复**（5/21）：sync regression 恢复 3 个工具注册、timeseries 空图修复、plan_charts.json 宿主机路径修复、新增 ASKVIZ 意图（模糊语义先解读再反问要不要出图）、前端双 thinking layout 统一 + 流式 thinking 闪烁消除、反问前 lead 必须先汇报发现
2. **3 个后续修复**（5/20）：handoff #1 #2 #3 修复 + frontend 流式 thinking + 32 个预存在测试 fail 归零
3. **6 个结构性问题修复**（5/25）：recursion_limit 动态计算、frontend B2+B3 lead 汇报可见性、paradigm key alignment、chart catalog P0-P3 实施（分发到独立 worktree agent）

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 5/20 | E2E followup 3 issues | [e2e-followup-3-issues](../handoffs/2026-05/2026-05-20-e2e-followup-3-issues-handoff.md) |
| 5/20 | 3 fix + TTFT + test cleanup (PR #18) | [e2e-3fix-ttft](../handoffs/2026-05/2026-05-20-e2e-3-fix-plus-ttft-plus-test-cleanup-handoff.md) |
| 5/20 | FST E2E dual bug fix | [fst-e2e-dual-bug-fix](../handoffs/2026-05/2026-05-20-fst-e2e-dual-bug-fix-handoff.md) |
| 5/21 | 7 fixes + ASKVIZ intent（HEAD `cd512536`） | [fst-e2e-7fixes-askviz](../handoffs/2026-05/2026-05-21-fst-e2e-7-fixes-and-askviz-intent-handoff.md) |
| 5/25 | 6 结构性问题修复（recursion + frontend + catalog + charts） | [fst-e2e-recursion-chart-final](../handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md) |

## 当前状态

- 完成项：FST E2E 流水线修复完毕，所有已知 bug 已合入 dev；ASKVIZ 意图路由就位；测试基线 0 退化
- 遗留项：EPM/OFT/LDB/Zero Maze 其余 4 个范式 dogfood 待跑
- 下一 milestone：[chart-catalog-p0-p3.md](chart-catalog-p0-p3.md)（5/25 完成）

## 相关 handoff

- [5/20 E2E followup 3 issues](../handoffs/2026-05/2026-05-20-e2e-followup-3-issues-handoff.md) — handoff #1 #2 #3 问题
- [5/20 3fix + TTFT + test cleanup](../handoffs/2026-05/2026-05-20-e2e-3-fix-plus-ttft-plus-test-cleanup-handoff.md) — PR #18 5 commit
- [5/20 FST E2E dual bug fix](../handoffs/2026-05/2026-05-20-fst-e2e-dual-bug-fix-handoff.md) — 双 bug 修复
- [5/20 EV19 template tool](../handoffs/2026-05/2026-05-20-ev19-template-tool-handoff.md) — 模板工具
- [5/21 FST E2E 7 fixes + ASKVIZ](../handoffs/2026-05/2026-05-21-fst-e2e-7-fixes-and-askviz-intent-handoff.md) — 核心成果
- [5/25 FST E2E fixes + chart catalog](../handoffs/2026-05/2026-05-25-fst-e2e-fixes-and-chart-catalog-handoff.md) — chart catalog 穿插
- [5/25 FST E2E recursion + frontend + chart final](../handoffs/2026-05/2026-05-25-fst-e2e-recursion-frontend-chart-final-handoff.md) — 最终修复
