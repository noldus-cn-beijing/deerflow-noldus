# Chart catalog P0-P3

**状态**：done
**时间跨度**：2026-05-25（1 天内分批完成）
**dev HEAD**：`eb2852ef`

## 做了什么

按同事拍板的 chart catalog implementation plan，为 6 个范式（FST/TST/EPM/OFT/LDB/Zero Maze）补充 SSOT 要求的 19 张图表 + 2 个脚本注册。分 4 个优先级分批 commit：

- **P0**（fst + tst）：柱状图（均值±SEM）、活动强度图（velocity proxy）、放弃挣扎分布图（eventplot）
- **P1**（zero_maze + ldb）：轨迹图、热区图、箱线图、柱状图
- **P2**（epm + oft）：轨迹图、热区图，通用脚本下沉 `_common/`
- **P3**（oft）：5min bin 时间进程图（运动距离 + 中心区滞留）

同时修复了 paradigm key alignment 问题：`load_catalog` 现在接受学术名 paradigm key（如 `forced_swim`），不再要求用户知道内部 key。

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 5/25 | P0-P3 实施 handoff 发布（需求规格全在此文档） | [chart-catalog-p0-p3](../handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md) |
| 5/25 | P0 fst+tst 6 张图 commit | 同上 handoff 的目标 |
| 5/25 | P1 zero_maze+ldb 8 张图 + 2 注册 commit | 同上 |
| 5/25 | P2 epm+oft 4 张图 commit（通用脚本下沉 `_common/`） | 同上 |
| 5/25 | P3 oft 时间进程图 commit | 同上 |
| 5/25 | paradigm key alignment（PR #41 worktree agent） | [paradigm-key-alignment](../handoffs/2026-05/2026-05-25-paradigm-key-alignment-handoff.md) |

## 当前状态

- 完成项：19 张图 + 2 注册全部完成，9 个 commit 合入 dev；SSOT 要求的图表全部就位
- 遗留项：无
- 下一 milestone：无（本 track 完成）

## 相关 handoff

- [5/25 chart catalog P0-P3 implementation](../handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md) — 需求 + 实施规格（本文档即执行手册）
- [5/25 paradigm key alignment](../handoffs/2026-05/2026-05-25-paradigm-key-alignment-handoff.md) — `load_catalog` 学术名兼容修复
