---
name: ethoinsight-charts
description: >
  服务对象:**chart-maker subagent**。图种 → 适用场景对照表,chart-maker
  决策选哪些图时的查询源。Lead 不读本 skill — capability-exposure 后
  "用户语义 → 图种" 归 chart-maker,lead 不持图选择 know-how。
version: 2.0.0
author: noldus-insight
---

# EthoInsight 图表指南（chart-maker 用）

**变更说明 (2026-05-18 W9)**:
- 服务对象从"lead agent"变为"chart-maker subagent"
- lead 不再读本 skill 决策图种
- "用户语义 → 图种" 决策树移到 `ethoinsight-chart-maker` skill (W21)
- 本 skill 保留作为图种适用性查询源 (chart-maker 在 W21 skill 决策时 reference 这里)

## 图种 → 适用场景对照表

(保留现有内容)

ethoinsight 库提供 8 种图表，通过 `charts.<函数名>()` 调用，输出 300 DPI PNG。

## 选择决策树

```
需要展示什么？
├── 组间对比（最常见）
│   ├── 发表级别/汇报 → raincloud_plot（首选）
│   ├── 小样本 n < 15 且需要看每个数据点 → beeswarm_plot
│   ├── 想看分布形态 → violin_plot
│   ├── 快速预览 → box_plot
│   └── 只需要均值对比 → bar_chart
├── 指标间关联
│   └── correlogram
├── 运动轨迹
│   └── trajectory_plot
└── 时间趋势
    └── timeseries_plot
```

## Reference Materials

- 分布类图表（5 种）的详细规格和调用签名：`references/distribution-charts.md`
- 关联类图表（correlogram）：`references/association-charts.md`
- 空间/时间类图表（trajectory、timeseries）：`references/spatial-temporal-charts.md`
- 标准/快速/论文级推荐组合：`templates/chart-combinations.md`
