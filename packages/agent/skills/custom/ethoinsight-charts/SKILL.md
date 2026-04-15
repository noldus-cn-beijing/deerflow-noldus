---
name: ethoinsight-charts
description: >
  Chart selection guide for behavioral data analysis. Lists all available
  ethoinsight chart types with usage scenarios, data requirements, and
  selection rules. Use when deciding which chart to generate for behavioral data.
version: 1.0.0
author: noldus-insight
---

# EthoInsight 图表选择指南

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
