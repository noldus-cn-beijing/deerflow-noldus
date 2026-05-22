---
name: ethoinsight-charts
description: >
  chart-maker subagent 的图种知识库（what-to-pick）。
  8 种图表 × 适用场景对照表 + 选择决策树。
  执行工作流（how-to-execute）见姐妹 skill `ethoinsight-chart-maker`。
  Lead 不读本 skill — capability-exposure 模式下，"用户语义 → 图种" 归 chart-maker。
version: 2.0.0
author: noldus-insight
---

# EthoInsight 图表指南（chart-maker 用）

**变更说明 (2026-05-22 PR-2)**:
- chart-maker subagent 通过 `skills=[..., "ethoinsight-charts"]` 把本 SKILL.md 主文注入 system prompt（L2 知识）
- 选图决策树（图种 → 适用场景）留在本 skill；fallback 决策树（catalog 命中 / 模糊语义时的 fallback）在姐妹 `ethoinsight-chart-maker` skill
- references/ 是 L3 深细节，按需 read_file（chart-maker prompt workflow Step 10 触发）

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
