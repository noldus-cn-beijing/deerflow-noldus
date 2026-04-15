# 数据质量检查

分析结果中需要关注的质量信号：

- metrics_summary 中某指标所有样本值方差=0 → 常量数据，结果不可信
- 每组 Subject 数量 < 3 → 统计检验力不足
- errors 中出现 "shapiro: Input data has range zero" → 某组数据无变异

## 定制需求处理

- 用户指定指标 → 在 metrics 参数中传入
- 用户指定图表类型 → 在 chart_types 参数中传入
- 用户未指定图表 → 默认 box_plot
