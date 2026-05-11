# 数据质量检查

每一步工具完成后都会在 `*_summary.json` 中给出 `quality_warnings` 或 `method_warnings`。
本文档说明这些警告的含义和处理方式。

## parse_trajectories 之后

读 `workspace/parsed_summary.json`：

- `n_files < 3`：样本量不足，先继续分析但在最终返回中提示
- `total_rows < 100`：数据严重不足，检查文件是否完整（EthoVision 正常导出每文件应 >1000 行）
- `columns` 缺少 `x_center_mm` 或 `y_center_mm`：轨迹数据不完整，空间指标无法计算

## compute_metrics 之后

读 `workspace/metrics_summary.json`：

- `group_summary` 中某指标所有组的 `std=0`：常量数据，统计检验会报错
- 某组 `n < 3`：统计检验力不足，优先使用非参数方法
- `quality_warnings` 含 `underpowered`：提示样本量问题
- `quality_warnings` 含 `[critical]` 或 `[warning]` 前缀：来自 ethoinsight 数据边界防御（单鱼输入、n<3 组等），按 severity 决定是否在 handoff 摘要中突出
- `computed_metrics` 远少于预期（例如 shoaling 应有 5 个核心指标但只算出 2 个）：检查 parse 结果的列是否齐全
- **shoaling 专用**：群体指标（mean_iid / mean_polarity）不再出现在 `per_subject`，
  改从 `group_level_metrics` 读取；若该字段是 `{"applicable": false, "reason": "..."}`，
  表示输入只有 1 只鱼，群体指标不适用，handoff 摘要要明确告诉用户

## run_statistics 之后

读 `workspace/statistics.json`：

- `method_warnings` 含 `n=(x,y) — consider non-parametric`：小样本用了参数检验，后续需在 handoff 中标注
- `comparisons` 为空：说明指标数据都被剔除（通常是 metrics 计算阶段的数值全是 NaN）
- `p_value` 全部接近 1.0：可能是分组错误或数据同质

## 实验设计与方法匹配

从 lead agent 的任务描述识别设计类型，判断 run_statistics 自动选的方法是否合适：

- "训练曲线/多天" → 需要 Repeated Measures ANOVA（当前工具不支持，需在 handoff 中标注提示）
- "给药前后 / baseline vs post" → 需要 paired t-test（同上）
- "3 组以上独立" → `run_statistics` 会自动用 one-way ANOVA 或 Kruskal-Wallis
- "对照 vs 实验" → `run_statistics` 会自动用 t-test 或 Mann-Whitney

若设计类型与方法不匹配，在 `assess_and_handoff` 产出的 handoff 的 `errors` 字段中应包含对应提示。
