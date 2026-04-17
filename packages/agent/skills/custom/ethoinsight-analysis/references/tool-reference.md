# 工具参数快速参考

## parse_trajectories

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| file_pattern | 是 | str | 文件匹配模式，如 `/mnt/user-data/uploads/*.txt` |
| workspace_dir | 否 | str | 默认 `/mnt/user-data/workspace/` |

产物：`workspace/parsed.pkl`, `workspace/parsed_summary.json`

## compute_metrics

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| paradigm | 是 | str | 范式名（shoaling, open_field, epm 等）|
| groups | 是 | str | JSON 字符串，例如 `'{"control":["Subject 1"]}'` |
| metrics | 否 | str | 逗号分隔指标名，空则用默认 |
| workspace_dir | 否 | str | 默认 `/mnt/user-data/workspace/` |
| output_dir | 否 | str | 默认 `/mnt/user-data/outputs/`（metrics.csv 位置）|

依赖：`workspace/parsed.pkl` 必须存在。
产物：`workspace/metrics.pkl`, `workspace/metrics_summary.json`, `outputs/metrics.csv`

## run_statistics

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| alpha | 否 | float | 默认 0.05 |
| correction | 否 | str | 默认 `bonferroni` |
| workspace_dir | 否 | str | 默认 `/mnt/user-data/workspace/` |
| output_dir | 否 | str | 默认 `/mnt/user-data/outputs/` |

依赖：`workspace/metrics.pkl` 必须存在。
产物：`workspace/statistics.json`, `outputs/statistics.json`

自动处理：Shapiro-Wilk 正态性检验 → 自动选择 t-test/Mann-Whitney/ANOVA/Kruskal-Wallis；
Levene 方差齐性检验 → independent vs Welch t-test。

## generate_charts

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| chart_types | 否 | str | 逗号分隔；可选值：box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot。默认 `box_plot` |
| metrics | 否 | str | 逗号分隔指标名，空则绘制所有 computed_metrics |
| include_trajectory | 否 | bool | 默认 True（生成 2D 轨迹图）|
| include_timeseries | 否 | bool | 默认 True（shoaling 专用 IID/polarity 时序图）|

依赖：`workspace/metrics.pkl`（必须）、`workspace/statistics.json`（可选，有则在图上画显著性星号）、`workspace/parsed.pkl`（可选，用于 trajectory 图）。
产物：`outputs/{metric}_{chart_type}.png`, `workspace/charts.json`

## assess_and_handoff

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| paradigm | 是 | str | 与 compute_metrics 相同 |
| groups | 是 | str | JSON 字符串，与 compute_metrics 相同 |
| handoff_path | 否 | str | 默认 `/mnt/user-data/workspace/handoff_code_executor.json` |

依赖：前 4 步全部完成（读取 workspace 下所有 *_summary.json）。
产物：`workspace/handoff_code_executor.json`（含 metrics_summary, statistics, assessment, output_files, metadata, errors）

## 最终产物文件树

```
/mnt/user-data/
├── outputs/
│   ├── metrics.csv
│   ├── statistics.json
│   ├── {metric1}_box_plot.png
│   ├── {metric2}_box_plot.png
│   ├── trajectory.png
│   ├── inter_individual_distance_timeseries.png   (shoaling 专有)
│   └── group_polarity_timeseries.png               (shoaling 专有)
└── workspace/
    ├── parsed.pkl                   (内部中间态)
    ├── parsed_summary.json          (可 read_file 查看)
    ├── metrics.pkl                  (内部中间态)
    ├── metrics_summary.json         (可 read_file 查看)
    ├── statistics.json              (可 read_file 查看)
    ├── charts.json                  (可 read_file 查看)
    └── handoff_code_executor.json   (最终 handoff，交给 lead agent)
```
