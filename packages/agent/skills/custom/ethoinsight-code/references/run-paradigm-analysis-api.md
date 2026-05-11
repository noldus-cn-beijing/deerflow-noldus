# run_paradigm_analysis 工具 API

## 参数说明

| 参数 | 必填 | 说明 | 示例 |
|------|------|------|------|
| paradigm | 是 | 范式名称 | "shoaling", "open_field", "epm" |
| file_pattern | 是 | 文件匹配模式 | "/mnt/user-data/uploads/*.txt" |
| groups | 是 | JSON 分组映射 | '{"control": ["Subject 1"], "treatment": ["Subject 3"]}' |
| metrics | 否 | 逗号分隔的指标名 | "distance_moved,mean_iid" |
| chart_types | 否 | 逗号分隔的图表类型 | "box_plot,violin_plot" |

## 返回值

JSON 包含以下字段：
- `status`: "completed" 或 "failed"
- `summary`: 分析摘要文本
- `output_files`: 含 metrics (CSV)、statistics (JSON)、charts (PNG 列表)
- `metrics_summary`: 每组每指标的 mean/std/n
- `statistics`: 统计检验详细结果
- `assessment`: 领域评估（表型推断、置信度）
- `errors`: 警告和错误列表

## 结果处理规则

- status == "completed" 且 errors 为空 → 分析成功，直接返回
- status == "completed" 但 errors 不为空 → 分析完成但有警告，记录后返回
- status == "failed" → 检查 errors，修正参数后重试一次

## 实验设计类型识别

从任务描述中识别设计类型：
- "训练曲线/多天/多时间点/longitudinal" → 重复测量设计
- "给药前后/baseline vs post" → 配对设计
- "3组以上/多剂量/low-mid-high" → 多组独立设计
- "对照 vs 实验" → 两组独立设计

## 自动处理的决策（无需干预）

- 正态性检验 → 参数/非参数方法选择
- 方差齐性 → independent-t / Welch-t 选择
- 多重比较 → Tukey HSD / pairwise 选择
- 效应量 → Cohen's d / Hedges' g / ω² 自动计算
- 样本量 < 5/组 → 自动使用非参数方法
