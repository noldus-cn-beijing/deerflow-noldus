# Catalog 字段映射表

各 role 按需查 catalog YAML 的字段：

| YAML 字段 | 消费 role | 用途 |
|-----------|-----------|------|
| `id` | lead, all | metric 唯一标识 |
| `script` | lead, code-executor | 执行脚本 dotted path |
| `requires_columns` | lead (resolve 自动检查) | 列依赖 glob patterns |
| `output_unit` | data-analyst | 输出值单位（用于效应量解释） |
| `display_name_zh` | report-writer | 中文展示名 |
| `unit_zh` | report-writer | 中文单位 |
| `one_liner` | report-writer | 一句话解释（首次提及用） |
| `direction_for_anxiety` | data-analyst | 判读方向 lower_is_anxious / higher_is_anxious / null |
| `statistical_default` | data-analyst | 默认统计方法 groupwise_compare / paired_compare |

## direction_for_anxiety 详解

- `lower_is_anxious`: 指标值越低 → 焦虑样回避越明显（如 EPM 开放臂时间比例、OFT 中心区时间比例）
- `higher_is_anxious`: 指标值越高 → 焦虑样行为越明显（如零迷宫犹豫次数）
- `null`: 该指标不属于焦虑判读维度（如总运动量、FST 不动时间属于抑郁域）

## 判读铁律

1. 只看组间显著差异 + 效应量方向，**不参考绝对阈值 / 常模**
2. direction_for_anxiety 只用于判断"差异方向是否符合实验假设"
3. 若 direction_for_anxiety 为 null，不要强行套焦虑解释框架
