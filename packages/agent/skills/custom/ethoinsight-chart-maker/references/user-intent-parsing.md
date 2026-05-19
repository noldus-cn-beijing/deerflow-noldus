# 用户语义解析规则

## 复数 vs 单数
- "画一个图" / "画 trajectory" → 单选
- "画几个图" / "画一些图" → 复数 → 选多张
- "画图" / "可视化" → 模糊 → 默认全选

## 图种关键词
| 关键词 | 匹配图 id |
|---|---|
| "trajectory" / "轨迹" / "运动路径" | trajectory_plot |
| "timeseries" / "时间序列" / "动态" | timeseries_plot |
| "box" / "箱线图" / "组间比较" | box_* |
| "violin" / "小提琴图" | violin_* |
| "heatmap" / "热图" | heatmap_* |

## 反向匹配(fallback 中只有 trajectory + timeseries 时)
| 用户原话 | 选 |
|---|---|
| 含 "trajectory" / "轨迹" | trajectory_plot |
| 含 "timeseries" / "时间" / "动态" | timeseries_plot |
| 含 "几个" / "都" / "全" | 两个都跑 |
| 其他模糊 | 两个都跑 |
