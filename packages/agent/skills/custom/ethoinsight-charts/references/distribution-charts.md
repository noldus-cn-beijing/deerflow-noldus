# 分布类图表（组间对比）

使用频率最高的图表类型，通过 `charts.<函数名>()` 调用，输出 300 DPI PNG。

## 可用图表

| 图表 | 函数名 | 适用场景 | 数据要求 |
|------|--------|---------|---------|
| **云雨图** | `raincloud_plot` | 首选！半小提琴+箱线图+抖动散点三合一，顶刊标配 | 每组 ≥ 5 个样本 |
| **蜂群图** | `beeswarm_plot` | 小样本（n < 15）每个数据点可见，叠加均值±SEM | 每组 3-15 个样本 |
| **箱线图** | `box_plot` | 经典选择，适合快速查看分布和异常值 | 每组 ≥ 3 个样本 |
| **小提琴图** | `violin_plot` | 查看分布形态（双峰/偏态），大样本优先 | 每组 ≥ 10 个样本 |
| **柱状图** | `bar_chart` | 均值±误差棒，最传统但信息量最少 | 任意样本量 |

## 统一调用签名

```python
from ethoinsight import charts

# 所有分布类图表的签名完全相同，可以直接替换函数名
path = charts.raincloud_plot(
    metrics=m,                      # compute_paradigm_metrics() 的返回值
    metrics_to_plot=["mean_iid"],   # 要画的指标列表
    significance=stat,              # compare_groups() 的返回值（可选）
    output_path="/mnt/user-data/outputs/mean_iid_raincloud.png"
)
```

支持的函数：`box_plot`, `bar_chart`, `violin_plot`, `raincloud_plot`, `beeswarm_plot`

## 在模板中使用

在分析模板的 `CHART_TYPES` 参数中指定：

```python
CHART_TYPES = ["raincloud_plot", "beeswarm_plot"]
```

模板会自动为每个指标 x 每种图表类型生成一张 PNG。
