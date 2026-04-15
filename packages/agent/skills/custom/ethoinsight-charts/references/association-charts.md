# 关联类图表

## 相关矩阵图 (correlogram)

多指标间 Pearson 相关性热力图。

- **适用场景**: 探索多个行为指标之间的关联关系
- **数据要求**: ≥ 2 个指标，每个有 ≥ 5 个值

### 调用签名

```python
from ethoinsight import charts

# correlogram 的签名略有不同：没有 significance 参数
path = charts.correlogram(
    metrics=m,
    metrics_to_plot=["mean_iid", "mean_nnd", "polarity"],  # 可选，None=全部指标
    output_path="/mnt/user-data/outputs/correlogram.png"
)
```
