# LDB (Light-Dark Box / 明暗箱) 代码执行参考

> 学术范式 key: `light_dark_box`
> EV19 模板映射: `NoTemplate` — 需实验员手动配置竞技场，画明区和暗区（无专用 EV19 模板）
> 行为学同事维护的领域知识: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/light_dark_box.md`

## 可用指标函数（在 `ethoinsight.metrics.ldb`）

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_light_time_ratio(df, light_zone="in_zone_light")` | DataFrame | `float \| None` | 明箱时间百分比（明区帧数 / 总帧数） |
| `compute_transition_count(df, light_zone="in_zone_light", dark_zone="in_zone_dark")` | DataFrame | `int \| None` | 穿梭次数（light→dark 和 dark→light 各自的 0→1 跨越，合计）|
| `compute_light_latency(df, light_zone="in_zone_light")` | DataFrame | `float \| None` | 潜伏期：首次进入明区的时间（秒；无 trial_time 时退化为帧索引）|

通用指标（任范式可用）:
- `ethoinsight.metrics._common.compute_distance_moved(df) -> float | None`
- `ethoinsight.metrics._common.compute_velocity_stats(df) -> dict`

## EthoVision 列名约定

LDB 无专用 EV19 模板，实验员手动定义观察区。典型列名：

| 含义 | 推荐列名 | 备注 |
|---|---|---|
| 明区占位指示符 | `in_zone_light` | 0/1，每帧 |
| 暗区占位指示符 | `in_zone_dark` | 0/1，每帧；= 1 - in_zone_light（若无过渡区） |
| 时间轴 | `trial_time` | 秒；缺失时 latency 退化为帧索引 |

## 派发器（一次性算全套）

```python
from ethoinsight.metrics import compute_paradigm_metrics

result = compute_paradigm_metrics(parsed_data, paradigm="light_dark_box", groups=groups)
# result = {
#   "paradigm": "light_dark_box",
#   "per_subject": {subject_name: {light_time_ratio, transition_count, light_latency, ...}},
#   "group_summary": {group_name: {metric: {mean, std, n, values}}},
#   "data_quality_warnings": [...],  # n<5/组 或 transition_count<4 自动警告
#   ...
# }
```

## 胶水脚本范例（end-to-end）

文件名建议: `${workspace_path}/analysis.py`

```python
"""LDB (明暗箱) 分析胶水脚本。

由 code-executor 写，bash 执行。
"""
import json
from pathlib import Path

from ethoinsight import parse, statistics, charts
from ethoinsight.metrics import compute_paradigm_metrics

WORKSPACE = Path("/mnt/user-data/workspace")
RAW_FILES = list(WORKSPACE.glob("inputs/*.txt"))  # 用户上传的 EthoVision 导出
GROUPS = {  # 由 lead 在 task() prompt 里给
    "control": ["subject_1", "subject_2", "subject_3", "subject_4", "subject_5"],
    "treatment": ["subject_6", "subject_7", "subject_8", "subject_9", "subject_10"],
}

# 1. 解析（仍用现成 parse 模块）
parsed_data = parse.parse_trajectories([str(f) for f in RAW_FILES])

# 2. 算指标
metrics_result = compute_paradigm_metrics(parsed_data, paradigm="light_dark_box", groups=GROUPS)

# 3. 统计
stats_result = statistics.run_groupwise(
    metrics_result["per_subject"],
    groups=GROUPS,
    metrics=["light_time_ratio", "transition_count", "light_latency"],
)

# 4. 出图
chart_files = []
for metric_name in ["light_time_ratio", "light_latency"]:
    fig_path = WORKSPACE / "outputs" / f"ldb_{metric_name}_boxplot.png"
    charts.box_plot(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

fig_path = WORKSPACE / "outputs" / "ldb_transition_count_bar.png"
charts.bar_chart(
    metrics_result["per_subject"],
    groups=GROUPS,
    metric="transition_count",
    output_path=str(fig_path),
)
chart_files.append(str(fig_path))

# 5. 写 handoff
handoff = {
    "paradigm": "light_dark_box",
    "metrics": metrics_result,
    "statistics": stats_result,
    "charts": chart_files,
    "data_quality_warnings": metrics_result.get("data_quality_warnings", []),
}
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)
print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
```

## 数据质量自动警告（dispatcher 内已实现）

- `n < 5/组` → `warning` 级警告，提示统计功效不足
- `transition_count < 4` → `warning` 级警告，提示探索动机不足（明箱时间百分比下降的替代解释）

这些警告会自动进 `metrics_result["data_quality_warnings"]`，不需要胶水脚本额外加。

**注意**（领域知识，非自动检测）：
- 若明暗区光照强度差异不足，动物可能无法区分两区，导致数据失效。此类硬件问题需实验前确认，dispatcher 无法自动检测。

## 出图建议（详见 ethoinsight-charts skill）

- `light_time_ratio` / `light_latency`: `box_plot` 或 `raincloud_plot`（连续变量 + 组间比较）
- `transition_count`: `bar_chart` 或 `box_plot`（计数变量）

## handoff JSON 必须字段

见 `../templates/output-contract.md`。LDB 特别需要 `data_quality_warnings` 字段（dispatcher 已自动填）。
