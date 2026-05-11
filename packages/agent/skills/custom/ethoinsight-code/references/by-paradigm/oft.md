# OFT (Open Field Test) 代码执行参考

> 学术范式 key: `open_field`（注意：dispatcher 用 snake_case，不是 "oft"）
> EV19 模板映射: `OpenFieldRectangle` / `OpenFieldCircle` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/open_field.md`

## 可用指标函数（在 `ethoinsight.metrics.oft`）

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_center_time_ratio(df, center_zone="in_zone_center")` | DataFrame | `float \| None` | 中心区域时间占比（中心帧数 / 总帧数） |
| `compute_center_distance_ratio(df, center_zone="in_zone_center")` | DataFrame | `float \| None` | 中心区域移动距离占比（中心位移 / 总位移） |
| `compute_center_entry_count(df, center_zone="in_zone_center")` | DataFrame | `int \| None` | 进入中心区域次数（0→1 跨越，首帧为 1 也算 1 次） |
| `compute_thigmotaxis_index(df, arena_center=None, arena_radius=None, periphery_fraction=0.2)` | DataFrame | `float \| None` | 趋触性指数（周边区滞留偏好），可从 zone 列或坐标计算 |

通用指标（任范式可用）:
- `ethoinsight.metrics._common.compute_distance_moved(df) -> float | None` — 总移动距离（用于运动混杂检查）
- `ethoinsight.metrics._common.compute_velocity_stats(df) -> dict` — 速度统计（mean/max/min/std）

## 派发器（一次性算全套）

```python
from ethoinsight.metrics import compute_paradigm_metrics

result = compute_paradigm_metrics(parsed_data, paradigm="open_field", groups=groups)
# result = {
#   "paradigm": "open_field",
#   "per_subject": {subject_name: {center_time_ratio, center_distance_ratio, center_entry_count, thigmotaxis_index, ...}},
#   "group_summary": {group_name: {metric: {mean, std, n, values}}},
#   "data_quality_warnings": [...],  # n < 5/组 或 distance_moved < 1000cm 自动警告
#   ...
# }
```

## 胶水脚本范例（end-to-end）

文件名建议: `${workspace_path}/analysis.py`

```python
"""OFT 分析胶水脚本。

由 code-executor 写，bash 执行。
"""
import json
from pathlib import Path

from ethoinsight import parse, statistics, charts
from ethoinsight.metrics import compute_paradigm_metrics

WORKSPACE = Path("/mnt/user-data/workspace")
RAW_FILES = list(WORKSPACE.glob("inputs/*.txt"))
GROUPS = {  # 由 lead 在 task() prompt 里给
    "control": ["subject_1", "subject_2", "subject_3", "subject_4", "subject_5"],
    "treatment": ["subject_6", "subject_7", "subject_8", "subject_9", "subject_10"],
}

# 1. 解析
parsed_data = parse.parse_batch([str(f) for f in RAW_FILES])

# 2. 算指标
metrics_result = compute_paradigm_metrics(parsed_data, paradigm="open_field", groups=GROUPS)

# 3. 统计
stats_result = statistics.run_groupwise(
    metrics_result["per_subject"],
    groups=GROUPS,
    metrics=["center_time_ratio", "center_distance_ratio", "center_entry_count", "thigmotaxis_index"],
)

# 4. 出图（read ethoinsight-charts skill 后选图）
chart_files = []
for metric_name in ["center_time_ratio", "center_distance_ratio"]:
    fig_path = WORKSPACE / "outputs" / f"oft_{metric_name}_boxplot.png"
    charts.box_plot(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

# 轨迹图（OFT 经典图）
trajectory_path = WORKSPACE / "outputs" / "oft_trajectory_plot.png"
charts.trajectory_plot(
    parsed_data["subjects"],
    groups=GROUPS,
    output_path=str(trajectory_path),
)
chart_files.append(str(trajectory_path))

# 5. 写 handoff
handoff = {
    "paradigm": "open_field",
    "metrics": metrics_result,
    "statistics": stats_result,
    "charts": chart_files,
    "data_quality_warnings": metrics_result.get("data_quality_warnings", []),
}
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)

# 给 lead 的结构化决策信号——让 lead 不读 handoff 也能做 Step 1.5 拦截。
# 由 Python 代码确定性生成，不依赖模型推理。lead 解析这个块的存在 +
# critical_count > 0 → 反问用户。详见 spec docs/superpowers/specs/2026-05-11-subagent-file-is-facts-design.md
warnings = handoff.get("data_quality_warnings", [])
critical = [w for w in warnings if w.get("severity") == "critical"]
warn_only = [w for w in warnings if w.get("severity") == "warning"]

print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
print()
print("[gate_signals]")
print("data_quality:")
print(f"  critical_count: {len(critical)}")
print(f"  warning_count: {len(warn_only)}")
print("  critical_items:")
if critical:
    for w in critical[:5]:
        msg = (w.get("message") or "")[:80]
        print(f"    - {msg}")
else:
    print("    (none)")
status = handoff.get("status", "completed")
if status == "failed":
    validity = "failed"
elif critical:
    validity = "warning"
else:
    validity = "ok"
print(f"statistical_validity: {validity}")
print(f"errors_count: {len(handoff.get('errors', []))}")
```

## 数据质量自动警告（dispatcher 内已实现）

- `n < 5/组` → `warning` 级警告，提示统计功效不足
- `total distance_moved < 1000 cm` → `warning` 级警告，提示可能为运动抑制（混杂因素）

这些警告会自动进 `metrics_result["data_quality_warnings"]`，不需要胶水脚本额外加。

## 出图建议（详见 ethoinsight-charts skill）

- `center_time_ratio` / `center_distance_ratio`: box_plot 或 raincloud_plot（连续 + 组比较）
- `center_entry_count`: bar_chart 或 box_plot（计数数据）
- `thigmotaxis_index`: box_plot 或 violin_plot（0-1 连续值）
- 轨迹图 (`trajectory_plot`): OFT 经典可视化，展示各 subject 的移动路径 + 中心/周边热区

## handoff JSON 必须字段

见 `../templates/output-contract.md`。OFT 特别需要 `center_time_ratio` / `center_distance_ratio` 两个互补指标（时间 vs 距离），以及 `distance_moved` 用于运动混杂排除。
