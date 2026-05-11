# Zero Maze 代码执行参考

> 学术范式 key: `zero_maze`
> EV19 模板映射: `ZeroMaze` 大类下所有变体（`ZeroMaze-AllZones`、`ZeroMaze-NoZones`）
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/zero_maze.md`

## 可用指标函数（在 `ethoinsight.metrics.zero_maze`）

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_open_zone_time_ratio(df, open_zones=None)` | DataFrame | `float \| None` | 开放区时间百分比（开放区帧数 / 总帧数） |
| `compute_open_zone_time(df, open_zones=None)` | DataFrame | `float \| None` | 开放区滞留时长（秒；从 trial_time 列估算 dt） |
| `compute_open_zone_distance(df, open_zones=None)` | DataFrame | `float \| None` | 开放区移动距离占比（开放区移动距离 / 总移动距离） |
| `compute_hesitation_count(df, open_zones=None, closed_zones=None, min_gap_frames=5)` | DataFrame | `int \| None` | 犹豫次数（封闭区探头后缩回；开放区滞留 < min_gap_frames 帧） |

通用指标（任范式可用）:
- `ethoinsight.metrics._common.compute_distance_moved(df) -> float | None` — 总移动距离
- `ethoinsight.metrics._common.compute_velocity_stats(df) -> dict` — 速度描述统计

### 列名自动检测规则

| 指标类型 | Regex 模式 | 典型列名示例 |
|---|---|---|
| 开放区 | `in_zone.*open`（忽略大小写） | `in_zone_open_1`, `in_zone_open_2` |
| 封闭区 | `in_zone.*closed`（忽略大小写） | `in_zone_closed_1`, `in_zone_closed_2` |

多列时 OR 合并（动物在任一开放区段均视为"在开放区"）。

### `compute_hesitation_count` 行为定义

- 当动物从封闭区短暂进入开放区（< `min_gap_frames` 帧，默认 5 帧 ≈ 0.2s at 25 Hz）后立即返回封闭区，计为一次犹豫。
- 犹豫次数与焦虑水平正相关（越焦虑，越频繁探头但不敢停留）。
- 若动物一直在封闭区或开放区（无过渡），返回 0。

## 派发器（一次性算全套）

```python
from ethoinsight.metrics import compute_paradigm_metrics

result = compute_paradigm_metrics(parsed_data, paradigm="zero_maze", groups=groups)
# result = {
#   "paradigm": "zero_maze",
#   "per_subject": {subject_name: {
#       "distance_moved": float,
#       "open_zone_time_ratio": float,   # 开放区时间百分比
#       "open_zone_time": float,         # 开放区滞留时长（秒）
#       "open_zone_distance": float,     # 开放区移动距离占比
#       "hesitation_count": int,         # 犹豫次数
#   }},
#   "group_summary": {group_name: {metric: {mean, std, n, values}}},
#   "data_quality_warnings": [...],      # n < 5/组 或 总移动距离过低时自动警告
#   ...
# }
```

## 胶水脚本范例（end-to-end）

文件名建议: `${workspace_path}/analysis.py`

```python
"""Zero Maze 分析胶水脚本。

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
metrics_result = compute_paradigm_metrics(parsed_data, paradigm="zero_maze", groups=GROUPS)

# 3. 统计
stats_result = statistics.run_groupwise(
    metrics_result["per_subject"],
    groups=GROUPS,
    metrics=["open_zone_time_ratio", "open_zone_time", "open_zone_distance", "hesitation_count", "distance_moved"],
)

# 4. 出图（read ethoinsight-charts skill 后选 box_plot / bar_chart 等）
chart_files = []
# 连续指标（时间比例 / 距离比例）→ box_plot
for metric_name in ["open_zone_time_ratio", "open_zone_distance"]:
    fig_path = WORKSPACE / "outputs" / f"zm_{metric_name}_boxplot.png"
    charts.box_plot(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

# 计数指标（犹豫次数）→ bar_chart
for metric_name in ["hesitation_count"]:
    fig_path = WORKSPACE / "outputs" / f"zm_{metric_name}_bar.png"
    charts.bar_chart(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

# 5. 写 handoff
handoff = {
    "paradigm": "zero_maze",
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
- `distance_moved < 10.0` → `warning` 级警告，提示可能为运动抑制（混杂因素，开放区指标的下降可能不代表焦虑增加）

这些警告会自动进 `metrics_result["data_quality_warnings"]`，不需要胶水脚本额外加。

### 犹豫次数主观性注意

`hesitation_count` 依赖 `min_gap_frames`（默认 5 帧≈0.2s）定义"短暂"。该阈值在不同实验室之间可能不一致。若需跨实验室比较，需在报告中明确标注所用阈值。

## 出图建议（详见 ethoinsight-charts skill）

- `open_zone_time_ratio` / `open_zone_distance`: `box_plot` 或 `raincloud_plot`（连续 + 组比较）
- `open_zone_time`: `box_plot`（连续时长，单位秒）
- `hesitation_count`: `bar_chart` 或 `box_plot`（计数，通常较小）
- `distance_moved`: `box_plot`（用于运动混杂检查，与 open_zone 指标并排）

## 与 EPM 的对应关系

Zero Maze 与 EPM 原理相同，结构不同：

| EPM 指标 | Zero Maze 等价指标 |
|---|---|
| `open_arm_time_ratio` | `open_zone_time_ratio` |
| `open_arm_time` | `open_zone_time` |
| `open_arm_entry_count` | — (Zero Maze 用 `hesitation_count` 替代) |
| `total_entry_count` | — (无中心区，概念不适用) |
| — | `open_zone_distance`（Zero Maze 新增） |
| — | `hesitation_count`（Zero Maze 特有，EPM 无直接对应） |

## handoff JSON 必须字段

见 `../templates/output-contract.md`。Zero Maze 特别需要 `data_quality_warnings` 字段（dispatcher 已自动填）。
