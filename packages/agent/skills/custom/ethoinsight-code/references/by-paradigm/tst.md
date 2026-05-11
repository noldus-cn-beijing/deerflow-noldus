# TST (Tail Suspension Test / 悬尾测试) 代码执行参考

> 学术范式 key: `tail_suspension`
> EV19 模板映射: `Tail Suspension Test` 大类（或等效变体）
> 行为学同事维护的领域知识: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/tst.md`

## 可用指标函数（在 `ethoinsight.metrics.tst`）

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_immobility_time_tst(df)` | DataFrame | `float \| None` | 累计不动时间（秒；运行长度编码 Activity_State==0 的总帧数 × dt） |
| `compute_immobility_latency_tst(df)` | DataFrame | `float \| None` | 不动潜伏期：首次不动帧的 trial_time（秒；无 trial_time 时退化为帧索引） |
| `compute_immobility_bout_count_tst(df)` | DataFrame | `int \| None` | 不动次数：RLE 检测 Activity_State 列中连续 0 段的个数 |

通用指标（任范式可用）:
- `ethoinsight.metrics._common.compute_distance_moved(df) -> float | None`
- `ethoinsight.metrics._common.compute_velocity_stats(df) -> dict`

## EthoVision 列名约定

TST 使用 EthoVision 内置的 Activity 分析模块，典型列名：

| 含义 | 列名 | 编码 |
|---|---|---|
| 活动状态 | `Activity_State` | 1 = 活动（active），0 = 不动（immobile） |
| 时间轴 | `trial_time` | 秒；缺失时 latency/time 退化为帧数 |

**与 FST 的区别**：TST 使用 `Activity_State`（活动分析模块），FST 使用 `Mobility_State`（运动分析模块）。两者编码相同（1=动，0=不动），算法完全一致。

## RLE 不动次数算法说明

连续 0 序列视为一次不动 bout，两段 0 之间哪怕只有一个 1 也算独立的 bout：

```
Activity_State: [0,0,0, 1,1, 0,0, 1, 0,0,0]
                 bout1        bout2    bout3
→ bout_count = 3
```

这与直接计 0 的帧数不同（总不动帧 = 8，不动次数 = 3）。

## 派发器（一次性算全套）

```python
from ethoinsight.metrics import compute_paradigm_metrics

result = compute_paradigm_metrics(parsed_data, paradigm="tail_suspension", groups=groups)
# result = {
#   "paradigm": "tail_suspension",
#   "per_subject": {subject_name: {immobility_time, immobility_latency, immobility_bout_count, ...}},
#   "group_summary": {group_name: {metric: {mean, std, n, values}}},
#   "data_quality_warnings": [...],  # n<5/组 自动警告
#   ...
# }
```

## 胶水脚本范例（end-to-end）

文件名建议: `${workspace_path}/analysis.py`

```python
"""TST (悬尾测试) 分析胶水脚本。

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

# 1. 解析
parsed_data = parse.parse_trajectories([str(f) for f in RAW_FILES])

# 2. 算指标
metrics_result = compute_paradigm_metrics(parsed_data, paradigm="tail_suspension", groups=GROUPS)

# 3. 统计
stats_result = statistics.run_groupwise(
    metrics_result["per_subject"],
    groups=GROUPS,
    metrics=["immobility_time", "immobility_latency", "immobility_bout_count"],
)

# 4. 出图
chart_files = []
for metric_name in ["immobility_time", "immobility_latency"]:
    fig_path = WORKSPACE / "outputs" / f"tst_{metric_name}_boxplot.png"
    charts.box_plot(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

fig_path = WORKSPACE / "outputs" / "tst_immobility_bout_count_bar.png"
charts.bar_chart(
    metrics_result["per_subject"],
    groups=GROUPS,
    metric="immobility_bout_count",
    output_path=str(fig_path),
)
chart_files.append(str(fig_path))

# 5. 写 handoff
handoff = {
    "paradigm": "tail_suspension",
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

这些警告会自动进 `metrics_result["data_quality_warnings"]`，不需要胶水脚本额外加。

**注意**（领域知识，非自动检测）：
- TST 通常持续 6 分钟。部分研究使用前 4 分钟（习惯化期被剔除），需在胶水脚本中按 `trial_time` 切片。
- 悬尾测试中，小鼠尾巴自我攀爬（tail climbing）会被误判为不动时间减少。若 `immobility_time` 异常低，需结合视频人工核实。
- Activity_State 阈值须经行为学同事确认，阈值设置不当会混淆挣扎帧和漂浮帧。

## 解读原则

- **不动时间增加 + 潜伏期缩短** → 抑郁样行为（behavioral despair）增强
- **不动次数增加但每次持续短** → 间歇性挣扎，提示动物仍有应对尝试
- **组间比较优先于绝对值判断** — 参见 CLAUDE.md 第 9 条判读哲学
- TST 与 FST 常配合使用：两个范式同向变化时，抑郁表型解读更可靠

## 出图建议（详见 ethoinsight-charts skill）

- `immobility_time` / `immobility_latency`: `box_plot` 或 `raincloud_plot`（连续变量 + 组间比较）
- `immobility_bout_count`: `bar_chart` 或 `box_plot`（计数变量）

## handoff JSON 必须字段

见 `../templates/output-contract.md`。TST 特别需要 `data_quality_warnings` 字段（dispatcher 已自动填）。
