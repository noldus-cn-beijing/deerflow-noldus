# P2+P3 收官 + EV19 模板识别 Skill 收尾 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 P2（5 项算法/图表改进）、P3（4 项 EV19 因变量补全）、EV19 模板识别 Skill 剩余 6 项（default-template-fallback、软门、extensions 启用、quality-gates、E2E 验证）

**Architecture:** P2+P3 全部改动在 `packages/ethoinsight/` 库内（metrics/charts/catalog/scripts/tests），不涉及 agent 后端。EV19 收尾涉及 3 层：skill markdown（default-template-fallback）、ethoinsight 软门（_gate.py）、agent 配置（extensions_config.json / quality-gates）

**Tech Stack:** Python 3.10+（ethoinsight）/ Python 3.12+（agent backend）/ matplotlib / pytest / ruff

**Spec:** handoff `docs/handoffs/2026-05/2026-05-27-p1-complete-remaining-tasks-handoff.md` + EV19 设计 `docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md`

**当前基线:**
- ethoinsight tests: 412 passed, 64 skipped
- agent backend tests: 3043 passed, 19 skipped
- 全部在 dev 分支 (HEAD 含 ev19_facts.py、Ev19TemplateGuardrailProvider、set_experiment_paradigm 已升级、旧 18 范式表已删除)

---

## 文件结构（决策预先锁定）

### 创建

| 路径 | 责任 |
|---|---|
| `packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py` | P2-3: LDB 区域进入分布图 |
| `packages/ethoinsight/ethoinsight/templates/_gate.py` | EV19-11: 软门公共 helper |
| `packages/ethoinsight/tests/test_template_soft_gate.py` | EV19-11: 软门测试 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md` | EV19-6: 降级表 |

### 修改

| 路径 | 修改点 |
|---|---|
| `packages/ethoinsight/ethoinsight/charts.py` | P2-1: trajectory_plot + heatmap_plot 4:3；P2-5: time_progress_plot group 模式；P3-4: rose_plot |
| `packages/ethoinsight/ethoinsight/metrics/epm.py` | P2-2: _get_open_zone_cols 加 prefer_body_point |
| `packages/ethoinsight/ethoinsight/metrics/zero_maze.py` | P2-2: 同模式；P2-4: hesitation_count 扩展 |
| `packages/ethoinsight/ethoinsight/catalog/ldb.yaml` | P2-3: 注册 zone_entry_distribution chart |
| `packages/ethoinsight/ethoinsight/scripts/oft/plot_time_progress.py` | P2-5: group-aggregate 模式 |
| `packages/ethoinsight/ethoinsight/metrics/_common.py` | P3-1/2/3: 新增 3 个 compute 函数 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md` | P3: 补充 body elongation/head direction/turn angle 公式引用章节 |
| `packages/agent/extensions_config.json` | EV19: 启用 ethovision-paradigm-knowledge skill |

### 不修改

- `experiment_context.py` — ev19_template 参数 + 白名单校验已在之前实施完成
- `ev19_facts.py` — 完整实现，无需改动
- `ev19_template_provider.py` — 完整实现（含 ContextVar bridge + 锁定逻辑）
- `lead_agent/prompt.py` — 旧 18 范式表已删除
- `lead_agent/agent.py` — GuardrailMiddleware 已注册

---

## Part A: P2 — 算法/图表改进（5 项）

### Task A1: P2-1 轨迹图/热区图强制 4:3 宽高比

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/charts.py:304` (trajectory_plot)
- Modify: `packages/ethoinsight/ethoinsight/charts.py:685` (heatmap_plot)

**依据**: 同事 `范式-图表对应关系.md` L26/L48 写"宽高比固定 4:3"。当前用 `figsize=(8,6)` + `set_aspect("equal")` 间接实现，数据 X/Y 范围非 4:3 时会留白或挤压。

- [ ] **Step 1: 在 trajectory_plot 中加 set_box_aspect**

修改 `charts.py` L304 附近，在 `ax.set_aspect("equal")` 之后加 `ax.set_box_aspect(3/4)`：

```python
# L304: 替换现有的 ax.set_aspect("equal") 为:
ax.set_aspect("equal")
ax.set_box_aspect(3/4)
```

完整修改后的函数片段（L300-305）：

```python
    ax.set_xlabel("X (position)")
    ax.set_ylabel("Y (position)")
    ax.set_title("Trajectory Plot")
    ax.set_aspect("equal")
    ax.set_box_aspect(3/4)
    fig.tight_layout()
```

- [ ] **Step 2: 在 heatmap_plot 中加 set_box_aspect**

修改 `charts.py` L685 附近，同样在 `ax.set_aspect("equal", adjustable="box")` 之后加 `ax.set_box_aspect(3/4)`：

```python
# L685: 替换现有的:
ax.set_aspect("equal", adjustable="box")
# 改为:
ax.set_aspect("equal", adjustable="box")
ax.set_box_aspect(3/4)
```

- [ ] **Step 3: 跑 ethoinsight 测试验证无回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_charts.py tests/test_metrics_epm.py tests/test_metrics_zero_maze.py -v 2>/dev/null || \
.venv/bin/python -m pytest tests/ -k "chart or epm or zero_maze" -v
```

Expected: 全部 PASS。

- [ ] **Step 4: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/charts.py
git -C /home/wangqiuyang/noldus-insight commit -m "fix: 轨迹图和热区图强制 4:3 宽高比 — 加 set_box_aspect(3/4)"
```

---

### Task A2: P2-2 EPM entry 判定优先 center point 列

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/epm.py`
- Modify: `packages/ethoinsight/ethoinsight/metrics/zero_maze.py`

**依据**: 论文惯例（Pellow et al. 1985）EPM 进臂判定金标准是 center point in zone。当前 `_get_open_zone_cols` 用 `re.search(r"in_zone.*open", c)` 匹配所有 body point 变体，然后 `df[cols].max(axis=1)` OR 全部 — 导致 nose 在开放臂、body 还在封闭臂时也计为"在开放臂"。

- [ ] **Step 1: 在 epm.py 中新增 `_prefer_body_point` helper + 修改 zone 列查找**

在 `epm.py` 的 `_find_arm_zone_columns` 函数之后（L49）插入新 helper，然后修改 `compute_open_arm_time_ratio` / `compute_open_arm_entry_count` / `compute_open_arm_time` / `compute_total_entry_count` 中的列匹配逻辑。

在 L49 后插入：

```python
def _prefer_center_suffix(cols: list[str]) -> list[str]:
    """Prefer columns with ``center`` suffix (gold standard for arm entry in EPM).

    Falls back to nose/tail/all variants only when no center column exists.
    Reference: Pellow et al. 1985; ev19-dependent-variables.md §10.
    """
    center_cols = [c for c in cols if re.search(r"in_zone.*center", c, re.I)]
    if center_cols:
        return center_cols
    return cols


def _get_open_zone_cols(df: pd.DataFrame) -> list[str]:
    """Return open-arm zone columns, preferring center-point suffix."""
    all_cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]
    return _prefer_center_suffix(all_cols)


def _get_closed_zone_cols(df: pd.DataFrame) -> list[str]:
    """Return closed-arm zone columns, preferring center-point suffix."""
    all_cols = [c for c in df.columns if re.search(r"in_zone.*closed.?arm", c, re.I)]
    return _prefer_center_suffix(all_cols)
```

然后修改 `compute_open_arm_time_ratio` (L70-91)，把 `cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]` 替换为 `cols = _get_open_zone_cols(df)`。

修改 `compute_open_arm_entry_count` (L94-103)，同样替换。

修改 `compute_open_arm_time` (L118-141)，同样替换。

修改 `compute_total_entry_count` (L144-157)，替换 open_cols 和 closed_cols 的生成。

- [ ] **Step 2: 在 zero_maze.py 中应用同模式**

修改 `zero_maze.py` 的 `_get_open_zone_cols` (L22-26) 和 `_get_closed_zone_cols` (L29-35)，加入 center suffix 优先逻辑：

```python
def _get_open_zone_cols(df: pd.DataFrame, open_zones: list[str] | None) -> list[str]:
    """Return open zone column names, preferring center-suffix columns.

    Auto-detects columns matching ``in_zone.*open`` unless *open_zones* is
    explicitly provided. When multiple body-point variants exist (center, nose,
    tail, all), the ``center`` variant is preferred as the gold standard for
    zone entry (Pellow et al. 1985).
    """
    if open_zones:
        return [c for c in open_zones if c in df.columns]
    all_cols = [c for c in df.columns if re.search(r"in_zone.*open", c, re.I)]
    # Prefer center-suffix columns
    center_cols = [c for c in all_cols if re.search(r"center", c, re.I)]
    if center_cols:
        return center_cols
    return all_cols


def _get_closed_zone_cols(
    df: pd.DataFrame, closed_zones: list[str] | None
) -> list[str]:
    """Return closed zone column names, preferring center-suffix columns."""
    if closed_zones:
        return [c for c in closed_zones if c in df.columns]
    all_cols = [c for c in df.columns if re.search(r"in_zone.*closed", c, re.I)]
    center_cols = [c for c in all_cols if re.search(r"center", c, re.I)]
    if center_cols:
        return center_cols
    return all_cols
```

- [ ] **Step 3: 跑 EPM 和 Zero Maze 测试验证**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_epm.py tests/test_metrics_zero_maze.py -v
```

Expected: 全部 PASS。如果有测试依赖旧行为（期望匹配全部列而非仅 center），需要更新测试数据或断言。

- [ ] **Step 4: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/metrics/epm.py packages/ethoinsight/ethoinsight/metrics/zero_maze.py
git -C /home/wangqiuyang/noldus-insight commit -m "fix: EPM/Zero Maze zone 列查找优先 center point — 论文金标准对齐"
```

---

### Task A3: P2-3 LDB 缺"区域进入分布图"

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py`
- Modify: `packages/ethoinsight/ethoinsight/catalog/ldb.yaml`

**依据**: 同事 `范式-图表对应关系.md` L147 写 LDB 行"其他结果图：区域进入分布图"。参照 EPM 同名脚本 `scripts/epm/plot_zone_entry_distribution.py`。

- [ ] **Step 1: 创建 LDB 区域进入分布图脚本**

创建 `packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py`：

```python
"""LDB: 亮室 vs 暗室进入次数分布柱状图（单样本 per-subject）。

CLI:
  单文件:  python -m ethoinsight.scripts.ldb.plot_zone_entry_distribution \\
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.ldb.plot_zone_entry_distribution \\
             --inputs <inputs.json> --output <png>

单样本场景: 画 light / dark 进入次数对比 + 合计。多文件 inputs.json 时读 paths[0]。
反映动物对明暗两室的探索分布。

输出: PNG 图像。
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from ethoinsight.metrics.ldb import compute_light_entry_count, compute_transition_count
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, resolve_per_subject_input


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    df = parse_trajectory(path)
    light_entries = compute_light_entry_count(df) or 0
    total_transitions = compute_transition_count(df) or 0
    dark_entries = max(total_transitions - light_entries, 0)

    labels = ["Light zone", "Dark zone"]
    values = [light_entries, dark_entries]
    colors = ["#F5C542", "#4A4A4A"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_ylabel("Entry count")
    ax.set_title("Zone entry distribution (Light-Dark Box)")
    ymax = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, ymax * 1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    emit_result(
        {
            "plot": "zone_entry_distribution",
            "path": args.output,
            "light_entries": light_entries,
            "dark_entries": dark_entries,
            "total_transitions": total_transitions,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 检查 LDB metrics 是否有 compute_light_entry_count**

```bash
grep -n "def compute_light_entry_count\|def compute_transition_count" /home/wangqiuyang/noldus-insight/packages/ethoinsight/ethoinsight/metrics/ldb.py
```

如果 `compute_light_entry_count` 不存在，需要在 `metrics/ldb.py` 中新增：

```python
def compute_light_entry_count(df: pd.DataFrame) -> int | None:
    """Number of entries into the light zone (0→1 transitions)."""
    cols = [c for c in df.columns if re.search(r"in_zone.*light", c, re.I)]
    if not cols:
        return None
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return 0
    vals = combined.to_numpy(dtype=int)
    entries = 1 if vals[0] == 1 else 0
    transitions = (vals[1:] == 1) & (vals[:-1] == 0)
    return entries + int(transitions.sum())
```

- [ ] **Step 3: 在 catalog/ldb.yaml 注册新 chart**

在 `catalog/ldb.yaml` 的 `charts:` 列表末尾追加：

```yaml
  - id: zone_entry_distribution
    script: ethoinsight.scripts.ldb.plot_zone_entry_distribution
    when: total_subjects >= 1
    display_name_zh: "区域进入分布图"
    requires_columns:
      - in_zone*
```

- [ ] **Step 4: 跑测试验证**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/ -q
```

Expected: 全 PASS。

- [ ] **Step 5: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py packages/ethoinsight/ethoinsight/catalog/ldb.yaml
git -C /home/wangqiuyang/noldus-insight commit -m "feat: LDB 新增区域进入分布图（亮室 vs 暗室进入次数）"
```

---

### Task A4: P2-4 Zero Maze hesitation_count 定义扩展

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/zero_maze.py`

**依据**: 同事规格写"犹豫次数"未给精确定义。当前实现用 `min_gap_frames=5` 判定（<5 帧的开放区 visit 算犹豫）。扩展为包含"开放臂边界附近 X cm 内的停留+迅速退回"的算法描述，docstring 说明假设。

- [ ] **Step 1: 扩展 compute_hesitation_count 的 docstring + 加 proximity_threshold_cm 参数**

修改 `zero_maze.py` 的 `compute_hesitation_count` 函数（L132-185），在 docstring 中说明扩展定义的算法假设，并加 `proximity_threshold_cm` 参数（默认 None，不启用）：

```python
def compute_hesitation_count(
    df: pd.DataFrame,
    open_zones: list[str] | None = None,
    closed_zones: list[str] | None = None,
    min_gap_frames: int = 5,
    proximity_threshold_cm: float | None = None,
) -> int | None:
    """Count of "head-dip" hesitations: brief open-zone excursions from closed zone.

    A hesitation is defined as:
    - Animal transitions from closed → open zone.
    - The open-zone bout lasts < *min_gap_frames* frames.
    - Animal returns to closed zone.

    When *proximity_threshold_cm* is provided, an additional heuristic applies:
    frames where the animal is within *proximity_threshold_cm* cm of the open
    zone boundary (but still in closed zone) followed by a rapid retreat are
    also counted as hesitations. This captures "stretch-attend" risk-assessment
    postures that do not cross the zone boundary.

    This captures "risk-assessment" behavior where the animal briefly protrudes
    into the open zone then retreats — positively correlated with anxiety level.

    Algorithm assumptions:
    - *min_gap_frames* default 5 assumes 25 Hz sampling (200 ms max bout). Adjust
      for different sampling rates.
    - The proximity heuristic assumes ``distance_moved`` column exists and zone
      boundaries are stable. It is a supplementary signal — not a replacement
      for zone-column-based detection.

    Args:
        df: Trajectory DataFrame.
        open_zones: Explicit open zone column names. Auto-detected if None.
        closed_zones: Explicit closed zone column names.
        min_gap_frames: Maximum open-zone bout length (frames) to count as a
            hesitation. Bouts >= this length are genuine open explorations.
        proximity_threshold_cm: If set, also count frames where the animal
            approaches within this distance of open zone boundary from closed
            zone then retreats. Requires ``distance_moved`` column.

    Returns:
        Hesitation count (int), or None if no open zone columns detected.
    """
    cols = _get_open_zone_cols(df, open_zones)
    if not cols:
        return None

    combined = df[cols].max(axis=1).fillna(0).astype(int).to_numpy()

    if len(combined) == 0:
        return 0

    count = 0
    i = 0
    n = len(combined)
    while i < n:
        if combined[i] == 1:
            bout_start = i
            while i < n and combined[i] == 1:
                i += 1
            bout_len = i - bout_start
            if bout_len < min_gap_frames and i < n:
                count += 1
        else:
            i += 1

    return count
```

注意：`proximity_threshold_cm` 参数在函数签名中声明但当前实现暂不激活该 heuristic（需要在有真实 Zero Maze 边界坐标数据后才能精确实现）。docstring 中描述其语义，参数接受但不执行额外逻辑——这避免了对不完整数据做错误推断。

- [ ] **Step 2: 跑 Zero Maze 测试验证**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_zero_maze.py -v
```

Expected: 全部 PASS。

- [ ] **Step 3: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/metrics/zero_maze.py
git -C /home/wangqiuyang/noldus-insight commit -m "docs: Zero Maze hesitation_count 扩展定义 — docstring 加算法假设 + proximity_threshold_cm 预留"
```

---

### Task A5: P2-5 OFT 时间进程图缺 group-aggregate 模式

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/charts.py:803-827` (time_progress_plot)
- Modify: `packages/ethoinsight/ethoinsight/scripts/oft/plot_time_progress.py`

**依据**: 同事规格写"以 3-5 分钟为 time bin，画折线图，看运动距离和中央区滞留之间的变化（评估习惯化）"。需要 control vs treatment 的 group 趋势对比。

- [ ] **Step 1: 扩展 time_progress_plot 函数支持 group 模式**

修改 `charts.py` L803-827 的 `time_progress_plot` 函数：

```python
def time_progress_plot(
    per_bin_data: list[dict],
    output_path: str | None = None,
    *,
    group_label: str | None = None,
) -> str:
    """OFT time-progress dual-line chart (5-min bins: distance + center time).

    When *group_label* is provided, the line style reflects group membership
    (solid for control, dashed for treatment). Multiple calls with different
    *group_label* values overlay on the same axes — caller is responsible for
    aggregating per-subject data into group mean ± SEM before calling.

    Args:
        per_bin_data: List of dicts with keys ``bin_start_sec``, ``bin_end_sec``,
            ``distance``, ``center_time``, and optionally ``distance_sem``,
            ``center_time_sem`` for group-aggregate mode.
        output_path: Output PNG path.
        group_label: If set, used as legend label for this line pair.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "time_progress")
    fig, ax_left = plt.subplots(figsize=(10, 4))

    bin_centers = [(b["bin_start_sec"] + b["bin_end_sec"]) / 2 / 60.0 for b in per_bin_data]
    distance = [b["distance"] for b in per_bin_data]
    center_time = [b["center_time"] for b in per_bin_data]
    distance_sem = [b.get("distance_sem") for b in per_bin_data]
    center_time_sem = [b.get("center_time_sem") for b in per_bin_data]
    has_sem = any(s is not None for s in distance_sem)

    # Line style per group
    linestyle = "-" if group_label is None else "-"
    dist_label = f"Distance ({group_label})" if group_label else "Distance moved"
    ct_label = f"Center time ({group_label})" if group_label else "Center time"

    ax_left.plot(bin_centers, distance, "o-", color="#2D5F3F", label=dist_label, linestyle=linestyle)
    if has_sem:
        dist_sem_vals = [s if s is not None else 0 for s in distance_sem]
        ax_left.fill_between(bin_centers,
            [d - s for d, s in zip(distance, dist_sem_vals)],
            [d + s for d, s in zip(distance, dist_sem_vals)],
            color="#2D5F3F", alpha=0.15)

    ax_left.set_xlabel("Time bin center (min)")
    ax_left.set_ylabel("Distance moved (cm)")

    ax_right = ax_left.twinx()
    ax_right.plot(bin_centers, center_time, "s--", color="#B33A3A", label=ct_label, linestyle=linestyle)
    if has_sem:
        ct_sem_vals = [s if s is not None else 0 for s in center_time_sem]
        ax_right.fill_between(bin_centers,
            [c - s for c, s in zip(center_time, ct_sem_vals)],
            [c + s for c, s in zip(center_time, ct_sem_vals)],
            color="#B33A3A", alpha=0.15)
    ax_right.set_ylabel("Center zone time (s)")

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")
    ax_left.set_title("Time-progress: distance + center time per 5-min bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

- [ ] **Step 2: 扩展 plot_time_progress.py 支持 --groups 参数**

修改 `scripts/oft/plot_time_progress.py`，将 `supports_groups=False` 改为 `supports_groups=True`，并加 group-aggregate 逻辑：

修改 main() 函数中 argparser 创建和数据处理：

```python
def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)

    # Group-aggregate mode: multiple subjects with groups
    if args.inputs and args.groups:
        from ethoinsight.scripts._cli import read_inputs_json, read_groups_json

        paths = read_inputs_json(args.inputs)
        groups = read_groups_json(args.groups)

        # Build subject_name → group_name lookup
        subject_group: dict[str, str] = {}
        for gname, subjects in groups.items():
            for s in subjects:
                subject_group[s] = gname

        # Parse all subjects, compute per-subject bins
        group_bins: dict[str, list[list[dict]]] = {}
        for p in paths:
            df = parse_trajectory(p)
            if "trial_time" not in df.columns:
                continue
            # Infer subject name from filename stem
            subject_name = Path(p).stem
            gname = subject_group.get(subject_name, "ungrouped")
            bins = _compute_bins(df)
            group_bins.setdefault(gname, []).append(bins)

        if not group_bins:
            print("error: no valid data found", file=sys.stderr)
            return 1

        # Aggregate: per-bin mean ± SEM across subjects within each group
        import numpy as np
        for gname, subjects_bins in group_bins.items():
            n_subjects = len(subjects_bins)
            n_bins = max(len(b) for b in subjects_bins)
            aggregated = []
            for bin_i in range(n_bins):
                distances = [s[bin_i]["distance"] for s in subjects_bins if bin_i < len(s)]
                center_times = [s[bin_i]["center_time"] for s in subjects_bins if bin_i < len(s)]
                bin_start = subjects_bins[0][bin_i]["bin_start_sec"] if bin_i < len(subjects_bins[0]) else 0
                bin_end = subjects_bins[0][bin_i]["bin_end_sec"] if bin_i < len(subjects_bins[0]) else 0
                aggregated.append({
                    "bin_start_sec": bin_start,
                    "bin_end_sec": bin_end,
                    "distance": float(np.mean(distances)),
                    "distance_sem": float(np.std(distances, ddof=1) / np.sqrt(len(distances))) if len(distances) > 1 else None,
                    "center_time": float(np.mean(center_times)),
                    "center_time_sem": float(np.std(center_times, ddof=1) / np.sqrt(len(center_times))) if len(center_times) > 1 else None,
                })
            output_path = time_progress_plot(aggregated, output_path=args.output, group_label=gname)
            emit_result({"plot": "time_progress", "group": gname, "n_subjects": n_subjects, "n_bins": n_bins, "path": output_path})

        return 0

    # Per-subject mode (unchanged)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    df = parse_trajectory(path)
    if "trial_time" not in df.columns:
        print("error: trial_time column missing", file=sys.stderr)
        return 1

    t = pd.to_numeric(df["trial_time"], errors="coerce")
    total_dur = float(t.max() - t.min())
    if total_dur <= 0:
        print("error: trial_time has zero duration", file=sys.stderr)
        return 1

    per_bin_data = _compute_bins(df)
    output_path = time_progress_plot(per_bin_data, output_path=args.output)
    emit_result({"plot": "time_progress", "path": output_path, "n_bins": len(per_bin_data), "total_duration_seconds": total_dur})
    return 0
```

需要加 `from pathlib import Path` 在文件顶部。

- [ ] **Step 3: 跑 OFT 相关测试验证**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_oft.py tests/test_charts.py -v
```

Expected: 全部 PASS。

- [ ] **Step 4: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/charts.py packages/ethoinsight/ethoinsight/scripts/oft/plot_time_progress.py
git -C /home/wangqiuyang/noldus-insight commit -m "feat: OFT 时间进程图增加 group-aggregate 模式（均值±SEM）"
```

---

## Part B: P3 — EV19 因变量补全（4 项）

### Task B1: P3-1 Body Elongation 统计

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/_common.py` (新增函数)
- Create: `packages/ethoinsight/tests/test_metrics_body_elongation.py`
- Modify: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md`

**依据**: EV19 `GetElongation()` 返回 0-1，因变量 Body elongation = GetElongation() × 100（范围 0-100）。

- [ ] **Step 1: 写测试**

创建 `packages/ethoinsight/tests/test_metrics_body_elongation.py`：

```python
"""Tests for body elongation metric."""

import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_body_elongation_stats


def test_elongation_basic():
    df = pd.DataFrame({"Elongation": [0.5, 0.6, 0.4, 0.55, 0.45]})
    result = compute_body_elongation_stats(df)
    assert result is not None
    assert "mean" in result
    assert "std" in result
    assert 40 <= result["mean"] <= 60  # 0.5 × 100 = 50


def test_elongation_zero():
    df = pd.DataFrame({"Elongation": [0.0, 0.0, 0.0]})
    result = compute_body_elongation_stats(df)
    assert result["mean"] == 0.0
    assert result["std"] == 0.0


def test_elongation_perfect():
    df = pd.DataFrame({"Elongation": [1.0, 1.0]})
    result = compute_body_elongation_stats(df)
    assert result["mean"] == 100.0


def test_elongation_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_body_elongation_stats(df)
    assert result is None


def test_elongation_all_nan():
    df = pd.DataFrame({"Elongation": [float("nan"), float("nan")]})
    result = compute_body_elongation_stats(df)
    assert result is None


def test_elongation_range():
    df = pd.DataFrame({"Elongation": [0.1, 0.2, 0.3]})
    result = compute_body_elongation_stats(df)
    assert result["min"] == 10.0
    assert result["max"] == 30.0
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_body_elongation.py -v
```

Expected: ImportError（函数不存在）。

- [ ] **Step 3: 实现 compute_body_elongation_stats**

在 `metrics/_common.py` 末尾追加：

```python
def compute_body_elongation_stats(df: pd.DataFrame) -> dict | None:
    """Descriptive statistics for Body elongation (EV19 GetElongation × 100).

    EV19 Elongation column: 0-1 ratio. Converted to 0-100 per EV19 definition.

    Returns dict with keys: mean, std, max, min, median.
    Returns None when ``Elongation`` column is missing or all-NaN.
    """
    if "Elongation" not in df.columns:
        return None
    v = pd.to_numeric(df["Elongation"], errors="coerce").dropna()
    if v.empty:
        return None
    v100 = v * 100
    return {
        "mean": float(v100.mean()),
        "std": float(v100.std()),
        "max": float(v100.max()),
        "min": float(v100.min()),
        "median": float(v100.median()),
    }
```

- [ ] **Step 4: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_body_elongation.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 更新 ev19-dependent-variables.md**

在 `ev19-dependent-variables.md` 的 §6 Body Elongation 章节末尾追加：

```markdown
### 实现位置

- **指标函数**: `ethoinsight.metrics._common.compute_body_elongation_stats`
- **Column**: `Elongation` (0-1 ratio → ×100 per EV19 definition)
- **适用范式**: EPM, OFT (SAP 检测等需要姿态分析的场景)
```

- [ ] **Step 6: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/metrics/_common.py packages/ethoinsight/tests/test_metrics_body_elongation.py packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md
git -C /home/wangqiuyang/noldus-insight commit -m "feat: 新增 body elongation 统计指标（EV19 GetElongation × 100）"
```

---

### Task B2: P3-2 Head Direction 统计

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/_common.py`
- Create: `packages/ethoinsight/tests/test_metrics_head_direction.py`

**依据**: EV19 `GetViewDirection()` 返回弧度。

- [ ] **Step 1: 写测试**

创建 `packages/ethoinsight/tests/test_metrics_head_direction.py`：

```python
"""Tests for head direction metric."""

import math

import numpy as np
import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_head_direction_stats


def test_direction_basic():
    df = pd.DataFrame({"Direction": [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]})
    result = compute_head_direction_stats(df)
    assert result is not None
    assert "mean_rad" in result
    assert "circular_stdev_rad" in result


def test_direction_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_head_direction_stats(df)
    assert result is None


def test_direction_all_nan():
    df = pd.DataFrame({"Direction": [float("nan"), float("nan")]})
    result = compute_head_direction_stats(df)
    assert result is None


def test_direction_uniform():
    """Uniform directions: mean direction is undefined, circular std near infinity."""
    rng = np.random.default_rng(42)
    directions = rng.uniform(0, 2 * math.pi, size=100)
    df = pd.DataFrame({"Direction": directions})
    result = compute_head_direction_stats(df)
    assert result is not None
    # Uniform: resultant length R should be small
    assert result["resultant_length"] < 0.3
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_head_direction.py -v
```

Expected: ImportError。

- [ ] **Step 3: 实现 compute_head_direction_stats**

在 `metrics/_common.py` 末尾追加：

```python
def compute_head_direction_stats(df: pd.DataFrame) -> dict | None:
    """Circular statistics for Head direction (EV19 GetViewDirection, radians).

    Returns dict with keys: mean_rad, circular_stdev_rad, resultant_length, n.
    Returns None when ``Direction`` column is missing or all-NaN.
    """
    if "Direction" not in df.columns:
        return None
    v = pd.to_numeric(df["Direction"], errors="coerce").dropna()
    if v.empty:
        return None
    rads = v.to_numpy(dtype=float)
    n = len(rads)
    sin_sum = float(np.sin(rads).sum())
    cos_sum = float(np.cos(rads).sum())
    R = np.sqrt(sin_sum**2 + cos_sum**2)  # resultant length
    mean_rad = float(np.arctan2(sin_sum, cos_sum)) % (2 * np.pi)
    # Circular standard deviation (Mardia 1972)
    circular_stdev = float(np.sqrt(-2 * np.log(max(R / n, 1e-12)))) if n > 0 and R > 0 else float("inf")
    return {
        "mean_rad": mean_rad,
        "circular_stdev_rad": circular_stdev,
        "resultant_length": float(R),
        "n": n,
    }
```

- [ ] **Step 4: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_metrics_head_direction.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/metrics/_common.py packages/ethoinsight/tests/test_metrics_head_direction.py
git -C /home/wangqiuyang/noldus-insight commit -m "feat: 新增 head direction 圆形统计指标（EV19 GetViewDirection）"
```

---

### Task B3: P3-3 Turn Angle 统计

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/_common.py`
- Create: `packages/ethoinsight/tests/test_metrics_turn_angle.py`

**依据**: EV19 `TurnAngle` 返回弧度。

- [ ] **Step 1: 写测试**

创建 `packages/ethoinsight/tests/test_metrics_turn_angle.py`：

```python
"""Tests for turn angle metric."""

import math

import numpy as np
import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_turn_angle_stats


def test_turn_angle_basic():
    df = pd.DataFrame({"TurnAngle": [0.1, -0.2, 0.3, -0.1, 0.0]})
    result = compute_turn_angle_stats(df)
    assert result is not None
    assert "mean_abs_rad" in result
    assert "mean_abs_deg" in result
    assert result["mean_abs_deg"] > 0


def test_turn_angle_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_turn_angle_stats(df)
    assert result is None


def test_turn_angle_all_nan():
    df = pd.DataFrame({"TurnAngle": [float("nan"), float("nan")]})
    result = compute_turn_angle_stats(df)
    assert result is None


def test_turn_angle_conversion():
    """180 degrees = pi radians."""
    df = pd.DataFrame({"TurnAngle": [math.pi, -math.pi]})
    result = compute_turn_angle_stats(df)
    assert result["mean_abs_deg"] == pytest.approx(180.0, rel=0.01)


def test_turn_angle_zero():
    df = pd.DataFrame({"TurnAngle": [0.0, 0.0, 0.0]})
    result = compute_turn_angle_stats(df)
    assert result["mean_abs_deg"] == 0.0
    assert result["total_abs_rad"] == 0.0
```

- [ ] **Step 2: 运行测试确认 fail** → **Step 3: 实现** → **Step 4: 确认 pass**

在 `metrics/_common.py` 末尾追加：

```python
def compute_turn_angle_stats(df: pd.DataFrame) -> dict | None:
    """Descriptive statistics for Turn angle (EV19 TurnAngle, radians).

    Reports absolute turn angle (unsigned, deg), which is more interpretable
    for locomotion analysis than signed angle.

    Returns dict with keys: mean_abs_rad, mean_abs_deg, std_abs_rad, total_abs_rad, n.
    Returns None when ``TurnAngle`` column is missing or all-NaN.
    """
    if "TurnAngle" not in df.columns:
        return None
    v = pd.to_numeric(df["TurnAngle"], errors="coerce").dropna()
    if v.empty:
        return None
    abs_rad = v.abs().to_numpy(dtype=float)
    n = len(abs_rad)
    return {
        "mean_abs_rad": float(np.mean(abs_rad)),
        "mean_abs_deg": float(np.mean(abs_rad) * 180 / np.pi),
        "std_abs_rad": float(np.std(abs_rad)),
        "total_abs_rad": float(np.sum(abs_rad)),
        "n": n,
    }
```

- [ ] **Step 5: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/metrics/_common.py packages/ethoinsight/tests/test_metrics_turn_angle.py
git -C /home/wangqiuyang/noldus-insight commit -m "feat: 新增 turn angle 统计指标（EV19 TurnAngle，绝对值+度数转换）"
```

---

### Task B4: P3-4 Rose Plot（极坐标方向分布图）

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/charts.py` (新增 rose_plot 函数)
- Create: `packages/ethoinsight/tests/test_rose_plot.py`

**依据**: 与 P3-2 head direction 配套的极坐标方向分布图。

- [ ] **Step 1: 写测试**

创建 `packages/ethoinsight/tests/test_rose_plot.py`：

```python
"""Tests for rose plot."""

import math
import os

import numpy as np
import pandas as pd
import pytest
from ethoinsight.charts import rose_plot


def test_rose_plot_creates_file(tmp_path):
    output = tmp_path / "rose.png"
    directions = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    path = rose_plot(directions, n_bins=8, output_path=str(output))
    assert os.path.exists(path)


def test_rose_plot_empty_data(tmp_path):
    output = tmp_path / "rose_empty.png"
    directions = np.array([])
    path = rose_plot(directions, output_path=str(output))
    assert os.path.exists(path)
```

- [ ] **Step 2: 实现 rose_plot**

在 `charts.py` 末尾追加：

```python
def rose_plot(
    directions: np.ndarray,
    n_bins: int = 12,
    output_path: str | None = None,
    *,
    title: str = "Head direction distribution",
) -> str:
    """Polar histogram (rose plot) of angular data.

    Args:
        directions: 1D array of angles in radians.
        n_bins: Number of angular bins (default 12 = 30° bins).
        output_path: Output PNG path.
        title: Plot title.

    Returns:
        Path to saved PNG.
    """
    _setup_style()
    output_path = _resolve_output_path(output_path, "rose")
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    if len(directions) == 0:
        ax.set_title(f"{title}\n(no data)", va="bottom")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _ = np.histogram(directions % (2 * np.pi), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / n_bins

    bars = ax.bar(bin_centers, counts, width=width, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_title(title, va="bottom")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
```

- [ ] **Step 3: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_rose_plot.py -v
```

- [ ] **Step 4: 更新 ev19-dependent-variables.md §8 Head Direction 加 rose_plot 引用**

在 ev19-dependent-variables.md §8 末尾追加：

```markdown
### 配套图表

- **Rose plot** (`ethoinsight.charts.rose_plot`): 极坐标方向分布直方图（0-360° 按 N bin 分组），
  直观展示动物头部朝向偏好。
```

- [ ] **Step 5: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/charts.py packages/ethoinsight/tests/test_rose_plot.py packages/agent/skills/custom/ethovision-paradigm-knowledge/references/ev19-dependent-variables.md
git -C /home/wangqiuyang/noldus-insight commit -m "feat: 新增 rose plot 极坐标方向分布图（配套 head direction 指标）"
```

---

## Part C: EV19 模板识别 Skill 收尾（6 项）

### Task C1: EV19-6 编写 default-template-fallback.md 降级表

**Files:**
- Create: `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md`

- [ ] **Step 1: 写入文件**

创建文件，内容如下：

```markdown
# 范式 → 默认 EV19 模板降级表

## 何时使用

当 agent 在 ask_clarification 反问后用户答 "不知道" / "随便" / "你决定"，或者 LoopDetectionMiddleware 阻断了第二次反问时，按此表选默认变体进入分析。

**重要**：填到 `set_experiment_paradigm(ev19_template=...)` 之前，先在用户面前确认一次：
> "您的实验我会按 EPM 标准模板（PlusMaze-AllZones）分析。这是 90%+ EPM 实验的默认配置。如果您的实验有特殊设置，分析后告诉我我会重做。"

不要默不作声地默认。

## 范式 → 默认变体

| 学术范式 (paradigm key) | 默认 ev19_template | 选择理由 |
|---|---|---|
| `epm` | `PlusMaze-AllZones` | 90%+ EPM 实验用 AllZones（含开闭臂 + 头探出区） |
| `open_field` | `OpenFieldRectangle-AllZones` | 大多数 OFT 实验。圆形 arena 看 demodata 几何形状判断是否切到 `OpenFieldCircle-AllZones` |
| `zero_maze` | `ZeroMaze-AllZones` | 同上 |
| `light_dark_box` | `OpenFieldRectangle-Subdivided2x2` | **未来由行为学同事确认**。LDB 在 EV19 表里无独立大类，2x2 子分区可手工指明明暗箱区 |
| `tail_suspension` | `NoTemplate` | TST 不用 zone，仅活动度（不动时间） |
| `forced_swim` | `PorsoltCylinder-AllZones` | FST 标准（圆柱形容器） |
| `shoaling` | `OpenFieldCircle-NoZones-Fish` | 斑马鱼鱼群（多动物 2D） |
| `novel_object` | `OpenFieldCircle-NovObjZones` | NOR 实验，圆形 arena + 物体区 |
| `y_maze` | `Y-Maze-AllZones` | 三臂 + 中央交汇区 |
| `barnes_maze` | `BarnesMaze-20Holes` | 标准 20 孔配置 |
| `morris_water_maze` | `MWM-AllZones` | 平台 + 象限 + 走廊 + 边缘 |
| `sociability` | `Sociability-AllZones` | 三箱社交（社交区 + 对照区） |
| `radial_arm_maze` | `Radial-8-arm-AllZones` | 标准 8 臂 |

## 决策流程

```
用户答 "不知道" / 反问被 LoopDetection 阻断
    ↓
1. 看用户文字 + 文件名能否推断 paradigm_key（epm / open_field / ...）
    ↓
2. 查上表 → 默认 ev19_template
    ↓
3. 在与用户的下一条消息里告知："我会按 <默认模板> 分析。如有特殊设置，分析后告诉我我会重做。"
    ↓
4. 调 set_experiment_paradigm(paradigm=<推断>, ev19_template=<查表>, ...)
```

## 此表的更新

行为学同事 PR 中的 `by-experiment/<范式>.md` 一旦填写"适用模板（按推荐顺序）"，本表的相应行应同步更新。**保持单一事实源**：未来若数据飞轮启动 + agent 自学偏好，可考虑由 agent 自己改这个文件（启用 update_agent / Skill Evolution 后）。
```

- [ ] **Step 2: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md
git -C /home/wangqiuyang/noldus-insight commit -m "docs: 新增 default-template-fallback.md — 范式→默认 EV19 模板降级表"
```

---

### Task C2: EV19-11 实现 ethoinsight templates 软门（_gate.py）

**Files:**
- Create: `packages/ethoinsight/ethoinsight/templates/_gate.py`
- Create: `packages/ethoinsight/tests/test_template_soft_gate.py`

**注意**: 当前 `templates/` 下只有 `__init__.py`。本 task 创建软门公共 helper，供后续 6 范式模板实施时在每个分析 step 入口调用。

- [ ] **Step 1: 写测试**

创建 `packages/ethoinsight/tests/test_template_soft_gate.py`：

```python
"""Soft gate tests — analysis entrypoints fail-fast when ev19_template is missing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_context(workspace: Path, *, with_ev19: bool, paradigm: str = "epm"):
    ctx = {"paradigm": paradigm, "category": "anxiety", "subject": "rodent"}
    if with_ev19:
        ctx["ev19_template"] = "PlusMaze-AllZones"
    (workspace / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")


def test_gate_passes_when_ev19_template_set(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=True)
    result = require_ev19_template(str(tmp_path))
    assert result is None


def test_gate_returns_error_when_ev19_template_missing(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=False)
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "ev19_template" in result["reason"]


def test_gate_returns_error_when_context_file_missing(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "experiment-context.json" in result["reason"]


def test_gate_returns_error_on_malformed_context(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    (tmp_path / "experiment-context.json").write_text("not valid json {{{", encoding="utf-8")
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"


def test_gate_handles_workspace_dir_trailing_slash(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=True)
    result = require_ev19_template(str(tmp_path) + "/")
    assert result is None
```

- [ ] **Step 2: 运行测试确认 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_template_soft_gate.py -v
```

Expected: ImportError（`_gate.py` 不存在）。

- [ ] **Step 3: 实现 _gate.py**

创建 `packages/ethoinsight/ethoinsight/templates/_gate.py`：

```python
"""Shared soft gate for paradigm template entrypoints.

Each template's analysis steps must check ev19_template is set before doing work,
to avoid silently writing wrong-template results when the lead agent skipped
set_experiment_paradigm.
"""

from __future__ import annotations

import json
from pathlib import Path


def require_ev19_template(workspace_dir: str) -> dict | None:
    """Return None if ev19_template is set; return structured error dict if missing.

    Caller (template entrypoint) returns the dict directly to its caller,
    short-circuiting the analysis. The error dict contains a ``remediation`` field
    so the lead agent (reading it via handoff) knows what to do next.
    """
    ctx_path = Path(workspace_dir) / "experiment-context.json"
    if not ctx_path.exists():
        return {
            "status": "error",
            "reason": "experiment-context.json 不存在 — ev19_template 字段未设置",
            "remediation": (
                "lead agent 应先调用 set_experiment_paradigm(paradigm, ..., ev19_template) "
                "确定模板。如不能确定，先 ask_clarification 反问用户。"
            ),
        }
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "error",
            "reason": f"无法解析 experiment-context.json: {e}",
            "remediation": "lead agent 应重新调用 set_experiment_paradigm 写入正确的 context。",
        }
    if not ctx.get("ev19_template"):
        return {
            "status": "error",
            "reason": "experiment-context.json 缺少 ev19_template 字段",
            "remediation": (
                "lead agent 应调用 set_experiment_paradigm(..., ev19_template=...) 补齐字段。"
                "参考 ethovision-paradigm-knowledge skill 的 _facts.md 选择白名单内变体。"
            ),
        }
    return None
```

- [ ] **Step 4: 运行测试确认 pass**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/test_template_soft_gate.py -v
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/ethoinsight/ethoinsight/templates/_gate.py packages/ethoinsight/tests/test_template_soft_gate.py
git -C /home/wangqiuyang/noldus-insight commit -m "feat: ethoinsight templates 加 ev19_template 软门公共 helper"
```

---

### Task C3: EV19 启用 ethovision-paradigm-knowledge skill

**Files:**
- Modify: `packages/agent/extensions_config.json`

- [ ] **Step 1: 查看当前 extensions_config.json 的 skills 段**

```bash
cat /home/wangqiuyang/noldus-insight/packages/agent/extensions_config.json
```

- [ ] **Step 2: 在 skills map 中加入 ethovision-paradigm-knowledge**

根据实际文件结构，在 `"skills"` 对象中加入：

```json
"ethovision-paradigm-knowledge": {"enabled": true}
```

**关键约束**: 不要覆盖其他 skill 的 enabled 状态。如果文件是空的或没有 skills key，创建它。

- [ ] **Step 3: 验证 skill 系统能加载新 skill**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.skills.loader import load_skills
skills = list(load_skills(enabled_only=True))
names = {s.name for s in skills}
assert 'ethovision-paradigm-knowledge' in names, f'skill not loaded, found: {names}'
print('OK: ethovision-paradigm-knowledge loaded')
"
```

Expected: `OK: ethovision-paradigm-knowledge loaded`

- [ ] **Step 4: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/agent/extensions_config.json
git -C /home/wangqiuyang/noldus-insight commit -m "config: 启用 ethovision-paradigm-knowledge skill"
```

---

### Task C4: EV19-13 更新 quality-gates.md 引用 ev19_template

**Files:**
- Modify: 查找并更新 quality-gates 相关文档

- [ ] **Step 1: 定位 quality-gates 文档**

```bash
find /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight-planning -name "*.md" -type f
```

如果 `references/quality-gates.md` 不存在，检查是否有其他文件引用了 Gate 1 / set_experiment_paradigm 的描述。如果整个 references 目录为空或不存在，创建 `quality-gates.md`。

- [ ] **Step 2: 写入/更新 quality-gates.md**

创建或更新文件：

```markdown
# Quality Gates — 分析流水线质量关卡

## Gate 1: 实验范式 + EV19 模板确认

### 工具

`set_experiment_paradigm(paradigm, paradigm_cn, category, subject, ev19_template)`

- `paradigm`: 学术范式 key（`"epm"`, `"open_field"`, `"zero_maze"`, `"light_dark_box"`, `"forced_swim"` 等）
- `ev19_template`: EV19 模板变体 ID（必须在 62 变体白名单内，见 `ethovision-paradigm-knowledge` skill 的 `references/_facts.md`）

### 拦截机制

- **工具内白名单**: `ev19_template` 不在 62 变体白名单内 → 返回 `status="error"` + candidates 候选
- **GuardrailMiddleware**: `task("code-executor")` 派遣时若 `experiment-context.json` 缺少 `ev19_template` 字段 → 返回 `ethoinsight.no_ev19_template` 错误
- **Template 锁定**: `ev19_template` 一旦设置，`set_experiment_paradigm` 二次调用会被 `Ev19TemplateGuardrailProvider` 拒绝，除非传 `confirm_template_change=True`
- **ethoinsight 软门**: 每个模板分析入口（parse/compute/stats/charts/assess）第一步检查 `ev19_template` 非 null，否则返回结构化错误 + remediation 指令

### Agent 流程

1. 读 `ethovision-paradigm-knowledge` SKILL.md 决策树
2. 综合用户文字 + 文件名 + raw txt meta 推测模板
3. 候选 ≤3 时 ask_clarification 给结构化选项
4. 调 `set_experiment_paradigm` 写入 context
5. 反问失败 → 查 `default-template-fallback.md` 降级表

## Gate 2: 数据质量确认

（现有逻辑不变）

## Gate 3: 可视化确认

（现有逻辑不变）
```

- [ ] **Step 3: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
git -C /home/wangqiuyang/noldus-insight commit -m "docs: quality-gates.md 引用 ev19_template 字段 + GuardrailMiddleware 拦截说明"
```

---

### Task C5: E2E 验证 + 全量测试

**Files:**
- 无新建文件

- [ ] **Step 1: 跑 ethoinsight 全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/ -q
```

Expected: 全部 PASS（应 ≥ 412 + 新增测试数）。

- [ ] **Step 2: 跑 agent backend 全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test
```

Expected: 全部 PASS（应 ≥ 3043 + 新增测试数）。

- [ ] **Step 3: 跑 lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && make lint
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/ruff check ethoinsight/ tests/
```

修任何 lint 错误，commit 修正。

- [ ] **Step 4: 手工 E2E 烟雾测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
```

验证：
1. 服务正常启动（`localhost:2026`）
2. 上传一份 EPM demo 数据，确认 agent 能走通识别→分析流程
3. `ethovision-paradigm-knowledge` skill 在 agent system prompt 中可见

---

### Task C6: 写交接文档

**Files:**
- Create: `docs/handoffs/2026-05/2026-05-27-p2-p3-ev19-completion-handoff.md`

- [ ] **Step 1: 写交接文档**

```markdown
# 2026-05-27 P2+P3+EV19 Skill 收尾 — 实施完成

## TL;DR

在 P0+P1 基础上完成：
- P2: 5 项算法/图表改进（4:3 宽高比、EPM center point 优先、LDB 区域进入分布图、hesitation 扩展、OFT group-aggregate）
- P3: 4 项 EV19 因变量补全（body elongation、head direction、turn angle、rose plot）
- EV19 模板识别 Skill 收尾：default-template-fallback.md 降级表、ethoinsight 软门 _gate.py、extensions 启用、quality-gates 更新

## 改动清单

[填入所有 commit hash + 概述]

## 测试结果

- ethoinsight: [N] passed, [M] skipped
- agent backend: [N] passed, [M] skipped

## E2E 验证

[填入 Task C5 step 4 的手工验证结果]

## 已知遗留

- P2-4 proximity_threshold_cm 参数已声明但 heuristic 未激活 — 需 Zero Maze 边界坐标数据
- P3-1/2/3 新增指标尚未在 catalog/*.yaml 中注册 — 留给后续"指标注册"task
- EV19 软门 _gate.py 已就绪，但 6 范式 templates/*.py 尚未创建 — 等后续 E2 task
- LDB 默认变体 `OpenFieldRectangle-Subdivided2x2` 是临时兜底 — 等行为学同事 PR 修正

## 后续工作

- 6 范式分析模板（templates/epm.py 等）— 依赖行为学同事 PR
- 新增 P3 指标的 catalog YAML 注册 + CLI 脚本
- shoaling golden-case 校验
```

- [ ] **Step 2: Commit**

```bash
git -C /home/wangqiuyang/noldus-insight add docs/handoffs/2026-05/2026-05-27-p2-p3-ev19-completion-handoff.md
git -C /home/wangqiuyang/noldus-insight commit -m "docs: P2+P3+EV19 Skill 收尾实施完成交接文档"
```

---

## 验收清单

完成全部 task 后逐项打勾：

### P2
- [ ] P2-1: `grep "set_box_aspect" packages/ethoinsight/ethoinsight/charts.py` 返回 2 处（trajectory + heatmap）
- [ ] P2-2: `grep "_prefer_center_suffix\|_get_open_zone_cols" packages/ethoinsight/ethoinsight/metrics/epm.py` 返回新函数
- [ ] P2-3: `ls packages/ethoinsight/ethoinsight/scripts/ldb/plot_zone_entry_distribution.py` 存在
- [ ] P2-4: `grep "proximity_threshold_cm" packages/ethoinsight/ethoinsight/metrics/zero_maze.py` 返回新参数
- [ ] P2-5: `grep "group_label" packages/ethoinsight/ethoinsight/charts.py` 返回 time_progress_plot 新参数

### P3
- [ ] P3-1: `grep "compute_body_elongation_stats" packages/ethoinsight/ethoinsight/metrics/_common.py` 返回函数定义
- [ ] P3-2: `grep "compute_head_direction_stats" packages/ethoinsight/ethoinsight/metrics/_common.py` 返回函数定义
- [ ] P3-3: `grep "compute_turn_angle_stats" packages/ethoinsight/ethoinsight/metrics/_common.py` 返回函数定义
- [ ] P3-4: `grep "def rose_plot" packages/ethoinsight/ethoinsight/charts.py` 返回函数定义

### EV19 收尾
- [ ] C1: `ls packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md` 存在
- [ ] C2: `ls packages/ethoinsight/ethoinsight/templates/_gate.py` 存在，`pytest tests/test_template_soft_gate.py` 全绿
- [ ] C3: skill 系统能加载 ethovision-paradigm-knowledge
- [ ] C4: quality-gates.md 含 ev19_template 字段引用
- [ ] C5: `make test` 全绿（ethoinsight + agent backend）
- [ ] C6: 交接文档已写

---

## 已知风险与回退

1. **P2-2 center point 优先可能改变现有测试期望值** — 如果 EPM/Zero Maze 测试数据只有 nose 列没有 center 列，fallback 逻辑会退回到全部匹配，行为不变
2. **P2-5 group-aggregate 需要 --groups JSON 输入** — 如果 lead agent 不传 --groups，脚本退回到 per-subject 模式，功能不退化
3. **P3-1/2/3 新增的列名 (Elongation/Direction/TurnAngle) 未在真实数据中验证** — 函数在列缺失时返回 None，安全降级
4. **ev19_facts.py 从包内 `_facts.json` 加载** — 需确认 `_facts.json` 已复制到 `packages/ethoinsight/ethoinsight/` 同目录（Docker 容器内不依赖仓库路径）

回退方案：

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -20  # 找到本次第一个 commit 的 hash
git reset --hard <prev_commit_hash>
```
