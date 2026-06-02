# 2026-05-27 Noldus JS B2-B10 Implementation Plan (v2 — post Opus review)

## Handoff Summary

Implement B2-B10 algorithms ported from Noldus JS examples (`Noldus/EthoVision-JavaScriptCustomAnalysis`).
All existing P0/P1/P2/P3/B1 are complete on dev HEAD ceec0581. No new SSOT, no new tools, no architecture changes.

## SSOT Note

Review-packages (`docs/review-packages/2026-04-29-ev19-templates/`) enumerate must-have metrics only (中心区时间百分比, 中心区距离百分比, 中心区进入次数, 总运动距离 for OFT). None of the B2-B10 metrics appear in the SSOT. These are introduced as `optional_metrics` mirroring existing precedent (`body_elongation_stats`, `turn_angle_stats` are also optional_metrics not in SSOT). This is consistent with the handoff principle: "不建新 SSOT".

## Pattern Reference (existing codebase conventions)

### Metric function
- Location: `metrics/_common.py` (shared) or `metrics/<paradigm>.py` (paradigm-specific)
- Signature: `def compute_*(df: pd.DataFrame, ...) -> dict | float | None`
- Returns None when required columns missing or all-NaN
- Uses `_estimate_dt()` for frame-rate adaptation

### Script (CLI wrapper)
- Location: `scripts/<paradigm>/compute_*.py` — **one script per paradigm**, even when the underlying function is in `_common.py`. Convention: `scripts/oft/compute_body_elongation_stats.py` + `scripts/epm/compute_body_elongation_stats.py` both wrap `_common.compute_body_elongation_stats`.
- `scripts/_common/` is reserved for paradigm-agnostic scripts not bound to any paradigm catalog.
- Pattern: `parse_trajectory(args.input) → compute_*(df) → save_output_json(args.output, payload) → emit_result(payload)`
- Uses `make_compute_parser` from `scripts/_cli.py`

### Catalog entry
- Location: `catalog/<paradigm>.yaml` under `optional_metrics`
- Fields: id, script, requires_columns, output_unit, display_name_zh, unit_zh, one_liner, direction_for_anxiety, statistical_default

### Test
- Location: `tests/test_metrics_*.py` or `tests/test_metrics.py`
- Pattern: synthetic DataFrames, test basic/missing-column/all-NaN/edge-cases

---

## PR 1: B2 + B3 — Motion Smoothing

### B2 — smoothed_activity (chart enhancement)

**Algorithm**: SMA queue (window=10) applied to Activity pixel-change values before plotting, matching Noldus JS `Smoothing = 10`.

**Files to modify**:
- `packages/ethoinsight/ethoinsight/charts.py:697-738` — `activity_intensity_plot()`
  - Add `smooth_window: int | None = None` parameter
  - When `smooth_window > 0`: apply `pd.Series(v).rolling(window=smooth_window, min_periods=1).mean()` before fill_between/plot
  - Default None preserves library-level backward compat
- `packages/ethoinsight/ethoinsight/scripts/fst/plot_activity_intensity.py:30`
  - Change `activity_intensity_plot(df, output_path=args.output)` → `activity_intensity_plot(df, output_path=args.output, smooth_window=10)`
- `packages/ethoinsight/ethoinsight/scripts/tst/plot_activity_intensity.py:30`
  - Same change: pass `smooth_window=10`

**No catalog change**. Existing chart tests (`test_plot_fst_activity_intensity_cli.py`, `test_plot_tst_activity_intensity_cli.py`) should continue to pass (smoothing changes curve shape but not the output path contract).

### B3 — turn_angle_filtered (new metric + catalog + script)

**Algorithm**: Sliding window of 3 center-points. Each new point is only added to window if its displacement from the previous point > `min_displacement_mm` (default 1.0). Compute TurnAngle from 3-point window. Filters out jitter when animal is stationary. NaN frames are skipped (matching Noldus JS `if (pt)` guard).

**Files to create**:
- `packages/ethoinsight/ethoinsight/scripts/oft/compute_turn_angle_filtered.py` — thin CLI wrapper

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_turn_angle_filtered(df, min_displacement_mm=1.0)`
  - Requires `x_center`, `y_center` columns
  - Skips NaN frames (matching Noldus `if (pt)` check)
  - Returns `{mean_abs_rad, mean_abs_deg, std_abs_rad, n}` or None
- `packages/ethoinsight/ethoinsight/catalog/oft.yaml` — add to `optional_metrics`:
  ```yaml
  - id: turn_angle_filtered
    script: ethoinsight.scripts.oft.compute_turn_angle_filtered
    requires_columns: [x_center, y_center]
    output_unit: radians
    display_name_zh: 过滤转角统计
    unit_zh: 弧度
    one_liner: 带位移过滤的转角统计（排除静止抖动），基于 Noldus Turn Angle with Distance moved filter
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```
- `packages/ethoinsight/tests/test_metrics_turn_angle.py` — add 4 tests:
  - `test_turn_angle_filtered_basic` — straight line + turn, verify angle
  - `test_turn_angle_filtered_stationary` — tiny jitter below 1mm, verify filtered out
  - `test_turn_angle_filtered_missing_columns` — returns None
  - `test_turn_angle_filtered_nan_handling` — NaN x_center/y_center frames skipped

---

## PR 2: B4 + B7 + B8 — Motion Analysis

### B4 — velocity_bins (new metric + catalog + script)

**Algorithm**: Bin per-frame velocity into preset ranges, return time-ratio per bin. Default bin edges `[0, 50, 100, 200, 400]` mm/s (EV19/Noldus standard).

**Files to create**:
- `packages/ethoinsight/ethoinsight/scripts/oft/compute_velocity_bins.py`

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_velocity_bins(df, bin_edges=None)`
  - Uses `velocity` column if present, else derives from `distance_moved` / dt
  - Returns `{f"{lo}-{hi}": ratio, ...}` or None
- `packages/ethoinsight/ethoinsight/catalog/oft.yaml` — add to `optional_metrics`:
  ```yaml
  - id: velocity_bins
    script: ethoinsight.scripts.oft.compute_velocity_bins
    requires_columns: [velocity]
    output_unit: ratio
    display_name_zh: 速度分箱分布
    unit_zh: 比例
    one_liner: 速度按预设边界分箱的时间占比分布，基于 Noldus Split track in bins based on velocity
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```
- `packages/ethoinsight/tests/test_metrics.py` — add 3 tests:
  - `test_velocity_bins_basic` — known velocities, verify bin ratios sum to ~1
  - `test_velocity_bins_custom_edges` — custom bin edges
  - `test_velocity_bins_missing_column` — returns None

### B7 — cumulative_distance_k (new metric + catalog + script)

**Algorithm**: `distance_moved.rolling(window=window_samples).sum()`. Used for OFT habituation analysis.

**Files to create**:
- `packages/ethoinsight/ethoinsight/scripts/oft/compute_cumulative_distance.py`

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_cumulative_distance(df, window_samples=25)`
  - Frame-rate adaptation: scale window_samples by `_estimate_dt()` ratio vs 25fps baseline
  - Returns `{mean, max, min, median}` or None
- `packages/ethoinsight/ethoinsight/catalog/oft.yaml` — add to `optional_metrics`:
  ```yaml
  - id: cumulative_distance
    script: ethoinsight.scripts.oft.compute_cumulative_distance
    requires_columns: [distance_moved]
    output_unit: cm
    display_name_zh: 滑动窗口累计距离
    unit_zh: cm
    one_liner: 滑动窗口内累计移动距离，用于习惯化分析，基于 Noldus Distance moved - Cumulative in last k samples
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```

### B8 — acceleration_stats (new metric + catalog + script)

**Algorithm**: SMA-smooth velocity (window=5), then diff/dt to get acceleration.

**Files to create**:
- `packages/ethoinsight/ethoinsight/scripts/oft/compute_acceleration_stats.py`

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_acceleration_stats(df, smooth_window=5)`
  - Gets velocity from `velocity` column or `distance_moved`/dt
  - Applies `rolling(window=smooth_window).mean()` SMA
  - Computes `diff() / dt` as acceleration
  - Returns `{mean, std, max, min}` or None
- `packages/ethoinsight/ethoinsight/catalog/oft.yaml` — add to `optional_metrics`:
  ```yaml
  - id: acceleration_stats
    script: ethoinsight.scripts.oft.compute_acceleration_stats
    requires_columns: [velocity]
    output_unit: mm_s2
    display_name_zh: 加速度统计
    unit_zh: mm/s²
    one_liner: 平滑后的加速度统计（均值/标准差/最大/最小），基于 Noldus Acceleration - Smoothed
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```

---

## PR 3: B5 — Body Measurement

### B5 — body_length_segments (new metric + catalog + per-paradigm scripts)

**Algorithm**: Nose→Center + Center→Tail Euclidean distance per frame. Requires all 6 body-point columns (not all paradigms/recordings have multi-point tracking).

**Files to create**:
- `packages/ethoinsight/ethoinsight/scripts/oft/compute_body_length.py`
- `packages/ethoinsight/ethoinsight/scripts/epm/compute_body_length.py`
  - Per-paradigm scripts follow the convention: `turn_angle_stats` and `body_elongation_stats` both have separate `scripts/oft/` + `scripts/epm/` wrappers around a shared `_common.py` function.

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_body_length(df)`
  - Checks all 6 columns exist: `x_nose, y_nose, x_center, y_center, x_tail, y_tail`
  - Returns None if ANY of the 6 columns are missing (per handoff instruction)
  - Returns `{mean, std, min, max, median}` (unit: same as coordinate columns, typically cm for EV19 mammal exports)
- `packages/ethoinsight/ethoinsight/catalog/epm.yaml` — add to `optional_metrics`:
  ```yaml
  - id: body_length
    script: ethoinsight.scripts.epm.compute_body_length
    requires_columns: [x_nose, y_nose, x_center, y_center, x_tail, y_tail]
    output_unit: cm
    display_name_zh: 身体长度（分段求和）
    unit_zh: cm
    one_liner: Nose→Center + Center→Tail 欧氏距离求和，基于 Noldus Body Length - Sum of segments
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```
- `packages/ethoinsight/ethoinsight/catalog/oft.yaml` — add to `optional_metrics`:
  ```yaml
  - id: body_length
    script: ethoinsight.scripts.oft.compute_body_length
    requires_columns: [x_nose, y_nose, x_center, y_center, x_tail, y_tail]
    output_unit: cm
    display_name_zh: 身体长度（分段求和）
    unit_zh: cm
    one_liner: Nose→Center + Center→Tail 欧氏距离求和，基于 Noldus Body Length - Sum of segments
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  ```
- `packages/ethoinsight/tests/test_metrics_body_elongation.py` — add 3 tests:
  - `test_body_length_basic` — synthetic 3-point body, verify length
  - `test_body_length_missing_columns` — missing tail columns → None
  - `test_body_length_all_nan` — NaN values → None

---

## PR 4: B6 + B10 — Zone/Heading Enhancement

### B6 — zone_min_duration (modify existing + pass-through)

**Algorithm**: Zone entry only counts if the animal stays in zone for ≥ N consecutive frames. Filters out brief "fly-through" crossings.

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add shared `_count_zone_entries(df, zone_cols, min_duration_frames=0)`
  - **Move** the function from `epm.py` to `_common.py` (shared by EPM + OFT)
  - Add `min_duration_frames=0` parameter (0 = disabled, backward compatible)
  - When > 0: use `_runs()` to find bouts, only count bouts with length ≥ min_duration_frames
- `packages/ethoinsight/ethoinsight/metrics/epm.py`:
  - Remove `_count_zone_entries` definition; import from `_common` instead
  - `compute_open_arm_entry_count(df, open_arm_zones=None, min_duration_frames=0)` — accept and pass through
  - `compute_total_entry_count(df, min_duration_frames=0)` — accept and pass through
- `packages/ethoinsight/ethoinsight/metrics/oft.py`:
  - `compute_center_entry_count(df, center_zone="in_zone_center", min_duration_frames=0)`:
    ```python
    def compute_center_entry_count(df, center_zone="in_zone_center", min_duration_frames=0):
        col = _find_center_zone_column(df, hint=center_zone)
        if col is None:
            return None
        return _count_zone_entries(df, [col], min_duration_frames=min_duration_frames)
    ```
    **Must preserve** `_find_center_zone_column` lookup before delegating to shared `_count_zone_entries`. This lookup handles column-name ambiguity (e.g., `in_zone_center_center_point` vs bare `in_zone`).

**No catalog change** — existing metric entries unchanged; default 0 is transparent.

**No new script** — parameter not exposed at CLI level; available as library API for future use.

### B10 — heading_smoothed (function only)

**Algorithm**: SMA on Direction column with circular unwrap to handle ±180° boundary correctly.

**Files to modify**:
- `packages/ethoinsight/ethoinsight/metrics/_common.py` — add `compute_heading_smoothed(df, window=5)`
  - Requires `Direction` column
  - `np.unwrap(rads)` → `pd.Series.rolling(window).mean()` → `% (2*pi)` re-wrap
  - Returns `{mean_rad, circular_stdev_rad, resultant_length, n}` or None

**No script, no catalog** — variant of existing `compute_head_direction_stats`; library-internal helper.

---

## Implementation Order

```
PR 1 first (B2+B3) → test → merge
PR 2 next (B4+B7+B8) → test → merge
PR 3 next (B5) → test → merge
PR 4 last (B6+B10) → test → merge
```

Each PR is independent. PR 4 should be done last because B6 touches shared infrastructure (`_count_zone_entries` move).

---

## Verification

After each PR:
```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m pytest tests/ -q
```

After all PRs:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test
```

## Git
All commands use `git -C /home/wangqiuyang/noldus-insight`.

---

## Key Design Decisions (updated from review)

1. **B2 `smooth_window=None` in library, `=10` in FST/TST scripts**: Library stays flexible; Noldus JS smoothing behavior activated at the script layer where it matters.
2. **B5 per-paradigm scripts**: Follows the `turn_angle_stats` / `body_elongation_stats` precedent: shared function in `_common.py`, separate thin wrappers in `scripts/oft/` + `scripts/epm/`.
3. **B5 unit = cm**: Matches existing OFT/EPM conventions (`center_distance`, `distance_moved`).
4. **B6 `_count_zone_entries` moves to `_common.py`**: Shared by EPM + OFT; import paths verified (no cross-module imports of this function exist).
5. **B6 OFT refactor preserves `_find_center_zone_column`**: Column-name resolution must happen before delegating to shared `_count_zone_entries`.
6. **B6 `min_duration_frames=0` default**: Zero behavioral change. Feature available as library API; CLI/catalog exposure deferred.
7. **B7/B8 have catalog entries**: Cumulative distance and acceleration are useful standalone metrics; without catalog entries the scripts would be unreachable dead code.
8. **B10 function only, no script/catalog**: Variant of existing `compute_head_direction_stats`; library-internal helper.
9. **Frame-rate adaptation**: `_estimate_dt()` used for B7 window_samples (scale to match 25fps Noldus baseline). B6 min_duration is in raw frames (per handoff parameter name).
10. **B3 NaN handling**: NaN x_center/y_center frames are skipped (matching Noldus JS `if (pt)` guard which skips null center points).
