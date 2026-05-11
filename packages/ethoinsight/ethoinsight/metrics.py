"""ethoinsight.metrics — behavioral metric computation.

Provides generic and paradigm-specific metric functions for EthoVision data.
All functions accept parsed DataFrames (from parse.py) and return computed
metrics. Functions return None when required columns are missing.
"""

from __future__ import annotations

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================================
# Generic metrics
# ============================================================================


def compute_distance_moved(df: pd.DataFrame) -> float | None:
    """Total distance moved (sum of ``distance_moved`` column)."""
    if "distance_moved" not in df.columns:
        return None
    return float(df["distance_moved"].dropna().sum())


def compute_velocity_stats(df: pd.DataFrame) -> dict | None:
    """Descriptive statistics for the ``velocity`` column.

    Returns dict with keys: mean, std, max, min, median.
    """
    if "velocity" not in df.columns:
        return None
    v = df["velocity"].dropna()
    if v.empty:
        return None
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "max": float(v.max()),
        "min": float(v.min()),
        "median": float(v.median()),
    }


# ============================================================================
# Shoaling metrics
# ============================================================================


def _align_subjects_xy(
    subjects: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Align subject coordinates to a common trial_time index.

    Returns:
        times: 1-D array of trial_time values (intersection)
        coords: array of shape (n_subjects, n_timepoints, 2)  — x, y
    """
    dfs = {}
    for name, df in subjects.items():
        if "trial_time" not in df.columns:
            continue
        if "x_center" not in df.columns or "y_center" not in df.columns:
            continue
        sub = df[["trial_time", "x_center", "y_center"]].dropna().copy()
        sub = sub.set_index("trial_time")
        # De-duplicate index (keep first)
        sub = sub[~sub.index.duplicated(keep="first")]
        dfs[name] = sub

    if len(dfs) < 2:
        return np.array([]), np.array([])

    # Intersect time indices
    common_idx = dfs[next(iter(dfs))].index
    for sub_df in dfs.values():
        common_idx = common_idx.intersection(sub_df.index)
    common_idx = common_idx.sort_values()

    if common_idx.empty:
        return np.array([]), np.array([])

    times = common_idx.to_numpy()
    coords = np.stack(
        [dfs[name].loc[common_idx, ["x_center", "y_center"]].to_numpy()
         for name in dfs],
        axis=0,
    )
    return times, coords


def compute_inter_individual_distance(
    subjects: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Inter-individual distance (IID) over time.

    For each timepoint, computes all pairwise Euclidean distances
    between subjects and summarises as mean, std, min, max.

    Returns DataFrame with columns:
        trial_time, mean_iid, std_iid, min_iid, max_iid
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size == 0:
        return None

    n_subjects = coords.shape[0]
    pair_indices = list(combinations(range(n_subjects), 2))
    n_pairs = len(pair_indices)

    # Compute pairwise distances: shape (n_pairs, n_timepoints)
    dists = np.empty((n_pairs, len(times)))
    for k, (i, j) in enumerate(pair_indices):
        diff = coords[i] - coords[j]
        dists[k] = np.sqrt((diff ** 2).sum(axis=1))

    return pd.DataFrame({
        "trial_time": times,
        "mean_iid": dists.mean(axis=0),
        "std_iid": dists.std(axis=0),
        "min_iid": dists.min(axis=0),
        "max_iid": dists.max(axis=0),
    })


def compute_nearest_neighbor_distance(
    subjects: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Nearest-neighbor distance (NND) per subject over time.

    For each subject at each timepoint, finds the distance to the
    closest other subject.

    Returns DataFrame with columns:
        trial_time, subject, nnd
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size == 0:
        return None

    n_sub = coords.shape[0]
    subject_names = list(subjects.keys())[:n_sub]

    rows = []
    for i in range(n_sub):
        # Distance from subject i to all others: (n_others, n_time)
        others = [j for j in range(n_sub) if j != i]
        other_coords = coords[others]  # (n_others, n_time, 2)
        diff = other_coords - coords[i][np.newaxis, :, :]  # broadcast
        dist_to_others = np.sqrt((diff ** 2).sum(axis=2))  # (n_others, n_time)
        nnd = dist_to_others.min(axis=0)  # (n_time,)
        for t_idx, t in enumerate(times):
            rows.append({"trial_time": t, "subject": subject_names[i], "nnd": float(nnd[t_idx])})

    return pd.DataFrame(rows)


def compute_group_polarity(
    subjects: dict[str, pd.DataFrame],
    smooth_window: int = 5,
) -> pd.DataFrame | None:
    """Group polarisation (alignment of movement directions) over time.

    Computes heading from consecutive (dx, dy), smooths, then calculates
    mean resultant length R = |mean(e^{i*theta})| at each timepoint.
    R in [0, 1]: 0 = random directions, 1 = perfectly aligned.

    Returns DataFrame with columns: trial_time, polarity
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size < smooth_window + 2:
        return None

    n_sub = coords.shape[0]

    # Compute heading angles from consecutive position differences
    dx = np.diff(coords[:, :, 0], axis=1)  # (n_sub, n_time-1)
    dy = np.diff(coords[:, :, 1], axis=1)
    theta = np.arctan2(dy, dx)  # (n_sub, n_time-1)

    # Smooth with rolling mean on unit vectors (circular smoothing)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    kernel = np.ones(smooth_window) / smooth_window
    cos_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), 1, cos_theta
    )
    sin_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), 1, sin_theta
    )
    theta_smooth = np.arctan2(sin_smooth, cos_smooth)

    # Mean resultant length at each timepoint
    n_valid = theta_smooth.shape[1]
    mean_cos = theta_smooth.copy()
    mean_sin = theta_smooth.copy()
    # Convert back to unit vectors for averaging across subjects
    cos_vals = np.cos(theta_smooth)  # (n_sub, n_valid)
    sin_vals = np.sin(theta_smooth)
    mean_r = np.sqrt(cos_vals.mean(axis=0) ** 2 + sin_vals.mean(axis=0) ** 2)

    # Trim time to match valid window
    offset = (len(times) - 1) - n_valid  # diff loses 1, convolve valid loses (window-1)
    start = 1 + (smooth_window - 1)  # skip first diff + convolution padding
    valid_times = times[start: start + n_valid]

    if len(valid_times) != len(mean_r):
        # Edge case: just use first n_valid times
        valid_times = times[:n_valid]

    return pd.DataFrame({
        "trial_time": valid_times,
        "polarity": mean_r,
    })


# ============================================================================
# Open Field metrics
# ============================================================================


def _find_zone_column(df: pd.DataFrame, pattern: str) -> str | None:
    """Find a column matching a regex pattern (case-insensitive)."""
    regex = re.compile(pattern, re.IGNORECASE)
    for col in df.columns:
        if regex.search(col):
            return col
    return None


def compute_center_time_ratio(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> float | None:
    """Ratio of time spent in center zone.

    Tries exact column name first, then regex ``in_zone.*center``.
    Zone columns are expected to be 0/1 integers.
    """
    col = None
    if center_zone in df.columns:
        col = center_zone
    else:
        col = _find_zone_column(df, r"in_zone.*center")
    if col is None:
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    return float(series.mean())


def compute_thigmotaxis_index(
    df: pd.DataFrame,
    arena_center: tuple[float, float] | None = None,
    arena_radius: float | None = None,
    periphery_fraction: float = 0.2,
) -> float | None:
    """Thigmotaxis index: fraction of time in peripheral zone.

    If arena geometry is provided, computes from coordinates.
    Otherwise tries to find a periphery zone column.
    """
    # Try zone column first
    col = _find_zone_column(df, r"in_zone.*(peripher|edge|wall|border)")
    if col is not None:
        series = df[col].dropna()
        if not series.empty:
            return float(series.mean())

    # Compute from coordinates
    if arena_center is None or arena_radius is None:
        return None
    if "x_center" not in df.columns or "y_center" not in df.columns:
        return None

    x = df["x_center"].dropna()
    y = df["y_center"].dropna()
    idx = x.index.intersection(y.index)
    if idx.empty:
        return None

    dist = np.sqrt(
        (x.loc[idx] - arena_center[0]) ** 2
        + (y.loc[idx] - arena_center[1]) ** 2
    )
    threshold = arena_radius * (1 - periphery_fraction)
    return float((dist > threshold).mean())


# ============================================================================
# EPM metrics
# ============================================================================


def compute_open_arm_time_ratio(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
) -> float | None:
    """Ratio of time in open arms of elevated plus maze.

    Searches for columns matching ``in_zone.*open_arm`` or uses
    explicitly provided column names.
    """
    if open_arm_zones:
        cols = [c for c in open_arm_zones if c in df.columns]
    else:
        cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]

    if not cols:
        return None

    # Combine: in open arm if any open arm zone column == 1
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return None
    return float(combined.mean())


def _count_zone_entries(df: pd.DataFrame, zone_cols: list[str]) -> int | None:
    """Count 0→1 transitions across zone columns (combined with OR).

    Returns None when no matching columns exist, 0 when no entries detected.
    """
    if not zone_cols:
        return None
    # Combine multiple zone columns: in zone if ANY column == 1
    combined = df[zone_cols].max(axis=1).dropna()
    if combined.empty:
        return 0
    vals = combined.to_numpy(dtype=int)
    # Entry: 0→1 transition. First frame being 1 also counts as an entry.
    entries = 1 if vals[0] == 1 else 0
    transitions = (vals[1:] == 1) & (vals[:-1] == 0)
    return entries + int(transitions.sum())


def _find_arm_zone_columns(df: pd.DataFrame) -> list[str]:
    """Find all arm-related zone columns (open_arm, closed_arm, arm_*).

    Excludes center/centre columns.
    """
    arm_cols = []
    for col in df.columns:
        if not col.startswith("in_zone_"):
            continue
        col_lower = col.lower()
        # Skip center/centre zones
        if "center" in col_lower or "centre" in col_lower:
            continue
        # Match arm patterns
        if re.search(r"(open.?arm|closed.?arm|arm.?\d)", col_lower):
            arm_cols.append(col)
    return arm_cols


def _get_frame_duration(df: pd.DataFrame) -> float | None:
    """Estimate frame duration (seconds) from trial_time column."""
    if "trial_time" not in df.columns:
        return None
    tt = df["trial_time"].dropna()
    if len(tt) < 2:
        return None
    diffs = tt.diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.median())


def compute_open_arm_entry_count(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
) -> int | None:
    """Number of entries into open arms (0→1 transitions)."""
    if open_arm_zones:
        cols = [c for c in open_arm_zones if c in df.columns]
    else:
        cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]
    return _count_zone_entries(df, cols)


def compute_open_arm_entry_ratio(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
) -> float | None:
    """Ratio of open arm entries to total arm entries."""
    open_count = compute_open_arm_entry_count(df, open_arm_zones)
    total_count = compute_total_entry_count(df)
    if open_count is None or total_count is None or total_count == 0:
        return None
    return open_count / total_count


def compute_open_arm_time(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
) -> float | None:
    """Total time (seconds) in open arms.

    Multiplies the number of open-arm frames by the median inter-frame
    interval from ``trial_time``. Falls back to frame count when
    ``trial_time`` is unavailable.
    """
    if open_arm_zones:
        cols = [c for c in open_arm_zones if c in df.columns]
    else:
        cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]
    if not cols:
        return None
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return 0.0
    n_frames = int(combined.sum())
    dt = _get_frame_duration(df)
    if dt is not None:
        return n_frames * dt
    return float(n_frames)


def compute_total_entry_count(df: pd.DataFrame) -> int | None:
    """Total entries into all arm zones (open + closed, excluding center).

    Counts entries into open-arm and closed-arm zone groups separately
    (each group combines its columns with OR to avoid overcounting),
    then sums across groups.
    """
    open_cols = [c for c in df.columns if re.search(r"in_zone.*open.?arm", c, re.I)]
    closed_cols = [c for c in df.columns if re.search(r"in_zone.*closed.?arm", c, re.I)]
    if not open_cols and not closed_cols:
        return None
    open_entries = _count_zone_entries(df, open_cols) or 0
    closed_entries = _count_zone_entries(df, closed_cols) or 0
    return open_entries + closed_entries


# ============================================================================
# Paradigm dispatcher
# ============================================================================


def compute_paradigm_metrics(
    parsed_data: dict,
    paradigm: str,
    groups: dict[str, list[str]] | None = None,
    metrics: list[str] | None = None,
) -> dict:
    """Compute metrics for a specific paradigm.

    Args:
        parsed_data: Output of ``parse.parse_batch()``.
        paradigm: Paradigm name (e.g. "shoaling", "epm", "open_field").
        groups: Optional grouping ``{group_name: [subject_name, ...]}``.
            If None, all subjects are in a single "all" group.
        metrics: Optional list of metric names to compute.
            If None, computes all metrics for the paradigm.

    Returns:
        {
            "paradigm": str,
            "per_subject": {subject: {metric: value, ...}, ...},
            "group_summary": {group: {metric: {"mean", "std", "n", "values"}, ...}, ...},
            "timeseries": {metric_name: DataFrame, ...},
            "metadata": {"n_subjects", "n_files", "duration_s", "computed_metrics"},
        }
    """
    subjects = parsed_data.get("subjects", {})
    summary = parsed_data.get("summary", {})

    # Default: all subjects in one group
    if groups is None:
        groups = {"all": list(subjects.keys())}

    # Compute per-subject scalar metrics
    per_subject: dict[str, dict] = {}
    for name, df in subjects.items():
        m: dict[str, float | dict | None] = {}
        m["distance_moved"] = compute_distance_moved(df)
        m["velocity_stats"] = compute_velocity_stats(df)
        if paradigm == "open_field":
            m["center_time_ratio"] = compute_center_time_ratio(df)
            m["thigmotaxis_index"] = compute_thigmotaxis_index(df)
        elif paradigm == "epm":
            m["open_arm_time_ratio"] = compute_open_arm_time_ratio(df)
            m["open_arm_entry_ratio"] = compute_open_arm_entry_ratio(df)
            m["open_arm_entry_count"] = compute_open_arm_entry_count(df)
            m["open_arm_time"] = compute_open_arm_time(df)
            m["total_entry_count"] = compute_total_entry_count(df)
        per_subject[name] = m

    # Compute shoaling group-level timeseries
    timeseries: dict[str, pd.DataFrame] = {}
    group_level_metrics: dict[str, float | dict] = {}
    if paradigm == "shoaling":
        n_sub = len(subjects)
        if n_sub >= 2:
            iid = compute_inter_individual_distance(subjects)
            if iid is not None:
                timeseries["inter_individual_distance"] = iid
                group_level_metrics["mean_iid"] = float(iid["mean_iid"].mean())
            nnd = compute_nearest_neighbor_distance(subjects)
            if nnd is not None:
                timeseries["nearest_neighbor_distance"] = nnd
                # NND is genuinely per-subject (each fish has its own nearest neighbour).
                for name in per_subject:
                    sub_nnd = nnd.loc[nnd["subject"] == name, "nnd"]
                    per_subject[name]["mean_nnd"] = float(sub_nnd.mean()) if not sub_nnd.empty else None
            pol = compute_group_polarity(subjects)
            if pol is not None:
                timeseries["group_polarity"] = pol
                group_level_metrics["mean_polarity"] = float(pol["polarity"].mean())
        else:
            # Single-subject (or empty) input — IID / polarity are not applicable.
            # These are group metrics that require ≥2 simultaneously tracked subjects.
            # DO NOT fabricate zero-variance per-subject scalars.
            group_level_metrics["mean_iid"] = {
                "applicable": False,
                "reason": "group metric requires ≥2 simultaneously tracked subjects",
            }
            group_level_metrics["mean_polarity"] = {
                "applicable": False,
                "reason": "group metric requires ≥2 simultaneously tracked subjects",
            }

    # Filter metrics if requested
    if metrics:
        for name in per_subject:
            per_subject[name] = {
                k: v for k, v in per_subject[name].items() if k in metrics
            }

    # Build group summary
    group_summary: dict[str, dict] = {}
    for grp_name, grp_subjects in groups.items():
        grp_metrics: dict[str, dict] = {}
        matched = [s for s in grp_subjects if s in per_subject]
        if not matched:
            continue
        # Collect all scalar metric names
        all_metric_names = set()
        for s in matched:
            for k, v in per_subject[s].items():
                if isinstance(v, (int, float)) and v is not None:
                    all_metric_names.add(k)

        for mname in sorted(all_metric_names):
            values = [
                per_subject[s][mname]
                for s in matched
                if mname in per_subject[s] and per_subject[s][mname] is not None
                and isinstance(per_subject[s][mname], (int, float))
            ]
            if values:
                arr = np.array(values, dtype=float)
                grp_metrics[mname] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                    "values": [float(v) for v in arr],
                }
        group_summary[grp_name] = grp_metrics

    computed = sorted({
        k for subj in per_subject.values()
        for k, v in subj.items() if v is not None
    })

    # Data quality warnings — surface critical sample-size concerns without blocking.
    data_quality_warnings: list[dict] = []
    for grp_name, grp_metrics in group_summary.items():
        if not grp_metrics:
            continue
        # All metrics within a group share the same n (they come from the same subjects)
        sample_n = next(iter(grp_metrics.values())).get("n", 0)
        if sample_n < 3:
            data_quality_warnings.append({
                "severity": "critical",
                "metric": "all",
                "message": (
                    f"Group '{grp_name}' has n={sample_n} (<3). "
                    "Statistical inference will be unreliable; descriptive statistics only."
                ),
            })
    if paradigm == "shoaling" and len(subjects) < 2:
        data_quality_warnings.append({
            "severity": "warning",
            "metric": "mean_iid,mean_polarity",
            "message": (
                "Shoaling group metrics (IID, polarity) are not applicable: "
                "only 1 subject detected. Group metrics require ≥2 simultaneously tracked subjects."
            ),
        })
    if paradigm == "epm":
        # Per epm.md: n < 5 per group → low statistical power
        for grp_name, grp_metrics in group_summary.items():
            if not grp_metrics:
                continue
            sample_n = next(iter(grp_metrics.values())).get("n", 0)
            if 0 < sample_n < 5:
                data_quality_warnings.append({
                    "severity": "warning",
                    "metric": "all",
                    "message": (
                        f"Group '{grp_name}' has n={sample_n} (<5). "
                        "统计功效不足，结论需谨慎。"
                    ),
                })
        # Per epm.md: total entries < 8 → motor suppression confound
        for name, m in per_subject.items():
            te = m.get("total_entry_count")
            if te is not None and isinstance(te, (int, float)) and te < 8:
                data_quality_warnings.append({
                    "severity": "warning",
                    "metric": "total_entry_count",
                    "message": (
                        f"Subject '{name}' 总进臂次数={int(te)} (<8)。"
                        "开臂指标的下降可能为运动抑制而非焦虑增加，需标注警告。"
                    ),
                })

    return {
        "paradigm": paradigm,
        "per_subject": per_subject,
        "group_summary": group_summary,
        "group_level_metrics": group_level_metrics,
        "timeseries": timeseries,
        "data_quality_warnings": data_quality_warnings,
        "metadata": {
            "n_subjects": len(subjects),
            "n_files": summary.get("total_files", 0),
            "duration_s": summary.get("duration_seconds", 0),
            "computed_metrics": computed,
        },
    }


# ============================================================================
# Export
# ============================================================================


def save_to_csv(metrics_result: dict, path: str) -> str:
    """Save per-subject metrics to CSV.

    Flattens ``per_subject`` into a table where each row is a subject
    and each column is a metric (nested dicts like velocity_stats are
    expanded with underscore-separated keys).

    Returns the saved file path.
    """
    rows = []
    for subject, mdict in metrics_result.get("per_subject", {}).items():
        row: dict[str, object] = {"subject": subject}
        for k, v in mdict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    row[f"{k}_{sub_k}"] = sub_v
            else:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path
