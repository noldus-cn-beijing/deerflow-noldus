"""范式无关的指标函数 + 共享 helper。"""

from __future__ import annotations

import os
import re

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
# Zone column helper (shared with oft.py and epm.py)
# ============================================================================


def _find_zone_column(df: pd.DataFrame, pattern: str) -> str | None:
    """Find a column matching a regex pattern (case-insensitive)."""
    regex = re.compile(pattern, re.IGNORECASE)
    for col in df.columns:
        if regex.search(col):
            return col
    return None


def _count_zone_entries(
    df: pd.DataFrame,
    zone_cols: list[str],
    min_duration_frames: int = 0,
) -> int | None:
    """Count 0→1 transitions across zone columns (combined with OR).

    Args:
        df: DataFrame with zone indicator columns (0/1).
        zone_cols: Column names to combine with OR.
        min_duration_frames: Minimum consecutive frames in zone for a bout
            to count as an entry. 0 = disabled (backward compatible).

    Returns:
        Entry count, or None when no matching columns exist.
    """
    if not zone_cols:
        return None
    combined = df[zone_cols].max(axis=1).dropna()
    if combined.empty:
        return 0
    vals = combined.to_numpy(dtype=int)

    if min_duration_frames <= 0:
        entries = 1 if vals[0] == 1 else 0
        transitions = (vals[1:] == 1) & (vals[:-1] == 0)
        return entries + int(transitions.sum())

    bouts = _runs(combined, value=1)
    count = 0
    for start, end in bouts:
        if end - start + 1 >= min_duration_frames:
            count += 1
    return count


# ============================================================================
# Immobility analysis (shared by FST / TST)
# ============================================================================


def _find_mobility_column(df: pd.DataFrame) -> str | None:
    """Find the mobility/activity state column in a DataFrame.

    Checks common EthoVision column names in order:
    Mobility_State, Activity_State, mobility_state, activity_state.
    Returns the first match.

    Only matches **state** (categorical / one-hot) columns — skips continuous
    columns like ``mobility_continuous`` or ``activity_continuous`` to avoid
    misinterpreting raw values as state labels.
    """
    candidates = [
        "Mobility_State",
        "Activity_State",
        "mobility_state",
        "activity_state",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # One-hot immobile indicator: EthoVision exports "Mobility state(Immobile)" as a
    # separate 0/1 column (after slugify: mobility_state_immobile / activity_state_immobile).
    for col in df.columns:
        cl = col.lower()
        if ("mobility_state" in cl or "activity_state" in cl) and "immobile" in cl:
            return col
    return None


def _find_activity_column(df: pd.DataFrame) -> str | None:
    """Find a continuous activity column (0–100%) for pendulum detection.

    Skips ``*_state*`` columns (categorical) and ``mobility*`` columns.
    Targets columns like ``activity``, ``Activity``, ``activity_sampleRate_1``.
    """
    for col in df.columns:
        cl = col.lower()
        if "activity" not in cl:
            continue
        if "state" in cl or "mobility" in cl:
            continue
        # Must be numeric/continuous (not a categorical state column)
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _estimate_dt(df: pd.DataFrame) -> float:
    """Estimate frame interval (seconds) from trial_time column."""
    if "trial_time" not in df.columns:
        return 0.04
    tt = df["trial_time"].dropna()
    if len(tt) < 2:
        return 0.04
    return float(tt.diff().median())


def _resolve_immobile_series(
    df: pd.DataFrame, mobility_col: str | None, **kwargs,
) -> tuple[pd.Series, int] | None:
    """Resolve the mobility column to a (series, immobile_value) pair.

    Handles two EthoVision export schemas:
    - Single combined column (``mobility_state``): 0=immobile, 1=mobile → immobile_value=0
    - One-hot immobile indicator (``mobility_state_immobile``): 1=immobile → immobile_value=1

    Returns None when no usable column is present.

    **重要假设（EV19 Mobility state 编码）**：
    EV19 ``Mobility state`` 列实际是 2/3/4 状态多分类（参见
    ev19-dependent-variables.md §14）。``immobile_value=0`` 只在 2-state 配置下
    正确；3-state（0=Inactive, 1=Moderately active, 2=Highly active，编码顺序
    为按阈值递增推断）或 4-state 下，Moderately active (1) / Active (2) 也是
    非 0 值，会被本函数错误归为 mobile。
    仅当列名含 ``immobile`` 时才确认为 one-hot 列。如用户数据来自 3/4-state 配置，
    需上游显式映射后再传入。
    """
    col = mobility_col or _find_mobility_column(df)
    if col is None or col not in df.columns:
        # Fallback 1: derive immobility from raw Activity via pendulum detection
        result = _resolve_immobile_from_activity(df, **kwargs)
        if result is not None:
            return result
        # Fallback 2: velocity-based (Noldus Non-movement bouts, last resort)
        return _resolve_immobile_from_velocity(df, **kwargs)
    series = df[col].dropna()
    if series.empty:
        result = _resolve_immobile_from_activity(df, **kwargs)
        if result is not None:
            return result
        return _resolve_immobile_from_velocity(df, **kwargs)
    immobile_value = 1 if "immobile" in col.lower() else 0
    return series, immobile_value


def _resolve_immobile_from_activity(
    df: pd.DataFrame,
    **pendulum_kwargs,
) -> tuple[pd.Series, int] | None:
    """Derive immobility labels from raw Activity via pendulum detection.

    Used as fallback when no ``mobility_state`` / ``activity_state`` column
    exists in the export.  Runs the autocorrelation-based pendulum detector on
    the continuous Activity column (pixel change %) and returns a one-hot
    immobility series (1=immobile, 0=mobile).

    pendulum_kwargs 透传给 pendulum_immobility_series → detect_pendulum。

    Returns None when no usable Activity column is present.
    """
    from ethoinsight.metrics._pendulum import pendulum_immobility_series

    activity_col = _find_activity_column(df)
    if activity_col is None:
        return None
    activity = df[activity_col].to_numpy(dtype=float)
    if np.all(np.isnan(activity)):
        return None
    dt = _estimate_dt(df)
    immobile = pendulum_immobility_series(activity, dt, **pendulum_kwargs)
    series = pd.Series(immobile, index=df.index[:len(immobile)], dtype=int)
    return series, 1  # one-hot: 1 = immobile


# Sprint 2b: _VELOCITY_THRESHOLD_MM_S / _VELOCITY_MIN_DURATION 模块常量已删除。
# 参数现在通过函数签名 kwargs 传入（与 catalog _common.yaml shared_parameters default 一致）。


def _resolve_immobile_from_velocity(
    df: pd.DataFrame,
    *,
    velocity_threshold: float = 30.0,
    velocity_min_duration: int = 25,
) -> tuple[pd.Series, int] | None:
    """Derive immobility from center-point velocity (Noldus Non-movement bouts).

    Last-resort fallback when neither ``mobility_state`` nor ``activity``
    columns are available.  Classifies a frame as immobile when the
    center-point velocity (Euclidean distance / time delta) ≤ velocity_threshold
    for at least velocity_min_duration consecutive samples (frame-rate adaptive).

    Sprint 2b: velocity_threshold / velocity_min_duration 从 catalog 传入,
        default 30.0 / 25 仅供本地 unit test 使用。

    Returns None when ``x_center`` or ``y_center`` columns are missing.
    """
    if "x_center" not in df.columns or "y_center" not in df.columns:
        return None
    x = df["x_center"].to_numpy(dtype=float)
    y = df["y_center"].to_numpy(dtype=float)
    dt = _estimate_dt(df)

    # Frame-rate adaptation: scale min-duration to match 25fps baseline
    scale = 0.04 / dt if dt > 0 and dt != 0.04 else 1.0
    min_dur = max(1, round(velocity_min_duration * scale))

    n = len(x)
    immobile = np.zeros(n, dtype=int)
    counter = 0
    for i in range(1, n):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(x[i - 1]) or np.isnan(y[i - 1]):
            counter = 0
            continue
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        velocity = np.sqrt(dx * dx + dy * dy) / dt
        if velocity <= velocity_threshold:
            counter += 1
        else:
            counter = 0
        if counter >= min_dur:
            immobile[i] = 1

    series = pd.Series(immobile, index=df.index[:n], dtype=int)
    return series, 1  # one-hot: 1 = immobile


def _runs(arr, value=0):
    """Return list of (start_idx, end_idx) for consecutive runs of `value`.

    Used for immobility bout detection.
    """
    a = np.asarray(arr, dtype=int)
    if len(a) == 0:
        return []
    is_val = (a == value).astype(int)
    starts = []
    ends = []
    if is_val[0] == 1:
        starts.append(0)
    for i in range(1, len(is_val)):
        if is_val[i] == 1 and is_val[i - 1] == 0:
            starts.append(i)
        if is_val[i] == 0 and is_val[i - 1] == 1:
            ends.append(i - 1)
    if is_val[-1] == 1:
        ends.append(len(is_val) - 1)
    return list(zip(starts, ends))


def compute_immobility_time(
    df: pd.DataFrame,
    mobility_col: str | None = None,
    **kwargs,
) -> float | None:
    """Total immobility time (seconds).

    Sums the duration of all immobility bouts. Works with both the combined
    ``mobility_state`` column (0=immobile) and the one-hot
    ``mobility_state_immobile`` column (1=immobile).

    kwargs 透传给 _resolve_immobile_series（Sprint 2b: velocity_threshold,
    velocity_min_duration, pendulum_* 参数）。
    """
    resolved = _resolve_immobile_series(df, mobility_col, **kwargs)
    if resolved is None:
        return None
    series, immobile_value = resolved

    bouts = _runs(series, value=immobile_value)
    if not bouts:
        return 0.0

    total_frames = sum(end - start + 1 for start, end in bouts)

    # Convert frames to seconds
    if "trial_time" in df.columns:
        tt = df["trial_time"].dropna()
        if len(tt) >= 2:
            dt = float(tt.diff().median())
            return total_frames * dt
    return float(total_frames)


def compute_immobility_latency(
    df: pd.DataFrame,
    mobility_col: str | None = None,
    **kwargs,
) -> float | None:
    """Latency to first immobility bout (seconds).

    Returns the trial_time value of the first immobile frame, or None if
    the animal was never immobile.

    kwargs 透传给 _resolve_immobile_series（Sprint 2b）。
    """
    resolved = _resolve_immobile_series(df, mobility_col, **kwargs)
    if resolved is None:
        return None
    series, immobile_value = resolved

    immobile_mask = series == immobile_value
    if not immobile_mask.any():
        return None

    first_idx = immobile_mask.idxmax()  # index of first True

    if "trial_time" in df.columns and first_idx in df.index:
        return float(df.loc[first_idx, "trial_time"])
    return float(first_idx)


def compute_immobility_bout_count(
    df: pd.DataFrame,
    mobility_col: str | None = None,
    **kwargs,
) -> int | None:
    """Number of immobility bouts (run-length encoding).

    Each consecutive run of immobile frames counts as one bout.

    kwargs 透传给 _resolve_immobile_series（Sprint 2b）。
    """
    resolved = _resolve_immobile_series(df, mobility_col, **kwargs)
    if resolved is None:
        return None
    series, immobile_value = resolved

    bouts = _runs(series, value=immobile_value)
    return len(bouts)


def extract_immobility_bouts(
    df: pd.DataFrame,
    mobility_col: str | None = None,
    **kwargs,
) -> list[tuple[float, float]]:
    """Extract (start_sec, end_sec) pairs for each immobility bout.

    Uses trial_time column to convert frame indices to seconds.
    Returns empty list if mobility column or trial_time is missing.

    kwargs 透传给 _resolve_immobile_series（Sprint 2b）。
    """
    resolved = _resolve_immobile_series(df, mobility_col, **kwargs)
    if resolved is None:
        return []
    series, immobile_value = resolved

    frame_bouts = _runs(series, value=immobile_value)
    if not frame_bouts:
        return []

    if "trial_time" not in df.columns:
        return [(float(start), float(end)) for start, end in frame_bouts]

    times = df["trial_time"].reset_index(drop=True)
    result = []
    for start_idx, end_idx in frame_bouts:
        try:
            t_start = float(times.iloc[start_idx])
            t_end = float(times.iloc[end_idx])
            result.append((t_start, t_end))
        except (IndexError, ValueError):
            continue
    return result


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


# ============================================================================
# EV19 dependent variables (P3 — body elongation, head direction, turn angle)
# ============================================================================


def compute_body_elongation_stats(df: pd.DataFrame) -> dict | None:
    """Descriptive statistics for Body elongation (EV19 GetElongation × 100).

    EV19 Elongation column: 0–1 ratio. Converted to 0–100 per EV19 definition.

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
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    mean_rad = float(np.arctan2(sin_sum, cos_sum)) % (2 * np.pi)
    circular_stdev = float(np.sqrt(-2 * np.log(max(min(R / n, 1.0 - 1e-12), 1e-12)))) if n > 0 and R > 0 else float("inf")
    return {
        "mean_rad": mean_rad,
        "circular_stdev_rad": circular_stdev,
        "resultant_length": float(R),
        "n": n,
    }


def compute_turn_angle_stats(df: pd.DataFrame) -> dict | None:
    """Descriptive statistics for Turn angle (EV19 TurnAngle, radians).

    Reports absolute turn angle (unsigned, deg) for locomotion analysis.

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


def compute_turn_angle_filtered(
    df: pd.DataFrame,
    min_displacement_mm: float = 1.0,
) -> dict | None:
    """Turn angle with distance-moved filter (Noldus "Turn Angle with Distance moved filter").

    Sliding window of 3 center-points. A new point is only added when its
    displacement from the previous point exceeds *min_displacement_mm*,
    filtering out jitter when the animal is stationary.

    Args:
        df: DataFrame with ``x_center``, ``y_center`` columns.
        min_displacement_mm: Minimum displacement (in coordinate units) for
            a point to enter the 3-point window. Noldus JS default: 1.0.

    Returns:
        Dict with keys mean_abs_rad, mean_abs_deg, std_abs_rad, n, or None
        when required columns are missing or fewer than 3 valid points exist.
    """
    if "x_center" not in df.columns or "y_center" not in df.columns:
        return None
    x = df["x_center"].to_numpy(dtype=float)
    y = df["y_center"].to_numpy(dtype=float)
    n = len(x)

    points: list[tuple[float, float]] = []
    angles: list[float] = []

    for i in range(n):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        pt = (float(x[i]), float(y[i]))
        if points:
            last = points[-1]
            d = np.sqrt((pt[0] - last[0]) ** 2 + (pt[1] - last[1]) ** 2)
            if d <= min_displacement_mm:
                continue
        points.append(pt)
        if len(points) > 3:
            points.pop(0)
        if len(points) == 3:
            v1 = (points[1][0] - points[0][0], points[1][1] - points[0][1])
            v2 = (points[2][0] - points[1][0], points[2][1] - points[1][1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            norm2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                angle = float(np.arccos(cos_angle))
                angles.append(angle)

    if len(angles) == 0:
        return None

    arr = np.array(angles)
    return {
        "mean_abs_rad": float(np.mean(arr)),
        "mean_abs_deg": float(np.mean(arr) * 180 / np.pi),
        "std_abs_rad": float(np.std(arr)),
        "n": len(angles),
    }


# ============================================================================
# B4 — velocity bins (Noldus "Split the track in bins based on velocity")
# ============================================================================


def compute_velocity_bins(
    df: pd.DataFrame,
    bin_edges: list[float] | None = None,
) -> dict | None:
    """Bin per-frame velocity into ranges and return time-ratio per bin.

    When invoked via the catalog, only the ``velocity`` column is exposed.
    The ``distance_moved`` fallback is available via direct API call.

    Args:
        df: DataFrame with ``velocity`` or ``distance_moved`` column.
        bin_edges: Velocity thresholds. Default: [0, 50, 100, 200, 400].
            Must have at least 2 elements. Values below the first edge
            are captured in a ``<{first}`` bin.

    Returns:
        Dict mapping bin label to ratio (0–1), or None when no velocity data.
        Bin labels: ``"<{first}"``, ``"{lo}-{hi}"``, ``"{last}+"``.
        Ratios always sum to 1.
    """
    if bin_edges is None:
        bin_edges = [0, 50, 100, 200, 400]

    if len(bin_edges) < 2:
        raise ValueError(f"bin_edges must have at least 2 elements, got {len(bin_edges)}")

    if "velocity" in df.columns:
        v = pd.to_numeric(df["velocity"], errors="coerce").dropna()
    elif "distance_moved" in df.columns:
        dt = _estimate_dt(df)
        if dt <= 0:
            return None
        v = pd.to_numeric(df["distance_moved"], errors="coerce").dropna() / dt
    else:
        return None

    if v.empty:
        return None

    result: dict[str, float] = {}

    # Leading bin for values below the first edge
    under_mask = v < bin_edges[0]
    result[f"<{bin_edges[0]}"] = float(under_mask.mean())

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (v >= lo) & (v < hi)
        result[f"{lo}-{hi}"] = float(mask.mean())

    over_mask = v >= bin_edges[-1]
    result[f"{bin_edges[-1]}+"] = float(over_mask.mean())

    return result


# ============================================================================
# B7 — cumulative distance (Noldus "Distance moved - Cumulative in last k samples")
# ============================================================================


def compute_cumulative_distance(
    df: pd.DataFrame,
    window_samples: int = 25,
) -> dict | None:
    """Cumulative distance moved over a sliding window (rolling sum).

    Frame-rate adaptive: *window_samples* is scaled from the 25-fps Noldus
    baseline to the actual frame rate via :func:`_estimate_dt`.

    Args:
        df: DataFrame with ``distance_moved`` column.
        window_samples: Window size in samples at 25 fps. Default 25 (= 1 s).

    Returns:
        Dict with keys mean, max, min, median, or None.
    """
    if "distance_moved" not in df.columns:
        return None

    dt = _estimate_dt(df)
    scale = 0.04 / dt if dt > 0 and dt != 0.04 else 1.0
    window = max(1, round(window_samples * scale))

    d = pd.to_numeric(df["distance_moved"], errors="coerce").fillna(0)
    cum = d.rolling(window=window, min_periods=1).sum()
    valid = cum.dropna()
    if valid.empty:
        return None

    return {
        "mean": float(valid.mean()),
        "max": float(valid.max()),
        "min": float(valid.min()),
        "median": float(valid.median()),
    }


# ============================================================================
# B8 — acceleration stats (Noldus "Acceleration - Smoothed")
# ============================================================================


def compute_acceleration_stats(
    df: pd.DataFrame,
    smooth_window: int = 5,
) -> dict | None:
    """Smoothed acceleration statistics.

    SMA-smooths velocity (window = *smooth_window*), then differentiates
    to get acceleration. Output unit matches the input velocity unit per
    second (e.g. mm/s² when velocity is in mm/s, cm/s² when in cm/s).

    When invoked via the catalog, only the ``velocity`` column is exposed.
    The ``distance_moved`` fallback is available via direct API call.

    Args:
        df: DataFrame with ``velocity`` or ``distance_moved`` column.
        smooth_window: SMA window size for velocity smoothing.

    Returns:
        Dict with keys mean, std, max, min, or None.
    """
    if "velocity" in df.columns:
        v = pd.to_numeric(df["velocity"], errors="coerce").dropna()
    elif "distance_moved" in df.columns:
        dt = _estimate_dt(df)
        if dt <= 0:
            return None
        v = pd.to_numeric(df["distance_moved"], errors="coerce").dropna() / dt
    else:
        return None

    if len(v) < 2:
        return None

    v_smooth = v.rolling(window=smooth_window, min_periods=1).mean()
    dt = _estimate_dt(df)
    if dt <= 0:
        return None
    acc = v_smooth.diff() / dt
    acc = acc.dropna()
    if acc.empty:
        return None

    return {
        "mean": float(acc.mean()),
        "std": float(acc.std()),
        "max": float(acc.max()),
        "min": float(acc.min()),
    }


# ============================================================================
# B5 — body length (Noldus "Body Length - Sum of segments")
# ============================================================================


def compute_body_length(df: pd.DataFrame) -> dict | None:
    """Body length as sum of Nose→Center + Center→Tail Euclidean distances.

    Requires all 6 body-point columns. Returns None when any column is
    missing (not all paradigms have multi-point body tracking).

    Returns:
        Dict with keys mean, std, min, max, median (same unit as
        coordinate columns, typically cm), or None.
    """
    required = ["x_nose", "y_nose", "x_center", "y_center", "x_tail", "y_tail"]
    if not all(c in df.columns for c in required):
        return None

    nose_to_center = np.sqrt(
        (df["x_nose"] - df["x_center"]) ** 2
        + (df["y_nose"] - df["y_center"]) ** 2
    )
    center_to_tail = np.sqrt(
        (df["x_center"] - df["x_tail"]) ** 2
        + (df["y_center"] - df["y_tail"]) ** 2
    )
    body_length = pd.to_numeric(nose_to_center + center_to_tail, errors="coerce")
    valid = body_length.dropna()
    if valid.empty:
        return None

    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()) if len(valid) > 1 else 0.0,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "median": float(valid.median()),
    }


# ============================================================================
# B10 — heading smoothed (Noldus "Heading - Smoothed")
# ============================================================================


def compute_heading_smoothed(
    df: pd.DataFrame,
    window: int = 5,
) -> dict | None:
    """Circular SMA on Direction (heading) column.

    Unwraps angles to handle the ±180° boundary, applies SMA, then re-wraps
    to [0, 2π). NaN gaps are forward-filled before unwrapping so they do not
    corrupt the angular continuity; only originally-valid frames contribute
    to the output statistics.

    Args:
        df: DataFrame with ``Direction`` column (radians).
        window: SMA window size. Default 5.

    Returns:
        Dict with keys mean_rad, circular_stdev_rad, resultant_length, n,
        or None when ``Direction`` column is missing or all-NaN.
    """
    if "Direction" not in df.columns:
        return None
    d = pd.to_numeric(df["Direction"], errors="coerce")
    valid_mask = d.notna()
    if not valid_mask.any():
        return None

    # Forward-fill NaN gaps so unwrap sees continuous angles
    d_filled = d.ffill().to_numpy(dtype=float)
    unwrapped = np.unwrap(d_filled)
    smoothed = pd.Series(unwrapped).rolling(window=window, min_periods=1).mean()
    smoothed_wrapped = smoothed % (2 * np.pi)

    # Only keep frames that were originally valid
    valid_smoothed = smoothed_wrapped[valid_mask.values]
    n = len(valid_smoothed)
    if n == 0:
        return None

    sin_sum = float(np.sin(valid_smoothed).sum())
    cos_sum = float(np.cos(valid_smoothed).sum())
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    mean_rad = float(np.arctan2(sin_sum, cos_sum)) % (2 * np.pi)
    circular_stdev = (
        float(np.sqrt(-2 * np.log(max(min(R / n, 1.0 - 1e-12), 1e-12))))
        if n > 0 and R > 0
        else float("inf")
    )

    return {
        "mean_rad": mean_rad,
        "circular_stdev_rad": circular_stdev,
        "resultant_length": float(R),
        "n": n,
    }
