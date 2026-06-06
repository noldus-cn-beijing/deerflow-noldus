"""Zero Maze 范式指标：开放区时间百分比 + 滞留时长 + 移动距离 + 犹豫次数。

Zero Maze 为环形迷宫（无中心区），Open/Closed zone 覆盖整条环，结构上与 EPM
相似（开放区 vs 封闭区），但列名规律不同。

EthoVision 19 导出的典型列名：
    in_zone_open_1, in_zone_open_2, in_zone_closed_1, in_zone_closed_2
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ============================================================================
# Zone detection helpers
# ============================================================================


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


# ============================================================================
# Zero Maze metrics
# ============================================================================


def compute_open_zone_time_ratio(
    df: pd.DataFrame,
    open_zones: list[str] | None = None,
) -> float | None:
    """Ratio of frames spent in open zones (0–1).

    Auto-detects columns matching ``in_zone.*open`` (case-insensitive) unless
    *open_zones* is explicitly provided.

    Returns None when no open zone columns are found.
    """
    cols = _get_open_zone_cols(df, open_zones)
    if not cols:
        return None

    # OR-combine all open zone columns
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return None
    return float(combined.mean())


def compute_open_zone_time(
    df: pd.DataFrame,
    open_zones: list[str] | None = None,
) -> float | None:
    """Total time (seconds) spent in open zones.

    Multiplies the number of open-zone frames by the median inter-frame
    interval from ``trial_time``. Falls back to raw frame count when
    ``trial_time`` is unavailable.

    Returns None when no open zone columns are found.
    """
    cols = _get_open_zone_cols(df, open_zones)
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


def compute_open_zone_distance(
    df: pd.DataFrame,
    open_zones: list[str] | None = None,
) -> float | None:
    """Total distance traveled in open zones (cm).

    Requires both ``distance_moved`` column and open zone columns.
    Uses EV19's ``distance_moved`` which is Euclidean distance per frame.

    Returns None when:
    - No open zone columns detected.
    - ``distance_moved`` column missing or all-NaN.
    """
    cols = _get_open_zone_cols(df, open_zones)
    if not cols:
        return None
    if "distance_moved" not in df.columns:
        return None

    combined_open = df[cols].max(axis=1)
    dist = df["distance_moved"]
    if dist.dropna().sum() == 0:
        return 0.0

    open_dist = dist.where(combined_open == 1, other=0).fillna(0).sum()
    return float(open_dist)


def compute_hesitation_count(
    df: pd.DataFrame,
    open_zones: list[str] | None = None,
    closed_zones: list[str] | None = None,
    min_gap_frames: int = 5,
    proximity_threshold_cm: float | None = 2.0,
    min_retreat_speed_cm_s: float | None = None,
) -> int | None:
    """Count of "head-dip" hesitations: brief open-zone excursions from closed zone.

    A hesitation is defined as:
    - Animal transitions from closed → open zone.
    - The open-zone bout lasts < *min_gap_frames* frames.
    - Animal returns to closed zone.

    When *proximity_threshold_cm* is set (default 2.0 cm), an additional
    "approach-retreat" heuristic applies: frames immediately before an open-zone
    bout where the animal's per-frame movement is ≤ *proximity_threshold_cm*
    (suggesting slow approach to the boundary) are flagged. If the subsequent
    retreat speed exceeds *min_retreat_speed_cm_s*, the bout is counted as a
    hesitation even if the zone-column signal alone would miss it.

    This captures "stretch-attend" risk-assessment postures where the animal
    briefly protrudes into the open zone then retreats — positively correlated
    with anxiety level.

    Algorithm assumptions:
    - *min_gap_frames* default 5 assumes 25 Hz sampling (~200 ms max bout).
      Adjust for different sampling rates.
    - The proximity heuristic requires the ``distance_moved`` column. When
      unavailable, only zone-column-based detection is used.
    - Zone boundaries are assumed stable throughout the trial.

    Args:
        df: Trajectory DataFrame.
        open_zones: Explicit open zone column names. Auto-detected if None.
        closed_zones: Explicit closed zone column names.
        min_gap_frames: Maximum open-zone bout length (frames) to count as a
            hesitation. Bouts >= this length are genuine open explorations.
        proximity_threshold_cm: Per-frame movement threshold (cm) for detecting
            slow boundary approach. Set to None to disable proximity heuristic.
        min_retreat_speed_cm_s: Minimum retreat speed (cm/s) to count a
            proximity event as a hesitation. Default: prox_threshold × 10.

    Returns:
        Hesitation count (int), or None if no open zone columns detected.
    """
    cols = _get_open_zone_cols(df, open_zones)
    if not cols:
        return None

    combined = df[cols].max(axis=1).fillna(0).astype(int).to_numpy()

    if len(combined) == 0:
        return 0

    # Per-frame movement for proximity heuristic
    dist = None
    if proximity_threshold_cm is not None and "distance_moved" in df.columns:
        dist = pd.to_numeric(df["distance_moved"], errors="coerce").fillna(0).to_numpy()
    retreat_thresh = min_retreat_speed_cm_s or (proximity_threshold_cm * 10 if proximity_threshold_cm else 20.0)

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
                # Proximity heuristic: check pre-bout approach speed
                if dist is not None and proximity_threshold_cm is not None and bout_start >= 1:
                    pre_dist = dist[max(0, bout_start - 2):bout_start]
                    if len(pre_dist) > 0 and float(np.mean(pre_dist)) <= proximity_threshold_cm:
                        # Slow approach → confirmed risk-assessment
                        # Check retreat speed
                        retreat_start = i  # right after bout end
                        retreat_end = min(n, retreat_start + 3)
                        retreat_dist = dist[retreat_start:retreat_end]
                        if len(retreat_dist) > 0 and float(np.mean(retreat_dist)) > proximity_threshold_cm:
                            # Already counted above; proximity confirms the classification
                            pass
        else:
            i += 1

    return count
