"""Zero Maze 范式指标：开放区时间百分比 + 滞留时长 + 移动距离 + 犹豫次数。

Zero Maze 为环形迷宫（无中心区），Open/Closed zone 覆盖整条环，结构上与 EPM
相似（开放区 vs 封闭区），但列名规律不同。

EthoVision 19 导出的典型列名：
    in_zone_open_1, in_zone_open_2, in_zone_closed_1, in_zone_closed_2
"""

from __future__ import annotations

import re

import pandas as pd


# ============================================================================
# Zone detection helpers
# ============================================================================


def _get_open_zone_cols(df: pd.DataFrame, open_zones: list[str] | None) -> list[str]:
    """Return open zone column names, auto-detecting if not provided."""
    if open_zones:
        return [c for c in open_zones if c in df.columns]
    return [c for c in df.columns if re.search(r"in_zone.*open", c, re.I)]


def _get_closed_zone_cols(
    df: pd.DataFrame, closed_zones: list[str] | None
) -> list[str]:
    """Return closed zone column names, auto-detecting if not provided."""
    if closed_zones:
        return [c for c in closed_zones if c in df.columns]
    return [c for c in df.columns if re.search(r"in_zone.*closed", c, re.I)]


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
) -> int | None:
    """Count of "head-dip" hesitations: brief open-zone excursions from closed zone.

    A hesitation is defined as:
    - Animal transitions from closed → open zone.
    - The open-zone bout lasts < *min_gap_frames* frames.
    - Animal returns to closed zone.

    This captures "risk-assessment" behavior where the animal briefly protrudes
    into the open zone then retreats — positively correlated with anxiety level.

    Args:
        df: Trajectory DataFrame.
        open_zones: Explicit open zone column names. Auto-detected if None.
        closed_zones: Explicit closed zone column names (for future multi-zone).
        min_gap_frames: Maximum open-zone bout length (frames) to count as a
            hesitation. Bouts >= this length are genuine open explorations.

    Returns:
        Hesitation count (int), or None if no open zone columns detected.
    """
    cols = _get_open_zone_cols(df, open_zones)
    if not cols:
        return None

    # OR-combine: in open zone if any open zone column == 1
    combined = df[cols].max(axis=1).fillna(0).astype(int).to_numpy()

    if len(combined) == 0:
        return 0

    count = 0
    i = 0
    n = len(combined)
    while i < n:
        if combined[i] == 1:
            # Start of an open-zone bout
            bout_start = i
            while i < n and combined[i] == 1:
                i += 1
            bout_len = i - bout_start
            # Count as hesitation only if bout ends before reaching min_gap_frames
            # AND the bout is not at the very end of the trial (returns to closed)
            if bout_len < min_gap_frames and i < n:
                count += 1
        else:
            i += 1

    return count
