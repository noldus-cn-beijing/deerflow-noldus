"""Elevated Plus Maze 范式指标：开臂滞留 + 开臂进入 + 总进臂次数。"""

from __future__ import annotations

import re

import pandas as pd


# ============================================================================
# EPM helpers
# ============================================================================


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


def _prefer_center_suffix(cols: list[str]) -> list[str]:
    """Prefer columns with ``center`` suffix (gold standard for arm entry in EPM).

    Falls back to nose/tail/all variants only when no center column exists.
    Reference: Pellow et al. 1985; ev19-dependent-variables.md §10.
    """
    center_cols = [c for c in cols if re.search(r"center", c, re.I)]
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
        cols = _get_open_zone_cols(df)

    if not cols:
        return None

    # Combine: in open arm if any open arm zone column == 1
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return None
    return float(combined.mean())


def compute_open_arm_entry_count(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
) -> int | None:
    """Number of entries into open arms (0→1 transitions)."""
    if open_arm_zones:
        cols = [c for c in open_arm_zones if c in df.columns]
    else:
        cols = _get_open_zone_cols(df)
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
        cols = _get_open_zone_cols(df)
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
    open_cols = _get_open_zone_cols(df)
    closed_cols = _get_closed_zone_cols(df)
    if not open_cols and not closed_cols:
        return None
    open_entries = _count_zone_entries(df, open_cols) or 0
    closed_entries = _count_zone_entries(df, closed_cols) or 0
    return open_entries + closed_entries
