"""Light-Dark Box (明暗箱) 范式指标：明箱时间占比 + 穿梭次数 + 潜伏期。"""

from __future__ import annotations

import pandas as pd


# ============================================================================
# LDB helpers
# ============================================================================


def _get_first_light_index(df: pd.DataFrame, light_zone: str) -> int | None:
    """Return the integer index of the first frame where light_zone == 1.

    Returns None if the animal never enters the light zone.
    """
    if light_zone not in df.columns:
        return None
    light_vals = df[light_zone].to_numpy()
    for i, v in enumerate(light_vals):
        if v == 1:
            return i
    return None


# ============================================================================
# LDB metrics
# ============================================================================


def compute_light_time_ratio(
    df: pd.DataFrame,
    light_zone: str = "in_zone_light",
) -> float | None:
    """Ratio of time spent in the light zone (明箱时间百分比).

    Computed as the mean of the light zone indicator column (0/1 per frame).
    Returns None when the required column is missing.
    """
    if light_zone not in df.columns:
        return None
    vals = df[light_zone].dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def compute_transition_count(
    df: pd.DataFrame,
    light_zone: str = "in_zone_light",
    dark_zone: str = "in_zone_dark",
) -> int | None:
    """Number of zone transitions between light and dark zones (穿梭次数).

    A transition is any single unidirectional crossing:
    - light→dark: 0→1 in the dark zone column
    - dark→light: 0→1 in the light zone column

    Both columns are summed so that each physical crossing is counted once.
    Returns None when neither column exists.
    """
    has_light = light_zone in df.columns
    has_dark = dark_zone in df.columns

    if not has_light and not has_dark:
        return None

    def _count_0_to_1(series: pd.Series) -> int:
        vals = series.dropna().to_numpy(dtype=int)
        if len(vals) < 2:
            return 0
        return int(((vals[1:] == 1) & (vals[:-1] == 0)).sum())

    total = 0
    if has_light:
        total += _count_0_to_1(df[light_zone])
    if has_dark:
        total += _count_0_to_1(df[dark_zone])
    return total


def compute_light_latency(
    df: pd.DataFrame,
    light_zone: str = "in_zone_light",
) -> float | None:
    """Latency to first enter the light zone in seconds (潜伏期).

    - If ``trial_time`` column exists: returns the trial_time value at the
      first frame where light_zone == 1.
    - Otherwise: returns the integer frame index (count-based fallback).
    - Returns None if the light zone column is missing or the animal never
      enters the light zone.
    """
    idx = _get_first_light_index(df, light_zone)
    if idx is None:
        return None
    if "trial_time" in df.columns:
        return float(df["trial_time"].iloc[idx])
    return float(idx)
