"""Open Field Test 范式指标：中心区滞留 + 趋触性。"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ethoinsight.metrics._common import _find_zone_column


# ============================================================================
# Open Field metrics
# ============================================================================


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


def compute_center_distance_ratio(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> float | None:
    """Ratio of distance traveled inside center zone to total distance.

    Requires ``x_center`` and ``y_center`` columns for displacement calculation,
    plus a center zone column (0/1 indicator).
    """
    col = None
    if center_zone in df.columns:
        col = center_zone
    else:
        col = _find_zone_column(df, r"in_zone.*center")
    if col is None:
        return None
    if "x_center" not in df.columns or "y_center" not in df.columns:
        return None

    mask = df[col] == 1
    if not mask.any():
        return 0.0

    x = df["x_center"]
    y = df["y_center"]

    # Total displacement per frame
    dx_total = x.diff().abs().dropna()
    dy_total = y.diff().abs().dropna()

    # Center-only displacement (aligned by index)
    common = dx_total.index.intersection(mask.index)
    dx_center = dx_total.loc[common][mask.loc[common]]
    dy_center = dy_total.loc[common][mask.loc[common]]

    total_dist = dx_total.sum() + dy_total.sum()
    if total_dist == 0:
        return 0.0

    center_dist = dx_center.sum() + dy_center.sum()
    return float(center_dist / total_dist)


def compute_center_entry_count(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> int | None:
    """Number of entries into the center zone (0→1 transitions).

    First frame in center also counts as 1 entry.
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
        return 0

    vals = series.to_numpy(dtype=int)
    entries = 1 if vals[0] == 1 else 0
    transitions = (vals[1:] == 1) & (vals[:-1] == 0)
    return entries + int(transitions.sum())
