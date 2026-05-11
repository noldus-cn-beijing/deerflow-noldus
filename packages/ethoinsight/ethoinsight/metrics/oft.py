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
