"""Open Field Test 范式指标：中心区滞留 + 趋触性。"""

from __future__ import annotations


import numpy as np
import pandas as pd

from ethoinsight.metrics._common import _find_zone_column


# ============================================================================
# OFT zone column resolution
# ============================================================================


def _find_center_zone_column(
    df: pd.DataFrame, hint: str = "in_zone_center"
) -> str | None:
    """Find the column representing the center zone.

    Returns:
        Column name if an explicit center-zone column exists.
        None otherwise. Bare ``in_zone`` (without ``_center`` / ``_centre`` suffix)
        is NOT treated as center by default — list-name ambiguity should
        trigger an upstream user clarification (see 2026-05-13 feedback Q2).
    """
    # 1. Exact hint match
    if hint in df.columns:
        return hint
    # 2. Regex: any in_zone column that mentions center/centre, excluding wall/edge/periphery
    for col in df.columns:
        cl = col.lower()
        if not cl.startswith("in_zone"):
            continue
        if any(bad in cl for bad in ("wall", "edge", "peripher", "border", "outer")):
            continue
        if "center" in cl or "centre" in cl:
            return col
    return None


def _find_periphery_zone_column(df: pd.DataFrame) -> str | None:
    """Locate the OFT periphery / wall-zone indicator column.

    Recognises ``in_zone.*(peripher|edge|wall|border|outer)``.
    Returns None if no such column exists (caller may fall back to
    1 − center_time_ratio).
    """
    return _find_zone_column(df, r"in_zone.*(peripher|edge|wall|border|outer)")


# ============================================================================
# Open Field metrics
# ============================================================================


def compute_center_time_ratio(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> float | None:
    """Ratio of time spent in center zone.

    Resolves the center zone column via :func:`_find_center_zone_column`."""
    col = _find_center_zone_column(df, hint=center_zone)
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

    Priority:
    1. Explicit periphery / wall-zone column
    2. Complement of the center zone (1 − center_time_ratio)
    3. Geometric derivation from x/y coordinates + arena spec
    """
    # 1. Explicit periphery column
    col = _find_periphery_zone_column(df)
    if col is not None:
        series = df[col].dropna()
        if not series.empty:
            return float(series.mean())

    # 2. Complement of center zone — works when EthoVision only exports center
    center_ratio = compute_center_time_ratio(df)
    if center_ratio is not None:
        return float(1.0 - center_ratio)

    # 3. Compute from coordinates
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
        (x.loc[idx] - arena_center[0]) ** 2 + (y.loc[idx] - arena_center[1]) ** 2
    )
    threshold = arena_radius * (1 - periphery_fraction)
    return float((dist > threshold).mean())


def compute_center_distance_ratio(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> float | None:
    """Ratio of distance traveled inside center zone to total distance.

    Prefers EV19's ``distance_moved`` column (Euclidean per-frame distance).
    When missing, falls back to Euclidean displacement from ``x_center`` /
    ``y_center`` (√(dx² + dy²)), never Manhattan.

    Returns None when center zone column is unresolved or no distance data
    is available.
    """
    col = _find_center_zone_column(df, hint=center_zone)
    if col is None:
        return None

    mask = df[col] == 1
    if not mask.any():
        return 0.0

    if "distance_moved" in df.columns:
        total = df["distance_moved"].dropna().sum()
        if total <= 0:
            return 0.0
        center = df.loc[mask, "distance_moved"].dropna().sum()
        return float(center / total)

    if "x_center" not in df.columns or "y_center" not in df.columns:
        return None

    x = df["x_center"]
    y = df["y_center"]

    dx = x.diff()
    dy = y.diff()
    per_frame = np.sqrt(dx**2 + dy**2).dropna()

    common = per_frame.index.intersection(mask.index)
    total_dist = per_frame.loc[common].sum()
    center_dist = per_frame.loc[common][mask.loc[common]].sum()

    if total_dist == 0:
        return 0.0

    return float(center_dist / total_dist)


def compute_center_entry_count(
    df: pd.DataFrame,
    center_zone: str = "in_zone_center",
) -> int | None:
    """Number of entries into the center zone (0→1 transitions).

    First frame in center also counts as 1 entry.
    """
    col = _find_center_zone_column(df, hint=center_zone)
    if col is None:
        return None

    series = df[col].dropna()
    if series.empty:
        return 0

    vals = series.to_numpy(dtype=int)
    entries = 1 if vals[0] == 1 else 0
    transitions = (vals[1:] == 1) & (vals[:-1] == 0)
    return entries + int(transitions.sum())


def compute_center_time(df: pd.DataFrame) -> float | None:
    """Total time the subject spent in center zone (seconds).

    = center_time_ratio * total_duration

    Returns None if center column cannot be resolved.
    """
    ratio = compute_center_time_ratio(df)
    if ratio is None:
        return None
    if "trial_time" not in df.columns:
        return None
    duration = float(df["trial_time"].iloc[-1] - df["trial_time"].iloc[0])
    return ratio * duration


def compute_center_distance(df: pd.DataFrame) -> float | None:
    """Total distance moved while in center zone (cm).

    Accumulates ``distance_moved`` only at frames where the center-zone indicator is 1.
    Returns None if either column is missing.
    """
    if "distance_moved" not in df.columns:
        return None
    center_col = _find_center_zone_column(df)
    if center_col is None:
        return None
    mask = df[center_col].fillna(0) > 0
    return float(df.loc[mask, "distance_moved"].sum())
