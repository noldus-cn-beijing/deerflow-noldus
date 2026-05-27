"""范式无关的指标函数 + 共享 helper。"""

from __future__ import annotations

import os
import re

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


# ============================================================================
# Immobility analysis (shared by FST / TST)
# ============================================================================


def _find_mobility_column(df: pd.DataFrame) -> str | None:
    """Find the mobility/activity state column in a DataFrame.

    Checks common EthoVision column names in order:
    Mobility_State, Activity_State, mobility_state, activity_state.
    Returns the first match.
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
    # Fallback: any column whose name contains mobility or activity.
    for col in df.columns:
        if "mobility" in col.lower() or "activity" in col.lower():
            return col
    return None


def _resolve_immobile_series(
    df: pd.DataFrame, mobility_col: str | None
) -> tuple[pd.Series, int] | None:
    """Resolve the mobility column to a (series, immobile_value) pair.

    Handles two EthoVision export schemas:
    - Single combined column (``mobility_state``): 0=immobile, 1=mobile → immobile_value=0
    - One-hot immobile indicator (``mobility_state_immobile``): 1=immobile → immobile_value=1

    Returns None when no usable column is present.

    **重要假设（EV19 Mobility state 编码）**：
    EV19 ``Mobility state`` 列实际是 2/3/4 状态多分类（参见
    ev19-dependent-variables.md §14）。``immobile_value=0`` 只在 2-state 配置下
    正确；3-state（0=Inactive, 1=Moderately active, 2=Highly active）或 4-state
    下，Moderately active (1) / Active (2) 也是非 0 值，会被本函数错误归为 mobile。
    仅当列名含 ``immobile`` 时才确认为 one-hot 列。如用户数据来自 3/4-state 配置，
    需上游显式映射后再传入。
    """
    col = mobility_col or _find_mobility_column(df)
    if col is None or col not in df.columns:
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    immobile_value = 1 if "immobile" in col.lower() else 0
    return series, immobile_value


def _runs(arr, value=0):
    """Return list of (start_idx, end_idx) for consecutive runs of `value`.

    Used for immobility bout detection.
    """
    import numpy as np

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
) -> float | None:
    """Total immobility time (seconds).

    Sums the duration of all immobility bouts. Works with both the combined
    ``mobility_state`` column (0=immobile) and the one-hot
    ``mobility_state_immobile`` column (1=immobile).
    """
    resolved = _resolve_immobile_series(df, mobility_col)
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
) -> float | None:
    """Latency to first immobility bout (seconds).

    Returns the trial_time value of the first immobile frame, or None if
    the animal was never immobile.
    """
    resolved = _resolve_immobile_series(df, mobility_col)
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
) -> int | None:
    """Number of immobility bouts (run-length encoding).

    Each consecutive run of immobile frames counts as one bout.
    """
    resolved = _resolve_immobile_series(df, mobility_col)
    if resolved is None:
        return None
    series, immobile_value = resolved

    bouts = _runs(series, value=immobile_value)
    return len(bouts)


def extract_immobility_bouts(
    df: pd.DataFrame,
    mobility_col: str | None = None,
) -> list[tuple[float, float]]:
    """Extract (start_sec, end_sec) pairs for each immobility bout.

    Uses trial_time column to convert frame indices to seconds.
    Returns empty list if mobility column or trial_time is missing.
    """
    resolved = _resolve_immobile_series(df, mobility_col)
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
