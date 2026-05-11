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
# Shoaling helper (shared with shoaling.py)
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
