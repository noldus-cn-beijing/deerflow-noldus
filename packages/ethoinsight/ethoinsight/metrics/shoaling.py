"""Shoaling 范式指标：群体游动相关。"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ethoinsight.metrics._common import _align_subjects_xy


# ============================================================================
# Shoaling metrics
# ============================================================================


def compute_inter_individual_distance(
    subjects: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Inter-individual distance (IID) over time.

    For each timepoint, computes all pairwise Euclidean distances
    between subjects and summarises as mean, std, min, max.

    Returns DataFrame with columns:
        trial_time, mean_iid, std_iid, min_iid, max_iid
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size == 0:
        return None

    n_subjects = coords.shape[0]
    pair_indices = list(combinations(range(n_subjects), 2))
    n_pairs = len(pair_indices)

    # Compute pairwise distances: shape (n_pairs, n_timepoints)
    dists = np.empty((n_pairs, len(times)))
    for k, (i, j) in enumerate(pair_indices):
        diff = coords[i] - coords[j]
        dists[k] = np.sqrt((diff**2).sum(axis=1))

    return pd.DataFrame(
        {
            "trial_time": times,
            "mean_iid": dists.mean(axis=0),
            "std_iid": dists.std(axis=0),
            "min_iid": dists.min(axis=0),
            "max_iid": dists.max(axis=0),
        }
    )


def compute_nearest_neighbor_distance(
    subjects: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Nearest-neighbor distance (NND) per subject over time.

    For each subject at each timepoint, finds the distance to the
    closest other subject.

    Returns DataFrame with columns:
        trial_time, subject, nnd
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size == 0:
        return None

    n_sub = coords.shape[0]
    subject_names = list(subjects.keys())[:n_sub]

    rows = []
    for i in range(n_sub):
        # Distance from subject i to all others: (n_others, n_time)
        others = [j for j in range(n_sub) if j != i]
        other_coords = coords[others]  # (n_others, n_time, 2)
        diff = other_coords - coords[i][np.newaxis, :, :]  # broadcast
        dist_to_others = np.sqrt((diff**2).sum(axis=2))  # (n_others, n_time)
        nnd = dist_to_others.min(axis=0)  # (n_time,)
        for t_idx, t in enumerate(times):
            rows.append(
                {"trial_time": t, "subject": subject_names[i], "nnd": float(nnd[t_idx])}
            )

    return pd.DataFrame(rows)


def compute_group_polarity(
    subjects: dict[str, pd.DataFrame],
    smooth_window: int = 5,
) -> pd.DataFrame | None:
    """Group polarisation (alignment of movement directions) over time.

    Computes heading from consecutive (dx, dy), smooths, then calculates
    mean resultant length R = |mean(e^{i*theta})| at each timepoint.
    R in [0, 1]: 0 = random directions, 1 = perfectly aligned.

    Returns DataFrame with columns: trial_time, polarity
    """
    times, coords = _align_subjects_xy(subjects)
    if times.size < smooth_window + 2:
        return None

    n_sub = coords.shape[0]

    # Compute heading angles from consecutive position differences
    dx = np.diff(coords[:, :, 0], axis=1)  # (n_sub, n_time-1)
    dy = np.diff(coords[:, :, 1], axis=1)
    theta = np.arctan2(dy, dx)  # (n_sub, n_time-1)

    # Smooth with rolling mean on unit vectors (circular smoothing)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    kernel = np.ones(smooth_window) / smooth_window
    cos_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), 1, cos_theta
    )
    sin_smooth = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), 1, sin_theta
    )
    theta_smooth = np.arctan2(sin_smooth, cos_smooth)

    # Mean resultant length at each timepoint
    n_valid = theta_smooth.shape[1]
    mean_cos = theta_smooth.copy()
    mean_sin = theta_smooth.copy()
    # Convert back to unit vectors for averaging across subjects
    cos_vals = np.cos(theta_smooth)  # (n_sub, n_valid)
    sin_vals = np.sin(theta_smooth)
    mean_r = np.sqrt(cos_vals.mean(axis=0) ** 2 + sin_vals.mean(axis=0) ** 2)

    # Trim time to match valid window
    offset = (len(times) - 1) - n_valid  # diff loses 1, convolve valid loses (window-1)
    start = 1 + (smooth_window - 1)  # skip first diff + convolution padding
    valid_times = times[start : start + n_valid]

    if len(valid_times) != len(mean_r):
        # Edge case: just use first n_valid times
        valid_times = times[:n_valid]

    return pd.DataFrame(
        {
            "trial_time": valid_times,
            "polarity": mean_r,
        }
    )
