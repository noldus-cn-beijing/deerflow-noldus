"""Tests for OFT (Open Field Test) metric functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
    compute_center_distance_ratio,
    compute_center_entry_count,
)


def _make_oft_df(n_frames: int = 100, *, center_pattern: list[int] | None = None, seed: int = 42) -> pd.DataFrame:
    """Synthetic OFT DataFrame with controllable center presence."""
    rng = np.random.default_rng(seed)
    if center_pattern is None:
        center_pattern = [1] * 20 + [0] * 80
    df = pd.DataFrame({
        "trial_time": np.arange(n_frames) * 0.04,
        "x_center": rng.uniform(-10, 10, n_frames),
        "y_center": rng.uniform(-10, 10, n_frames),
        "in_zone_center": center_pattern[:n_frames] + [0] * max(0, n_frames - len(center_pattern)),
    })
    return df


class TestComputeCenterDistanceRatio:
    def test_center_distance_ratio_range(self):
        """Ratio should be between 0 and 1."""
        df = _make_oft_df(100)
        result = compute_center_distance_ratio(df)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_no_center_column_returns_none(self):
        df = pd.DataFrame({"x_center": [1, 2, 3], "y_center": [1, 2, 3]})
        assert compute_center_distance_ratio(df) is None

    def test_all_center_returns_one(self):
        """If all frames are in center, 100% of distance is center distance."""
        df = _make_oft_df(50, center_pattern=[1] * 50)
        result = compute_center_distance_ratio(df)
        assert result == pytest.approx(1.0)

    def test_no_center_frames_returns_zero(self):
        df = _make_oft_df(50, center_pattern=[0] * 50)
        result = compute_center_distance_ratio(df)
        assert result == 0.0


class TestComputeCenterEntryCount:
    def test_no_center_presence_returns_zero(self):
        df = _make_oft_df(100, center_pattern=[0] * 100)
        assert compute_center_entry_count(df) == 0

    def test_single_entry(self):
        df = _make_oft_df(100, center_pattern=[0] * 20 + [1] * 30 + [0] * 50)
        assert compute_center_entry_count(df) == 1

    def test_multiple_entries(self):
        df = _make_oft_df(96, center_pattern=[1, 0, 1, 0, 1, 0] * 16)
        assert compute_center_entry_count(df) == 48

    def test_starts_in_center_counts_as_entry(self):
        df = _make_oft_df(10, center_pattern=[1] * 5 + [0] * 5)
        assert compute_center_entry_count(df) == 1

    def test_no_center_column_returns_none(self):
        df = pd.DataFrame({"x_center": [1, 2], "y_center": [1, 2]})
        assert compute_center_entry_count(df) is None
