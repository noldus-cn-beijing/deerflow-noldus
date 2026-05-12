"""Tests for FST (Forced Swim Test) immobility metric functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics.fst import (
    compute_immobility_time_fst,
    compute_immobility_latency_fst,
    compute_immobility_bout_count_fst,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_immobility_df(n_frames=100, *, pattern=None, mobility_col="mobility_state"):
    """Synthetic FST DataFrame with trial_time and Mobility_State columns."""
    if pattern is None:
        # 3 bouts: [0]*10, [0]*10, [0]*5
        pattern = [0] * 10 + [1] * 30 + [0] * 10 + [1] * 40 + [0] * 5 + [1] * 5
    # Truncate or pad with 1s (mobile) to reach n_frames
    pattern = list(pattern[:n_frames]) + [1] * max(0, n_frames - len(pattern))
    return pd.DataFrame({
        "trial_time": np.arange(n_frames) * 0.04,
        mobility_col: pattern,
    })


# ============================================================================
# compute_immobility_time_fst
# ============================================================================


class TestComputeImmobilityTimeFst:
    """Tests for compute_immobility_time_fst."""

    def test_total_immobility_time_calculation(self):
        # pattern: 10 immobile + 30 mobile + 10 immobile + 40 mobile + 5 immobile + 5 mobile
        # total immobile frames = 25; dt = 0.04s → 25 * 0.04 = 1.0s
        df = _make_immobility_df(n_frames=100)
        result = compute_immobility_time_fst(df)
        assert result == pytest.approx(25 * 0.04)

    def test_no_mobility_column_returns_none(self):
        df = pd.DataFrame({"trial_time": [0.0, 0.04, 0.08], "x_center": [1, 2, 3]})
        result = compute_immobility_time_fst(df)
        assert result is None

    def test_all_mobile_returns_zero(self):
        df = _make_immobility_df(n_frames=50, pattern=[1] * 50)
        result = compute_immobility_time_fst(df)
        assert result == pytest.approx(0.0)

    def test_all_immobile(self):
        # all 50 frames immobile → 50 * 0.04 = 2.0s
        df = _make_immobility_df(n_frames=50, pattern=[0] * 50)
        result = compute_immobility_time_fst(df)
        assert result == pytest.approx(50 * 0.04)

    def test_no_trial_time_returns_frame_count(self):
        # Without trial_time, should return raw frame count
        df = pd.DataFrame({
            "mobility_state": [0, 0, 1, 1, 0, 0, 0, 1],
        })
        result = compute_immobility_time_fst(df)
        assert result == pytest.approx(5.0)  # 5 immobile frames

    def test_single_immobile_frame(self):
        df = _make_immobility_df(n_frames=10, pattern=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        result = compute_immobility_time_fst(df)
        assert result == pytest.approx(1 * 0.04)


# ============================================================================
# compute_immobility_latency_fst
# ============================================================================


class TestComputeImmobilityLatencyFst:
    """Tests for compute_immobility_latency_fst."""

    def test_latency_to_first_immobility(self):
        # First immobile frame is frame 10 (0-indexed), trial_time = 10 * 0.04 = 0.4s
        df = _make_immobility_df(n_frames=100)
        result = compute_immobility_latency_fst(df)
        assert result == pytest.approx(0.0)  # frame 0 is immobile (pattern starts with 0s)

    def test_latency_when_first_bout_starts_later(self):
        # 10 mobile frames, then immobile
        pattern = [1] * 10 + [0] * 10 + [1] * 80
        df = _make_immobility_df(n_frames=100, pattern=pattern)
        result = compute_immobility_latency_fst(df)
        assert result == pytest.approx(10 * 0.04)

    def test_never_immobile_returns_none(self):
        df = _make_immobility_df(n_frames=50, pattern=[1] * 50)
        result = compute_immobility_latency_fst(df)
        assert result is None

    def test_starts_immobile_returns_zero(self):
        pattern = [0] * 5 + [1] * 45
        df = _make_immobility_df(n_frames=50, pattern=pattern)
        result = compute_immobility_latency_fst(df)
        assert result == pytest.approx(0.0)

    def test_no_mobility_column_returns_none(self):
        df = pd.DataFrame({"trial_time": [0.0, 0.04, 0.08]})
        result = compute_immobility_latency_fst(df)
        assert result is None


# ============================================================================
# compute_immobility_bout_count_fst
# ============================================================================


class TestComputeImmobilityBoutCountFst:
    """Tests for compute_immobility_bout_count_fst."""

    def test_counts_three_bouts(self):
        # pattern has 3 immobility bouts
        df = _make_immobility_df(n_frames=100)
        result = compute_immobility_bout_count_fst(df)
        assert result == 3

    def test_no_immobility_returns_zero(self):
        df = _make_immobility_df(n_frames=50, pattern=[1] * 50)
        result = compute_immobility_bout_count_fst(df)
        assert result == 0

    def test_single_bout(self):
        pattern = [1] * 20 + [0] * 10 + [1] * 20
        df = _make_immobility_df(n_frames=50, pattern=pattern)
        result = compute_immobility_bout_count_fst(df)
        assert result == 1

    def test_alternating_bouts(self):
        # 0,1,0,1,0 → 3 bouts
        pattern = [0, 1, 0, 1, 0]
        df = _make_immobility_df(n_frames=5, pattern=pattern)
        result = compute_immobility_bout_count_fst(df)
        assert result == 3

    def test_no_mobility_column_returns_none(self):
        df = pd.DataFrame({"trial_time": [0.0, 0.04, 0.08]})
        result = compute_immobility_bout_count_fst(df)
        assert result is None

    def test_all_immobile_is_one_bout(self):
        df = _make_immobility_df(n_frames=20, pattern=[0] * 20)
        result = compute_immobility_bout_count_fst(df)
        assert result == 1

    def test_rle_does_not_merge_separated_bouts(self):
        # 0→1→0 should be two separate bouts, not one
        pattern = [0] * 5 + [1] * 1 + [0] * 5
        df = _make_immobility_df(n_frames=11, pattern=pattern)
        result = compute_immobility_bout_count_fst(df)
        assert result == 2
