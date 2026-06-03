"""Tests for Phase 2 signal distribution functions:
- pendulum_periodicity_series (_pendulum.py)
- _compute_distribution_stats (_common.py)
- _resolve_immobile_from_velocity(return_signal=True) (_common.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics._common import (
    _compute_distribution_stats,
    _resolve_immobile_from_velocity,
)


# ============================================================================
# pendulum_periodicity_series
# ============================================================================


class TestPendulumPeriodicitySeries:
    def test_returns_float_array(self):
        from ethoinsight.metrics._pendulum import pendulum_periodicity_series

        activity = np.full(200, 0.5)
        result = pendulum_periodicity_series(activity, dt=0.04)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        assert len(result) == 200

    def test_values_in_zero_one_range(self):
        from ethoinsight.metrics._pendulum import pendulum_periodicity_series

        rng = np.random.default_rng(42)
        activity = rng.uniform(0, 5, 300)
        result = pendulum_periodicity_series(activity, dt=0.04)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_consistent_with_detect_pendulum(self):
        """periodicity values must match detect_pendulum output."""
        from ethoinsight.metrics._pendulum import detect_pendulum, pendulum_periodicity_series

        rng = np.random.default_rng(99)
        activity = rng.uniform(0, 3, 200)
        dt = 0.04
        results = detect_pendulum(activity, dt)
        expected_periodicity = [r["periodicity"] for r in results]
        actual = pendulum_periodicity_series(activity, dt)
        np.testing.assert_allclose(actual, expected_periodicity, atol=1e-10)

    def test_nan_activity_produces_zero_periodicity(self):
        from ethoinsight.metrics._pendulum import pendulum_periodicity_series

        activity = np.array([np.nan] * 10 + [0.5] * 90)
        result = pendulum_periodicity_series(activity, dt=0.04)
        assert len(result) == 100
        # NaN frames → detect_pendulum assigns periodicity=0.0
        assert result[0] == 0.0

    def test_kwargs_forwarded(self):
        """Custom pendulum parameters are forwarded to detect_pendulum."""
        from ethoinsight.metrics._pendulum import pendulum_periodicity_series

        activity = np.full(200, 0.5)
        # Different smooth_window should not crash
        result = pendulum_periodicity_series(activity, dt=0.04, pendulum_smooth_window=5)
        assert len(result) == 200


# ============================================================================
# _compute_distribution_stats
# ============================================================================


class TestComputeDistributionStats:
    def test_basic_statistics(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = _compute_distribution_stats(vals, "periodicity")
        assert result["signal_key"] == "periodicity"
        assert result["n_frames"] == 10
        assert result["max"] == 10.0
        assert result["median"] == 5.5
        # p10 ≈ 1.9, p90 ≈ 9.1 (numpy percentile interpolation)
        assert 0 < result["p10"] < 3
        assert 7 < result["p90"] <= 10

    def test_nan_values_excluded(self):
        vals = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = _compute_distribution_stats(vals, "velocity")
        assert result["n_frames"] == 3
        assert result["max"] == 5.0
        assert result["median"] == 3.0
        assert result["signal_key"] == "velocity"

    def test_empty_array(self):
        result = _compute_distribution_stats(np.array([], dtype=float), "periodicity")
        assert result["n_frames"] == 0
        assert result["signal_key"] == "periodicity"
        assert "p10" not in result
        assert "median" not in result

    def test_all_nan_array(self):
        vals = np.full(10, np.nan)
        result = _compute_distribution_stats(vals, "velocity")
        assert result["n_frames"] == 0
        assert result["signal_key"] == "velocity"

    def test_single_value(self):
        vals = np.array([3.14])
        result = _compute_distribution_stats(vals, "periodicity")
        assert result["n_frames"] == 1
        assert result["p10"] == 3.14
        assert result["p90"] == 3.14
        assert result["median"] == 3.14
        assert result["max"] == 3.14

    def test_output_types(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = _compute_distribution_stats(vals, "periodicity")
        assert isinstance(result["p10"], float)
        assert isinstance(result["p90"], float)
        assert isinstance(result["median"], float)
        assert isinstance(result["max"], float)
        assert isinstance(result["n_frames"], int)
        assert isinstance(result["signal_key"], str)


# ============================================================================
# _resolve_immobile_from_velocity(return_signal=True)
# ============================================================================


class TestResolveImmobileFromVelocityWithSignal:
    def _make_stationary_df(self, n: int = 200) -> pd.DataFrame:
        """DataFrame with stationary subject at end (velocity → 0)."""
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.uniform(-0.5, 0.5, n))
        y = np.cumsum(rng.uniform(-0.5, 0.5, n))
        # Last 100 frames: near-zero movement
        x[-100:] = x[-100]
        y[-100:] = y[-100]
        return pd.DataFrame({
            "trial_time": np.arange(n) * 0.04,
            "x_center": x,
            "y_center": y,
        })

    def test_return_signal_false_gives_two_tuple(self):
        """Default return_signal=False returns (series, immobile_value)."""
        df = self._make_stationary_df()
        result = _resolve_immobile_from_velocity(df, return_signal=False)
        assert result is not None
        assert len(result) == 2  # (series, immobile_value)

    def test_return_signal_true_gives_three_tuple(self):
        """return_signal=True returns (series, immobile_value, velocity_arr)."""
        df = self._make_stationary_df()
        result = _resolve_immobile_from_velocity(df, return_signal=True)
        assert result is not None
        assert len(result) == 3  # (series, immobile_value, velocity_arr)
        series, immobile_value, velocity_arr = result
        assert isinstance(series, pd.Series)
        assert immobile_value == 1
        assert isinstance(velocity_arr, np.ndarray)
        assert len(velocity_arr) == len(df)
        assert velocity_arr.dtype == float

    def test_velocity_arr_has_nan_at_frame_zero(self):
        """Frame 0 has no previous frame → velocity should be NaN."""
        df = self._make_stationary_df(200)
        result = _resolve_immobile_from_velocity(df, return_signal=True)
        assert result is not None
        _, _, velocity_arr = result
        assert np.isnan(velocity_arr[0])

    def test_velocity_arr_nonnegative(self):
        """All non-NaN velocity values should be ≥ 0."""
        df = self._make_stationary_df()
        result = _resolve_immobile_from_velocity(df, return_signal=True)
        assert result is not None
        _, _, velocity_arr = result
        valid = velocity_arr[~np.isnan(velocity_arr)]
        assert np.all(valid >= 0)

    def test_velocity_arr_compatible_with_distribution_stats(self):
        """Velocity array should work directly with _compute_distribution_stats."""
        df = self._make_stationary_df()
        result = _resolve_immobile_from_velocity(df, return_signal=True)
        assert result is not None
        _, _, velocity_arr = result
        stats = _compute_distribution_stats(velocity_arr, "velocity")
        assert stats["signal_key"] == "velocity"
        assert stats["n_frames"] > 0

    def test_no_xy_columns_returns_none(self):
        df = pd.DataFrame({"trial_time": np.arange(100) * 0.04})
        result = _resolve_immobile_from_velocity(df, return_signal=True)
        assert result is None

    def test_backward_compat_default_is_false(self):
        """Calling without return_signal still returns 2-tuple (backward compat)."""
        df = self._make_stationary_df()
        result = _resolve_immobile_from_velocity(df)
        assert result is not None
        assert len(result) == 2
