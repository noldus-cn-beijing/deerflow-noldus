"""Tests for pendulum-based immobility detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics._pendulum import detect_pendulum, pendulum_immobility_series


class TestDetectPendulum:
    def test_constant_low_activity_is_still(self):
        """Activity ~0 for entire trial → all frames classified as still."""
        activity = np.full(500, 0.01)
        results = detect_pendulum(activity, dt=0.04)
        # After warm-up (ANALYSIS_WINDOW=25) + duration filter, should be still (0)
        states = [r["state"] for r in results]
        # First ANALYSIS_WINDOW frames default to struggling
        assert all(s == 0 for s in states[100:]), (
            f"Expected still after warm-up, got states[100:]={states[100:120]}..."
        )

    def test_high_activity_is_struggling(self):
        """Activity consistently high → struggling."""
        activity = np.full(500, 5.0)
        results = detect_pendulum(activity, dt=0.04)
        states = [r["state"] for r in results]
        assert all(s == 1 for s in states[100:])

    def test_periodic_signal_detected_as_pendulum(self):
        """A clean sine wave should trigger periodicity detection."""
        t = np.arange(200) * 0.04
        # Sine wave with amplitude 2, offset 1 — crosses thresholds
        activity = 1.0 + 2.0 * np.sin(2 * np.pi * t / 0.3)
        results = detect_pendulum(activity, dt=0.04)
        pendulum_frames = sum(1 for r in results if r["is_pendulum"])
        assert pendulum_frames > 0, "Expected some pendulum frames from periodic signal"

    def test_nan_handling(self):
        """NaN frames should not crash; they get default state=0."""
        activity = np.array([np.nan] * 10 + [1.0] * 490)
        results = detect_pendulum(activity, dt=0.04)
        assert len(results) == 500

    def test_returns_correct_structure(self):
        """Each result dict has the expected keys and types."""
        activity = np.random.default_rng(0).uniform(0, 3, 200)
        results = detect_pendulum(activity, dt=0.04)
        for r in results:
            assert "state" in r
            assert "periodicity" in r
            assert "is_pendulum" in r
            assert r["state"] in (0, 1)
            assert 0.0 <= r["periodicity"] <= 1.0
            assert isinstance(r["is_pendulum"], bool)

    def test_periodicity_range(self):
        """Periodicity values are within [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            activity = rng.uniform(0, 3, 300)
            results = detect_pendulum(activity, dt=0.04)
            for r in results:
                assert 0.0 <= r["periodicity"] <= 1.0


class TestPendulumImmobilitySeries:
    def test_output_is_0_or_1(self):
        activity = np.random.default_rng(1).uniform(0, 5, 300)
        result = pendulum_immobility_series(activity, dt=0.04)
        assert result.dtype == int
        assert set(np.unique(result)).issubset({0, 1})

    def test_same_length_as_input(self):
        activity = np.full(100, 0.5)
        result = pendulum_immobility_series(activity, dt=0.04)
        assert len(result) == 100


class TestResolveImmobileFromActivity:
    def test_falls_back_when_no_mobility_column(self):
        """_resolve_immobile_series should use pendulum when no state column."""
        from ethoinsight.metrics._common import _resolve_immobile_series

        df = pd.DataFrame({
            "trial_time": np.arange(500) * 0.04,
            "activity": np.full(500, 0.01),  # nearly zero → still
        })
        result = _resolve_immobile_series(df, mobility_col=None)
        assert result is not None
        series, immobile_value = result
        assert immobile_value == 1  # one-hot style from pendulum
        assert len(series) > 0

    def test_returns_none_when_no_activity_column(self):
        from ethoinsight.metrics._common import _resolve_immobile_series

        df = pd.DataFrame({
            "trial_time": np.arange(100) * 0.04,
            "x_center": np.zeros(100),
        })
        result = _resolve_immobile_series(df, mobility_col=None)
        assert result is None
