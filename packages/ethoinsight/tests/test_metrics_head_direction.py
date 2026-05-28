"""Tests for head direction metric."""

import math

import numpy as np
import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_head_direction_stats


def test_direction_basic():
    df = pd.DataFrame({"Direction": [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]})
    result = compute_head_direction_stats(df)
    assert result is not None
    assert "mean_rad" in result
    assert "circular_stdev_rad" in result


def test_direction_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_head_direction_stats(df)
    assert result is None


def test_direction_all_nan():
    df = pd.DataFrame({"Direction": [float("nan"), float("nan")]})
    result = compute_head_direction_stats(df)
    assert result is None


def test_direction_uniform():
    """Uniform directions: resultant length R should be small."""
    rng = np.random.default_rng(42)
    directions = rng.uniform(0, 2 * math.pi, size=100)
    df = pd.DataFrame({"Direction": directions})
    result = compute_head_direction_stats(df)
    assert result is not None
    # Uniform: R/n should be small (< 0.15 for n=100)
    assert result["resultant_length"] / result["n"] < 0.15
    assert result["n"] == 100


def test_direction_single_value():
    """Single direction: R should equal n."""
    df = pd.DataFrame({"Direction": [math.pi / 4] * 10})
    result = compute_head_direction_stats(df)
    assert result["resultant_length"] == pytest.approx(10.0, rel=0.01)


# ============================================================================
# B10 — heading_smoothed
# ============================================================================


def test_heading_smoothed_basic():
    from ethoinsight.metrics._common import compute_heading_smoothed

    df = pd.DataFrame({"Direction": [0.1, 0.2, 0.3, 0.4, 0.5]})
    result = compute_heading_smoothed(df, window=3)
    assert result is not None
    assert "mean_rad" in result
    assert "circular_stdev_rad" in result
    assert result["n"] == 5


def test_heading_smoothed_circular_boundary():
    """SMA should handle ±π wrap correctly."""
    from ethoinsight.metrics._common import compute_heading_smoothed

    # Values that cross the ±π boundary
    df = pd.DataFrame({"Direction": [3.0, 3.1, -3.1, -3.0, 3.0]})
    result = compute_heading_smoothed(df, window=3)
    assert result is not None
    # Should not produce wild values from boundary discontinuity
    assert 0 <= result["mean_rad"] < 2 * math.pi


def test_heading_smoothed_missing_column():
    from ethoinsight.metrics._common import compute_heading_smoothed

    df = pd.DataFrame({"velocity": [1, 2, 3]})
    assert compute_heading_smoothed(df) is None


def test_heading_smoothed_all_nan():
    from ethoinsight.metrics._common import compute_heading_smoothed

    df = pd.DataFrame({"Direction": [float("nan"), float("nan")]})
    assert compute_heading_smoothed(df) is None


def test_heading_smoothed_nan_gap():
    """NaN gaps should be forward-filled so unwrap doesn't break."""
    from ethoinsight.metrics._common import compute_heading_smoothed

    df = pd.DataFrame({"Direction": [0.0, float("nan"), float("nan"), 0.5, 1.0]})
    result = compute_heading_smoothed(df, window=3)
    assert result is not None
    assert result["n"] == 3  # only 3 originally-valid frames
