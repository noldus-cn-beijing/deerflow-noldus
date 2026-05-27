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
