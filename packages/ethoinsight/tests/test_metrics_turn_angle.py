"""Tests for turn angle metric."""

import math

import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_turn_angle_stats


def test_turn_angle_basic():
    df = pd.DataFrame({"TurnAngle": [0.1, -0.2, 0.3, -0.1, 0.0]})
    result = compute_turn_angle_stats(df)
    assert result is not None
    assert "mean_abs_rad" in result
    assert "mean_abs_deg" in result
    assert result["mean_abs_deg"] > 0


def test_turn_angle_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_turn_angle_stats(df)
    assert result is None


def test_turn_angle_all_nan():
    df = pd.DataFrame({"TurnAngle": [float("nan"), float("nan")]})
    result = compute_turn_angle_stats(df)
    assert result is None


def test_turn_angle_conversion():
    """180 degrees = pi radians."""
    df = pd.DataFrame({"TurnAngle": [math.pi, -math.pi]})
    result = compute_turn_angle_stats(df)
    assert result["mean_abs_deg"] == pytest.approx(180.0, rel=0.01)


def test_turn_angle_zero():
    df = pd.DataFrame({"TurnAngle": [0.0, 0.0, 0.0]})
    result = compute_turn_angle_stats(df)
    assert result["mean_abs_deg"] == 0.0
    assert result["total_abs_rad"] == 0.0
