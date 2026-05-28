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


# ============================================================================
# B3 — turn_angle_filtered
# ============================================================================


def test_turn_angle_filtered_basic():
    """Straight line with a 90° turn should be detected."""
    import numpy as np
    from ethoinsight.metrics._common import compute_turn_angle_filtered

    df = pd.DataFrame({
        "x_center": [0.0, 10.0, 10.0],
        "y_center": [0.0,  0.0, 10.0],
    })
    result = compute_turn_angle_filtered(df, min_displacement_mm=0.5)
    assert result is not None
    assert result["n"] == 1
    assert result["mean_abs_deg"] == pytest.approx(90.0, abs=1)


def test_turn_angle_filtered_stationary():
    """Tiny jitter below 1mm threshold should be filtered out."""
    import numpy as np
    from ethoinsight.metrics._common import compute_turn_angle_filtered

    df = pd.DataFrame({
        "x_center": [0.0, 0.01, 0.02, 0.01, 0.0],
        "y_center": [0.0, 0.01, 0.0, -0.01, 0.0],
    })
    result = compute_turn_angle_filtered(df, min_displacement_mm=1.0)
    assert result is None


def test_turn_angle_filtered_missing_columns():
    """Returns None when x_center/y_center missing."""
    from ethoinsight.metrics._common import compute_turn_angle_filtered

    df = pd.DataFrame({"velocity": [1, 2, 3]})
    assert compute_turn_angle_filtered(df) is None


def test_turn_angle_filtered_nan_handling():
    """NaN frames are skipped (matching Noldus JS 'if (pt)' guard)."""
    import numpy as np
    from ethoinsight.metrics._common import compute_turn_angle_filtered

    df = pd.DataFrame({
        "x_center": [0.0, np.nan, 10.0, np.nan, 20.0, 20.0, 20.0],
        "y_center": [0.0, np.nan,  0.0, np.nan,  0.0, 10.0, 20.0],
    })
    result = compute_turn_angle_filtered(df, min_displacement_mm=0.5)
    assert result is not None
    assert result["n"] >= 1
