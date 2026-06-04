"""Tests for body elongation metric."""

import pandas as pd
import pytest
from ethoinsight.metrics._common import compute_body_elongation_stats


def test_elongation_basic():
    df = pd.DataFrame({"Elongation": [0.5, 0.6, 0.4, 0.55, 0.45]})
    result = compute_body_elongation_stats(df)
    assert result is not None
    assert "mean" in result
    assert "std" in result
    assert result["mean"] == pytest.approx(50.0)


def test_elongation_zero():
    df = pd.DataFrame({"Elongation": [0.0, 0.0, 0.0]})
    result = compute_body_elongation_stats(df)
    assert result["mean"] == 0.0
    assert result["std"] == 0.0


def test_elongation_perfect():
    df = pd.DataFrame({"Elongation": [1.0, 1.0]})
    result = compute_body_elongation_stats(df)
    assert result["mean"] == 100.0


def test_elongation_missing_column():
    df = pd.DataFrame({"velocity": [1, 2, 3]})
    result = compute_body_elongation_stats(df)
    assert result is None


def test_elongation_all_nan():
    df = pd.DataFrame({"Elongation": [float("nan"), float("nan")]})
    result = compute_body_elongation_stats(df)
    assert result is None


def test_elongation_min_max():
    df = pd.DataFrame({"Elongation": [0.1, 0.2, 0.3]})
    result = compute_body_elongation_stats(df)
    assert result["min"] == 10.0
    assert result["max"] == 30.0


# ============================================================================
# B5 — body_length (Noldus "Body Length - Sum of segments")
# ============================================================================


def test_body_length_basic():
    import numpy as np
    from ethoinsight.metrics._common import compute_body_length

    df = pd.DataFrame({
        "x_nose":   [0.0, 0.0, 0.0],
        "y_nose":   [2.0, 2.0, 2.0],
        "x_center": [0.0, 0.0, 0.0],
        "y_center": [0.0, 0.0, 0.0],
        "x_tail":   [0.0, 0.0, 0.0],
        "y_tail":   [-2.0, -2.0, -2.0],
    })
    result = compute_body_length(df)
    assert result is not None
    assert result["mean"] == pytest.approx(4.0)  # 2 + 2
    assert result["std"] == 0.0


def test_body_length_missing_columns():
    from ethoinsight.metrics._common import compute_body_length

    df = pd.DataFrame({"x_center": [1, 2], "y_center": [1, 2]})
    assert compute_body_length(df) is None


def test_body_length_all_nan():
    import numpy as np
    from ethoinsight.metrics._common import compute_body_length

    df = pd.DataFrame({
        "x_nose":   [np.nan, np.nan],
        "y_nose":   [np.nan, np.nan],
        "x_center": [np.nan, np.nan],
        "y_center": [np.nan, np.nan],
        "x_tail":   [np.nan, np.nan],
        "y_tail":   [np.nan, np.nan],
    })
    assert compute_body_length(df) is None
