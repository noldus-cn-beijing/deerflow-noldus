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
