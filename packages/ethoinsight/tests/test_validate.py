"""Tests for ethoinsight.validate — deterministic metric validation (S2).

AutoResearch-inspired: code-enforced NaN / Inf / range checks that should
not be left to LLM judgment.

Naming conventions tested:
  - *_pct  → 0–100 range (future percentage metrics)
  - *_ratio → 0–1 range (current catalog convention, e.g. open_arm_time_ratio)
  - distance_*, duration_*, velocity_*, count_* → non-negative
"""

import math

import pytest

from ethoinsight.validate import validate_metrics


class TestValidateMetrics:
    """Unit tests for validate_metrics."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_normal_values_pass(self):
        """Well-formed metric values should produce no violations."""
        assert validate_metrics({
            "open_arm_time_ratio": 0.452,
            "distance_total": 1500.0,
            "duration_immobile": 120.5,
            "velocity_mean": 5.2,
            "count_entries": 3,
        }) == []

    def test_empty_metrics(self):
        """Empty dict should produce no violations."""
        assert validate_metrics({}) == []

    def test_non_numeric_values_are_skipped(self):
        """String and None values should not trigger violations."""
        assert validate_metrics({"note": "no data"}) == []
        assert validate_metrics({"missing": None}) == []

    def test_list_values_are_skipped(self):
        """List/array values should not trigger violations."""
        assert validate_metrics({"results": [1.0, 2.0, 3.0]}) == []

    def test_bool_values_are_skipped(self):
        """Boolean values should not trigger violations."""
        assert validate_metrics({"has_data": True}) == []
        assert validate_metrics({"has_data": False}) == []

    # ------------------------------------------------------------------
    # NaN / Inf
    # ------------------------------------------------------------------

    def test_nan_is_detected(self):
        """NaN values should be caught."""
        violations = validate_metrics({"open_arm_time_ratio": float("nan")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "NaN"
        assert violations[0]["metric"] == "open_arm_time_ratio"

    def test_inf_is_detected(self):
        """Inf values should be caught."""
        violations = validate_metrics({"distance_total": float("inf")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "Inf"

    def test_negative_inf_is_detected(self):
        """-Inf values should also be caught."""
        violations = validate_metrics({"distance_total": float("-inf")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "Inf"

    # ------------------------------------------------------------------
    # Ratio range (0–1)
    # ------------------------------------------------------------------

    def test_ratio_out_of_range_high(self):
        """Ratios above 1.0 should be flagged."""
        violations = validate_metrics({"open_arm_time_ratio": 1.5})
        assert len(violations) == 1
        assert violations[0]["issue"] == "ratio_out_of_range"
        assert violations[0]["value"] == "1.5"

    def test_ratio_out_of_range_negative(self):
        """Negative ratios should be flagged."""
        violations = validate_metrics({"center_time_ratio": -0.1})
        assert len(violations) == 1
        assert violations[0]["issue"] == "ratio_out_of_range"

    def test_ratio_at_boundary_passes(self):
        """0.0 and 1.0 are valid boundary ratio values."""
        assert validate_metrics({"open_arm_time_ratio": 0.0}) == []
        assert validate_metrics({"open_arm_time_ratio": 1.0}) == []

    # ------------------------------------------------------------------
    # Percentage range (0–100, *_pct convention)
    # ------------------------------------------------------------------

    def test_pct_out_of_range_high(self):
        """Percentages above 100 should be flagged."""
        violations = validate_metrics({"mobility_pct": 150.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "percentage_out_of_range"

    def test_pct_out_of_range_negative(self):
        """Negative percentages should be flagged."""
        violations = validate_metrics({"immobility_pct": -5.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "percentage_out_of_range"

    def test_pct_at_boundary_passes(self):
        """0% and 100% are valid boundary values."""
        assert validate_metrics({"mobility_pct": 0.0}) == []
        assert validate_metrics({"mobility_pct": 100.0}) == []

    # ------------------------------------------------------------------
    # Non-negative checks
    # ------------------------------------------------------------------

    def test_negative_distance(self):
        """Negative distance should be flagged."""
        violations = validate_metrics({"distance_total": -50.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_duration(self):
        """Negative duration should be flagged."""
        violations = validate_metrics({"duration_immobile": -1.5})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_velocity(self):
        """Negative velocity should be flagged."""
        violations = validate_metrics({"velocity_mean": -10.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_count(self):
        """Negative count should be flagged."""
        violations = validate_metrics({"count_entries": -1})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    # ------------------------------------------------------------------
    # Multiple violations
    # ------------------------------------------------------------------

    def test_multiple_violations(self):
        """All violating metrics should be reported in a single pass."""
        violations = validate_metrics({
            "immobility_pct": 120.0,          # percentage_out_of_range
            "distance_total": -5.0,           # negative_value
            "duration_immobile": float("nan"), # NaN
            "count_entries": 3,               # OK
            "velocity_mean": -0.5,            # negative_value
        })
        assert len(violations) == 4
        issues = {v["issue"] for v in violations}
        assert "percentage_out_of_range" in issues
        assert "negative_value" in issues
        assert "NaN" in issues

    def test_ratio_and_pct_different_ranges(self):
        """*_ratio uses 0-1 range, *_pct uses 0-100 range."""
        # 0.8 is valid for ratio (0-1) but 80.0 for pct would be valid too
        assert validate_metrics({"open_arm_time_ratio": 0.8}) == []
        assert validate_metrics({"mobility_pct": 80.0}) == []
        # 1.5 is out of range for ratio, but 150.0 is out of range for pct
        v_ratio = validate_metrics({"open_arm_time_ratio": 1.5})
        v_pct = validate_metrics({"mobility_pct": 150.0})
        assert v_ratio[0]["issue"] == "ratio_out_of_range"
        assert v_pct[0]["issue"] == "percentage_out_of_range"
