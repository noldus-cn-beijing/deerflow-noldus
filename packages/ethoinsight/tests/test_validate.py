"""Tests for ethoinsight.validate — deterministic metric validation (S2).

AutoResearch-inspired: code-enforced NaN / Inf / range checks that should
not be left to LLM judgment.

Naming conventions tested (suffix-based, matching real catalog metric names):
  - *_pct     → 0–100 range (future percentage metrics)
  - *_ratio   → 0–1 range (real: open_arm_time_ratio, center_distance_ratio, …)
  - *_count   → non-negative (real: center_entry_count, transition_count, …)
  - *_time    → non-negative (real: open_zone_time, immobility_time, …)
  - *_latency → non-negative (real: light_latency, immobility_latency, …)
  - *_distance → non-negative (real: cumulative_distance, …)
"""

from ethoinsight.validate import validate_metrics


class TestValidateMetrics:
    """Unit tests for validate_metrics."""

    # ------------------------------------------------------------------
    # Happy path — real catalog metric names
    # ------------------------------------------------------------------

    def test_real_metric_names_pass(self):
        """Well-formed real metric values should produce no violations."""
        assert validate_metrics({
            "open_arm_time_ratio": 0.452,
            "center_distance_ratio": 0.3,
            "center_entry_count": 8,
            "cumulative_distance": 1500.0,
            "immobility_time": 120.5,
            "light_latency": 5.2,
            "open_zone_time": 45.0,
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

    def test_bool_is_skipped(self):
        """bool is explicitly skipped before isinstance(int) check,
        since isinstance(True, int) is True. Boolean True/False (=1/0)
        could coincidentally pass range checks but should not be treated
        as numeric metrics."""
        assert validate_metrics({"has_data": True}) == []
        assert validate_metrics({"has_data": False}) == []

    # ------------------------------------------------------------------
    # NaN / Inf — name-agnostic (the core AutoResearch safety net)
    # ------------------------------------------------------------------

    def test_nan_is_detected(self):
        """NaN values should be caught regardless of metric name."""
        violations = validate_metrics({"open_arm_time_ratio": float("nan")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "NaN"
        assert violations[0]["metric"] == "open_arm_time_ratio"

    def test_inf_is_detected(self):
        """Inf values should be caught."""
        violations = validate_metrics({"cumulative_distance": float("inf")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "Inf"

    def test_negative_inf_is_detected(self):
        """-Inf values should also be caught."""
        violations = validate_metrics({"cumulative_distance": float("-inf")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "Inf"

    # ------------------------------------------------------------------
    # Ratio range (0–1) — *_ratio suffix, real catalog convention
    # ------------------------------------------------------------------

    def test_ratio_out_of_range_high(self):
        violations = validate_metrics({"open_arm_time_ratio": 1.5})
        assert len(violations) == 1
        assert violations[0]["issue"] == "ratio_out_of_range"

    def test_ratio_out_of_range_negative(self):
        violations = validate_metrics({"center_distance_ratio": -0.1})
        assert len(violations) == 1
        assert violations[0]["issue"] == "ratio_out_of_range"

    def test_ratio_at_boundary_passes(self):
        assert validate_metrics({"open_arm_time_ratio": 0.0}) == []
        assert validate_metrics({"open_arm_time_ratio": 1.0}) == []

    # ------------------------------------------------------------------
    # Percentage range (0–100) — *_pct suffix
    # ------------------------------------------------------------------

    def test_pct_out_of_range_high(self):
        violations = validate_metrics({"mobility_pct": 150.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "percentage_out_of_range"

    def test_pct_out_of_range_negative(self):
        violations = validate_metrics({"immobility_pct": -5.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "percentage_out_of_range"

    def test_pct_at_boundary_passes(self):
        assert validate_metrics({"mobility_pct": 0.0}) == []
        assert validate_metrics({"mobility_pct": 100.0}) == []

    # ------------------------------------------------------------------
    # Non-negative checks — suffix-based, real catalog metric names
    # ------------------------------------------------------------------

    def test_negative_count(self):
        """Real metric: center_entry_count"""
        violations = validate_metrics({"center_entry_count": -1})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_time(self):
        """Real metric: immobility_time"""
        violations = validate_metrics({"immobility_time": -10.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_latency(self):
        """Real metric: light_latency"""
        violations = validate_metrics({"light_latency": -2.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_negative_distance(self):
        """Real metric: cumulative_distance"""
        violations = validate_metrics({"cumulative_distance": -50.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_non_negative_suffix_at_zero_passes(self):
        """Zero is valid for all non-negative metrics."""
        assert validate_metrics({
            "center_entry_count": 0,
            "immobility_time": 0.0,
            "light_latency": 0.0,
            "cumulative_distance": 0.0,
        }) == []

    # ------------------------------------------------------------------
    # Multiple violations
    # ------------------------------------------------------------------

    def test_multiple_violations_with_real_names(self):
        """All violating metrics should be reported in a single pass."""
        violations = validate_metrics({
            "immobility_pct": 120.0,             # percentage_out_of_range
            "cumulative_distance": -5.0,          # negative_value
            "center_entry_count": -1,             # negative_value
            "immobility_time": float("nan"),      # NaN
            "light_latency": -0.5,                # negative_value
            "open_arm_time_ratio": 0.45,          # OK
        })
        assert len(violations) == 5
        issues = {v["issue"] for v in violations}
        assert "percentage_out_of_range" in issues
        assert "negative_value" in issues
        assert "NaN" in issues

    def test_ratio_and_pct_have_different_ranges(self):
        """*_ratio uses 0-1 range, *_pct uses 0-100 range."""
        assert validate_metrics({"open_arm_time_ratio": 0.8}) == []
        assert validate_metrics({"mobility_pct": 80.0}) == []
        v_ratio = validate_metrics({"open_arm_time_ratio": 1.5})
        v_pct = validate_metrics({"mobility_pct": 150.0})
        assert v_ratio[0]["issue"] == "ratio_out_of_range"
        assert v_pct[0]["issue"] == "percentage_out_of_range"
