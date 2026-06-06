"""Tests for ethoinsight.validate — NaN/Inf safety net (L-A layer).

L-A 只保留 NaN/Inf 确定性安全检查（不依赖 catalog）。
范围校验（ratio/pct/非负）已迁移到 L-B（catalog-driven validate_catalog.py）。
"""

from ethoinsight.validate import validate_metrics


class TestValidateMetrics:
    """Unit tests for validate_metrics — L-A NaN/Inf safety net."""

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
    # NaN / Inf (L-A 唯一保留的检查, name-agnostic)
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
    # L-A 不再做 suffix 范围校验 — 全部迁移到 L-B catalog-driven 验证
    # ------------------------------------------------------------------

    def test_ratio_out_of_range_no_longer_checked_by_la(self):
        """Ratios above 1.0 are NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"open_arm_time_ratio": 1.5}) == []

    def test_ratio_negative_no_longer_checked_by_la(self):
        """Negative ratios are NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"center_time_ratio": -0.1}) == []

    def test_ratio_at_boundary_still_passes(self):
        """0.0 and 1.0 — no issue in either L-A or L-B."""
        assert validate_metrics({"open_arm_time_ratio": 0.0}) == []
        assert validate_metrics({"open_arm_time_ratio": 1.0}) == []

    def test_pct_out_of_range_no_longer_checked_by_la(self):
        """Percentages above 100 are NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"mobility_pct": 150.0}) == []

    def test_pct_negative_no_longer_checked_by_la(self):
        """Negative percentages are NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"immobility_pct": -5.0}) == []

    def test_pct_at_boundary_still_passes(self):
        """0% and 100% — no issue in either L-A or L-B."""
        assert validate_metrics({"mobility_pct": 0.0}) == []
        assert validate_metrics({"mobility_pct": 100.0}) == []

    def test_negative_count_no_longer_checked_by_la(self):
        """Negative count is NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"center_entry_count": -1}) == []

    def test_negative_time_no_longer_checked_by_la(self):
        """Negative time is NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"immobility_time": -10.0}) == []

    def test_negative_latency_no_longer_checked_by_la(self):
        """Negative latency is NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"light_latency": -2.0}) == []

    def test_negative_distance_no_longer_checked_by_la(self):
        """Negative distance is NOT checked by L-A — moved to L-B."""
        assert validate_metrics({"cumulative_distance": -50.0}) == []

    def test_non_negative_values_at_zero_pass(self):
        """Zero is valid (and L-A never flags non-NaN/Inf anyway)."""
        assert validate_metrics({
            "center_entry_count": 0,
            "immobility_time": 0.0,
            "light_latency": 0.0,
            "cumulative_distance": 0.0,
        }) == []

    # ------------------------------------------------------------------
    # Multiple violations (L-A: only NaN/Inf now)
    # ------------------------------------------------------------------

    def test_multiple_violations_only_nan_inf(self):
        """After narrowing, only NaN/Inf produce violations in L-A.

        Range-violating values (out-of-range pct, negative distance) are
        silently passed by L-A — they are L-B's responsibility now.
        """
        violations = validate_metrics({
            "immobility_pct": 120.0,            # L-A no longer checks
            "cumulative_distance": -5.0,        # L-A no longer checks
            "immobility_time": float("nan"),    # NaN — still caught
            "center_entry_count": 3,            # OK
            "open_zone_time": float("-inf"),    # Inf — still caught
        })
        assert len(violations) == 2
        issues = {v["issue"] for v in violations}
        assert issues == {"NaN", "Inf"}

    def test_ratio_and_pct_no_longer_checked(self):
        """L-A no longer checks *_ratio or *_pct ranges (moved to L-B)."""
        assert validate_metrics({"open_arm_time_ratio": 0.8}) == []
        assert validate_metrics({"mobility_pct": 80.0}) == []
        # These used to be violations — now empty (moved to L-B)
        assert validate_metrics({"open_arm_time_ratio": 1.5}) == []
        assert validate_metrics({"mobility_pct": 150.0}) == []
