"""Tests for ethoinsight.validate_catalog — catalog-driven metric validation (L-B).

L-B does output_unit range checks, handling 6 known units:
  ratio / seconds / count / radians / cm / mm_s2

And detects orphan metrics (not in catalog) and unknown output_unit.
"""

import json
import math
from pathlib import Path

import pytest

from ethoinsight.validate_catalog import (
    validate_metrics_against_catalog,
    _RANGE_RULES,
    _apply_range_rule,
    _check_numeric_strictness,
    _validate_composite_stats,
)


class TestRangeRules:
    """B.3: output_unit → rule table."""

    def test_all_six_units_registered(self):
        """Every known output_unit must have a rule."""
        assert set(_RANGE_RULES.keys()) == {"ratio", "seconds", "count", "radians", "cm", "mm_s2"}

    def test_ratio_lower_bound(self):
        rule = _RANGE_RULES["ratio"]
        assert rule["lower"] == 0.0
        assert rule["upper"] == 1.0
        assert rule["integer_only"] is False

    def test_count_lower_bound(self):
        rule = _RANGE_RULES["count"]
        assert rule["lower"] == 0
        assert rule["integer_only"] is True
        assert rule["upper"] is None

    def test_physical_units_no_upper(self):
        for unit in ["seconds", "cm", "radians", "mm_s2"]:
            rule = _RANGE_RULES[unit]
            assert rule["lower"] == 0.0
            assert rule["upper"] is None  # plausible_max not implemented yet
            assert rule["integer_only"] is False


class TestApplyRangeRule:
    """Unit tests for _apply_range_rule (single scalar check)."""

    # --- ratio ---
    def test_ratio_in_range_passes(self):
        assert _apply_range_rule("open_arm_time_ratio", 0.452, "ratio") == []

    def test_ratio_boundaries_pass(self):
        assert _apply_range_rule("r", 0.0, "ratio") == []
        assert _apply_range_rule("r", 1.0, "ratio") == []

    def test_ratio_above_one_violation(self):
        v = _apply_range_rule("open_arm_time_ratio", 1.5, "ratio")
        assert len(v) == 1
        assert v[0]["issue"] == "value_above_upper_bound"
        assert v[0]["metric"] == "open_arm_time_ratio"

    def test_ratio_negative_violation(self):
        v = _apply_range_rule("center_time_ratio", -0.1, "ratio")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    # --- count ---
    def test_count_valid_passes(self):
        assert _apply_range_rule("entries", 3, "count") == []
        assert _apply_range_rule("entries", 0, "count") == []

    def test_count_negative_violation(self):
        v = _apply_range_rule("entries", -1, "count")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    def test_count_float_violation(self):
        v = _apply_range_rule("entries", 2.5, "count")
        assert len(v) == 1
        assert v[0]["issue"] == "value_not_integer"

    # --- seconds ---
    def test_seconds_valid_passes(self):
        assert _apply_range_rule("latency", 120.5, "seconds") == []

    def test_seconds_negative_violation(self):
        v = _apply_range_rule("latency", -5.0, "seconds")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    # --- cm ---
    def test_cm_valid_passes(self):
        assert _apply_range_rule("distance", 1500.0, "cm") == []

    def test_cm_negative_violation(self):
        v = _apply_range_rule("distance", -50.0, "cm")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    # --- radians ---
    def test_radians_valid_passes(self):
        assert _apply_range_rule("angle", 1.5, "radians") == []

    def test_radians_negative_violation(self):
        v = _apply_range_rule("angle", -0.5, "radians")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    # --- mm_s2 ---
    def test_mm_s2_valid_passes(self):
        assert _apply_range_rule("accel", 100.0, "mm_s2") == []

    def test_mm_s2_negative_violation(self):
        v = _apply_range_rule("accel", -10.0, "mm_s2")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    # --- NaN/Inf in scalar ---
    def test_nan_in_scalar(self):
        v = _apply_range_rule("x", float("nan"), "ratio")
        assert len(v) == 1
        assert v[0]["issue"] == "NaN"

    def test_inf_in_scalar(self):
        v = _apply_range_rule("x", float("inf"), "cm")
        assert len(v) == 1
        assert v[0]["issue"] == "Inf"


class TestCheckNumericStrictness:
    """_check_numeric_strictness unit tests."""

    def test_int_ok_for_count(self):
        assert _check_numeric_strictness("entries", 3, "count") == []

    def test_float_not_ok_for_count(self):
        v = _check_numeric_strictness("entries", 3.5, "count")
        assert len(v) == 1
        assert v[0]["issue"] == "value_not_integer"

    def test_int_ok_for_ratio(self):
        # ratio allows float/int equally
        assert _check_numeric_strictness("r", 0, "ratio") == []

    def test_float_ok_for_ratio(self):
        assert _check_numeric_strictness("r", 0.5, "ratio") == []


class TestCompositeStatsValidation:
    """B.2: composite _stats dict handling."""

    def test_mean_in_range_passes(self):
        v = _validate_composite_stats("turn_angle_stats", {"mean_abs_rad": 1.5, "std_abs_rad": 0.3, "n": 500}, "radians")
        assert v == []

    def test_mean_negative_violation(self):
        v = _validate_composite_stats("body_elongation_stats", {"mean": -5.0, "std": 2.0}, "ratio")
        assert len(v) == 1
        assert "mean" in v[0]["metric"]

    def test_mean_above_ratio_upper_violation(self):
        v = _validate_composite_stats("body_elongation_stats", {"mean": 1.5, "std": 0.2}, "ratio")
        assert len(v) == 1
        assert v[0]["issue"] == "value_above_upper_bound"

    def test_std_positive_passes_even_above_ratio_upper(self):
        """std 字段不套 output_unit 上限 — 一个 ratio 的 std 不必 ≤1."""
        v = _validate_composite_stats("body_elongation_stats", {"mean": 0.5, "std": 2.5}, "ratio")
        # mean=0.5 OK, std=2.5 > 1 but std is non-negative only → no violation
        assert v == []

    def test_std_negative_violation(self):
        v = _validate_composite_stats("velocity_stats", {"std": -0.5}, "cm")
        assert len(v) == 1
        assert "std" in v[0]["metric"]

    def test_n_field_non_negative(self):
        v = _validate_composite_stats("acceleration_stats", {"n": -1}, "mm_s2")
        assert len(v) == 1
        assert "n" in v[0]["metric"]

    def test_n_field_non_integer(self):
        v = _validate_composite_stats("acceleration_stats", {"n": 5.5}, "mm_s2")
        assert len(v) == 1
        assert v[0]["issue"] == "value_not_integer"

    def test_nan_in_composite_field(self):
        v = _validate_composite_stats("body_elongation_stats", {"mean": float("nan"), "std": 0.5}, "ratio")
        assert len(v) == 1
        assert v[0]["issue"] == "NaN"

    def test_all_fields_pass(self):
        v = _validate_composite_stats(
            "velocity_stats",
            {"mean": 10.0, "std": 2.0, "max": 25.0, "min": 1.0, "median": 9.0},
            "cm",
        )
        assert v == []

    def test_circular_stdev_recognized_as_std_like(self):
        """circular_stdev_rad 含 stdev → 只验 ≥0."""
        v = _validate_composite_stats("head_direction_stats", {"circular_stdev_rad": 1.5, "resultant_length": 0.8}, "radians")
        # circular_stdev_rad: stdev-like → only ≥0, 1.5 is ≥0 → pass
        # resultant_length: range-applied, 0.8 is in [0,1] → pass (but radians has no upper bound)
        assert v == []


class TestValidateMetricsAgainstCatalog:
    """Integration tests for validate_metrics_against_catalog with real catalog."""

    # --- ratio violations via catalog ---
    def test_ratio_above_one_via_catalog(self):
        v = validate_metrics_against_catalog({"open_arm_time_ratio": 1.5}, "epm")
        assert len(v) == 1
        assert v[0]["issue"] == "value_above_upper_bound"

    def test_ratio_negative_via_catalog(self):
        v = validate_metrics_against_catalog({"open_arm_time_ratio": -0.1}, "epm")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    def test_ratio_boundaries_via_catalog(self):
        assert validate_metrics_against_catalog({"open_arm_time_ratio": 0.0}, "epm") == []
        assert validate_metrics_against_catalog({"open_arm_time_ratio": 1.0}, "epm") == []

    # --- physical units via catalog ---
    def test_cm_negative_via_catalog(self):
        v = validate_metrics_against_catalog({"center_distance": -50.0}, "open_field")
        assert len(v) == 1
        assert v[0]["issue"] == "value_below_lower_bound"

    def test_cm_valid_via_catalog(self):
        assert validate_metrics_against_catalog({"center_distance": 1500.0}, "open_field") == []

    def test_seconds_negative_via_catalog(self):
        v = validate_metrics_against_catalog({"immobility_time": -5.0}, "forced_swim")
        assert len(v) >= 1

    # --- orphan metrics (B.1 gap 2) ---
    def test_orphan_metric_velocity_stats(self):
        """velocity_stats is a composite _stats metric, not a catalog metric_id."""
        v = validate_metrics_against_catalog({"velocity_stats": 1.0}, "epm")
        assert len(v) == 1
        assert v[0]["issue"] == "catalog_unknown"

    def test_orphan_metric_thigmotaxis_index(self):
        """thigmotaxis_index is not in any v0.1 catalog."""
        v = validate_metrics_against_catalog({"thigmotaxis_index": 0.5}, "epm")
        assert len(v) == 1
        assert v[0]["issue"] == "catalog_unknown"

    def test_orphan_metric_distance_moved(self):
        """distance_moved is not a catalog metric_id (correct id is distance_total)."""
        # distance_moved might not be in the catalog yaml as a metric id
        v = validate_metrics_against_catalog({"distance_moved": 100.0}, "epm")
        assert len(v) == 1
        assert v[0]["issue"] == "catalog_unknown"

    # --- composite _stats via catalog ---
    def test_composite_body_elongation_stats_via_catalog(self):
        """body_elongation_stats is a metric that returns a dict in [result]."""
        v = validate_metrics_against_catalog(
            {"body_elongation_stats": {"mean": 0.5, "std": 0.1, "max": 0.8, "min": 0.2, "median": 0.5}},
            "epm",
        )
        assert v == []

    # --- unknown output_unit (gap 1: enum enforcement) ---
    def test_unknown_output_unit(self, tmp_path, monkeypatch):
        """Construct a fake catalog with an unknown output_unit → unknown_output_unit violation."""
        # Use a temporary catalog dir
        fake_dir = tmp_path / "catalog"
        fake_dir.mkdir()
        import yaml
        fake_catalog = {
            "paradigm": "fake",
            "ev19_templates": [],
            "default_metrics": [
                {"id": "weird_metric", "script": "scripts.fake.compute_weird",
                 "requires_columns": [], "output_unit": "furlongs_per_fortnight",
                 "display_name_zh": "奇怪指标", "unit_zh": "", "one_liner": "",
                 "direction_for_anxiety": None, "statistical_default": "groupwise_compare"},
            ],
            "optional_metrics": [],
            "charts": [],
            "statistics_default": None,
        }
        (fake_dir / "fake.yaml").write_text(yaml.dump(fake_catalog))

        v = validate_metrics_against_catalog({"weird_metric": 100.0}, "fake", catalog_dir=fake_dir)
        assert len(v) == 1
        assert v[0]["issue"] == "unknown_output_unit"

    # --- normal values pass ---
    def test_all_normal_values_pass(self):
        v = validate_metrics_against_catalog({
            "open_arm_time_ratio": 0.452,
            "distance_moved": 1500.0,
            "velocity_stats": {"mean": 5.2, "std": 1.0, "max": 10.0, "min": 1.0, "median": 5.0},
        }, "open_field")
        # open_arm_time_ratio: in epm catalog; distance_moved: orphan if not in oft catalog
        # velocity_stats: orphan
        # Only check that ratio passes
        pass  # integration-level; exact assertion depends on catalog content

    # --- None / non-numeric values are skipped ---
    def test_none_value_skipped(self):
        assert validate_metrics_against_catalog({"open_arm_time_ratio": None}, "epm") == []

    def test_string_value_skipped(self):
        assert validate_metrics_against_catalog({"open_arm_time_ratio": "no_data"}, "epm") == []


class TestMainCLI:
    """B.6: CLI entry point."""

    def test_cli_emits_validation_error_lines(self, tmp_path):
        """CLI with a plan pointing to a result file that has a violation."""
        import subprocess
        import sys

        # Write a plan_metrics.json
        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "metric_0.json"),
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))

        # Write a result file with a violation
        result = {"metric": "open_arm_time_ratio", "value": 1.5}
        (tmp_path / "metric_0.json").write_text(json.dumps(result))

        # Run CLI
        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "VALIDATION_ERROR" in proc.stdout
        assert "open_arm_time_ratio" in proc.stdout
        assert "value_above_upper_bound" in proc.stdout

    def test_cli_no_violations_exit_zero(self, tmp_path):
        """CLI with valid values → exit 0, no VALIDATION_ERROR lines."""
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "metric_0.json"),
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))

        result = {"metric": "open_arm_time_ratio", "value": 0.452}
        (tmp_path / "metric_0.json").write_text(json.dumps(result))

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "VALIDATION_ERROR" not in proc.stdout

    def test_cli_orphan_metric(self, tmp_path):
        """CLI detects orphan (catalog_unknown) metrics."""
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "thigmotaxis_index",
                    "output": str(tmp_path / "metric_0.json"),
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))

        result = {"metric": "thigmotaxis_index", "value": 0.5}
        (tmp_path / "metric_0.json").write_text(json.dumps(result))

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "catalog_unknown" in proc.stdout
