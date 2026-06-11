"""Tests for ethoinsight.validate_catalog — catalog-driven metric validation (L-B).

L-B does output_unit range checks, handling 6 known units:
  ratio / seconds / count / radians / cm / mm_s2

And detects orphan metrics (not in catalog) and unknown output_unit.
"""

import json


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
        """Direct-function entry: in-catalog metric with a normal value passes,
        while metrics not in the given paradigm's catalog surface as orphans."""
        violations = validate_metrics_against_catalog({
            "center_time_ratio": 0.452,   # real open_field metric, in range → OK
        }, "open_field")
        assert violations == []


    # --- None / non-numeric values are skipped ---
    def test_none_value_skipped(self):
        assert validate_metrics_against_catalog({"open_arm_time_ratio": None}, "epm") == []

    def test_string_value_skipped(self):
        assert validate_metrics_against_catalog({"open_arm_time_ratio": "no_data"}, "epm") == []


class TestMainCLI:
    """B.6: CLI entry point — plan-driven (uses plan entry's own output_unit)."""

    def test_cli_emits_validation_error_lines(self, tmp_path):
        """CLI with a plan pointing to a result file that has a violation."""
        import subprocess
        import sys

        # Write a plan_metrics.json (entry carries its own output_unit, P1-1)
        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "metric_0.json"),
                    "output_unit": "ratio",
                    "subject_index": 0,
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
                    "output_unit": "ratio",
                    "subject_index": 0,
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

    def test_cli_per_subject_no_collision(self, tmp_path):
        """P0-2: same metric_id across N subjects → each validated independently.

        plan_metrics.json expands one entry per subject. A buggy implementation
        that keys results by metric_id alone would only validate the LAST
        subject. This test has subject 0 = OK, subject 1 = violation, and
        asserts subject 1's violation is reported (not dropped) and labelled
        with its subject suffix.
        """
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "metric_s0.json"),
                    "output_unit": "ratio",
                    "subject_index": 0,
                },
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "metric_s1.json"),
                    "output_unit": "ratio",
                    "subject_index": 1,
                },
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))

        # subject 0: valid; subject 1: out of range (would be lost if keyed by id)
        (tmp_path / "metric_s0.json").write_text(
            json.dumps({"metric": "open_arm_time_ratio", "value": 0.4})
        )
        (tmp_path / "metric_s1.json").write_text(
            json.dumps({"metric": "open_arm_time_ratio", "value": 1.7})
        )

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        # The violating subject IS reported, labelled with its subject index.
        assert "open_arm_time_ratio#1" in proc.stdout
        assert "value_above_upper_bound" in proc.stdout
        # The valid subject is NOT reported.
        assert "open_arm_time_ratio#0" not in proc.stdout

    def test_cli_all_subjects_violating_all_reported(self, tmp_path):
        """P0-2: when every subject violates, every subject is reported."""
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "center_entry_count",
                    "output": str(tmp_path / f"m_{i}.json"),
                    "output_unit": "count",
                    "subject_index": i,
                }
                for i in range(3)
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))

        # All three subjects: negative count → all violate
        for i in range(3):
            (tmp_path / f"m_{i}.json").write_text(
                json.dumps({"metric": "center_entry_count", "value": -1})
            )

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        # Each subject reported separately.
        for i in range(3):
            assert f"center_entry_count#{i}" in proc.stdout

    def test_cli_uses_plan_output_unit_not_catalog(self, tmp_path):
        """P1-1: CLI uses the plan entry's output_unit, no catalog re-load.

        Proof: an entry whose id is NOT in any catalog still gets range-checked
        purely from its declared output_unit (load_catalog would have rejected
        / orphaned it). Here a made-up metric with output_unit=ratio and value
        2.0 must produce a range violation.
        """
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "made_up_metric_not_in_catalog",
                    "output": str(tmp_path / "m.json"),
                    "output_unit": "ratio",
                    "subject_index": 0,
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))
        (tmp_path / "m.json").write_text(
            json.dumps({"metric": "made_up_metric_not_in_catalog", "value": 2.0})
        )

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        # Range-checked from output_unit alone — no catalog lookup needed.
        assert "made_up_metric_not_in_catalog#0" in proc.stdout
        assert "value_above_upper_bound" in proc.stdout

    def test_cli_missing_output_unit_surfaced(self, tmp_path):
        """A plan entry lacking output_unit is surfaced, not silently skipped."""
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "m.json"),
                    "subject_index": 0,
                    # output_unit intentionally absent
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))
        (tmp_path / "m.json").write_text(
            json.dumps({"metric": "open_arm_time_ratio", "value": 0.4})
        )

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "plan_missing_output_unit" in proc.stdout

    def test_cli_unreadable_result_file_surfaced(self, tmp_path):
        """A metric whose output file is missing is surfaced as a gap."""
        import subprocess
        import sys

        plan = {
            "paradigm": "epm",
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "output": str(tmp_path / "does_not_exist.json"),
                    "output_unit": "ratio",
                    "subject_index": 0,
                }
            ],
        }
        plan_path = tmp_path / "plan_metrics.json"
        plan_path.write_text(json.dumps(plan))
        # deliberately do NOT create the result file

        proc = subprocess.run(
            [sys.executable, "-m", "ethoinsight.validate_catalog",
             "--plan", str(plan_path)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "result_file_unreadable" in proc.stdout

