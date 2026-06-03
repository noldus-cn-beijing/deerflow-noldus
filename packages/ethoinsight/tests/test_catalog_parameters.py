"""Sprint 2a: catalog parameter specs — unit tests.

Verifies:
- ParamSpec parsing (required fields, valid_range consistency, default within range)
- MetricEntry.parameters / parameters_ref parsing
- ParadigmParameters parsing
- SharedParameters parsing in _common.yaml
- validate_catalog_consistency (parameters_ref validity, duplicate detection)
- load_all_catalogs e2e consistency
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from ethoinsight.catalog.loader import (
    CatalogError,
    CommonCatalog,
    SharedParameters,
    _parse_param_block,
    _parse_param_spec,
    load_all_catalogs,
    load_catalog,
    load_common_catalog,
    validate_catalog_consistency,
)
from ethoinsight.catalog.schema import Catalog, MetricEntry, ParamSpec, ParadigmParameters


# ============================================================================
# ParamSpec parsing
# ============================================================================


class TestParamSpec:
    def test_required_fields(self, tmp_path: Path):
        """Missing any required field → CatalogError."""
        for missing in ("default", "unit", "description", "tunable_by_user"):
            item = {
                "default": 30.0,
                "unit": "mm/s",
                "description": "test",
                "tunable_by_user": True,
            }
            del item[missing]
            with pytest.raises(CatalogError, match=f"missing '{missing}'"):
                _parse_param_spec(item, where="test", source=tmp_path / "t.yaml")

    def test_valid_range_min_gt_max(self, tmp_path: Path):
        """valid_range=[100, 50] (min > max) → CatalogError."""
        item = {
            "default": 75.0,
            "unit": "x",
            "description": "test",
            "tunable_by_user": True,
            "valid_range": [100, 50],
        }
        with pytest.raises(CatalogError, match="min.*100.*max.*50"):
            _parse_param_spec(item, where="test", source=tmp_path / "t.yaml")

    def test_default_outside_range(self, tmp_path: Path):
        """default=200, valid_range=[1, 100] → CatalogError."""
        item = {
            "default": 200,
            "unit": "x",
            "description": "test",
            "tunable_by_user": True,
            "valid_range": [1, 100],
        }
        with pytest.raises(CatalogError, match="default 200 outside valid_range"):
            _parse_param_spec(item, where="test", source=tmp_path / "t.yaml")

    def test_default_within_range_ok(self, tmp_path: Path):
        """default=50, valid_range=[1, 100] → OK."""
        item = {
            "default": 50,
            "unit": "x",
            "description": "test",
            "tunable_by_user": True,
            "valid_range": [1, 100],
        }
        spec = _parse_param_spec(item, where="test", source=tmp_path / "t.yaml")
        assert spec.default == 50
        assert spec.valid_range == [1, 100]

    def test_no_valid_range_ok(self, tmp_path: Path):
        """Omitting valid_range → None."""
        item = {
            "default": "hello",
            "unit": "str",
            "description": "test",
            "tunable_by_user": False,
        }
        spec = _parse_param_spec(item, where="test", source=tmp_path / "t.yaml")
        assert spec.default == "hello"
        assert spec.valid_range is None


# ============================================================================
# MetricEntry parameters / parameters_ref
# ============================================================================


class TestMetricEntryParameters:
    def test_metric_with_parameters_ref(self):
        """EPM metrics that reference shared velocity params."""
        cat = load_catalog("epm")
        # No metrics currently have parameters_ref (velocity not directly used)
        # But the field exists and defaults to []
        for m in cat.default_metrics + cat.optional_metrics:
            assert isinstance(m.parameters_ref, list)
            assert isinstance(m.parameters, dict)

    def test_parse_metric_with_parameters_ref(self, tmp_path: Path):
        """YAML metric with parameters_ref → parsed correctly."""
        yaml_text = textwrap.dedent("""\
            paradigm: test_paradigm
            ev19_templates: []
            default_metrics:
              - id: test_metric
                script: test.script
                requires_columns: [x]
                output_unit: count
                display_name_zh: 测试
                unit_zh: 个
                one_liner: test
                direction_for_anxiety: null
                statistical_default: groupwise_compare
                parameters_ref:
                  - velocity_threshold
            optional_metrics: []
            charts: []
        """)
        yaml_file = tmp_path / "test_paradigm.yaml"
        yaml_file.write_text(yaml_text, encoding="utf-8")
        cat = load_catalog("test_paradigm", catalog_dir=tmp_path)
        assert cat.default_metrics[0].parameters_ref == ["velocity_threshold"]

    def test_parse_metric_with_inline_parameters(self, tmp_path: Path):
        """YAML metric with inline parameters → parsed correctly."""
        yaml_text = textwrap.dedent("""\
            paradigm: test_paradigm
            ev19_templates: []
            default_metrics:
              - id: test_metric
                script: test.script
                requires_columns: [x]
                output_unit: count
                display_name_zh: 测试
                unit_zh: 个
                one_liner: test
                direction_for_anxiety: null
                statistical_default: groupwise_compare
                parameters:
                  my_param:
                    default: 42
                    unit: items
                    description: "test param"
                    tunable_by_user: true
            optional_metrics: []
            charts: []
        """)
        yaml_file = tmp_path / "test_paradigm.yaml"
        yaml_file.write_text(yaml_text, encoding="utf-8")
        cat = load_catalog("test_paradigm", catalog_dir=tmp_path)
        assert "my_param" in cat.default_metrics[0].parameters
        assert cat.default_metrics[0].parameters["my_param"].default == 42


# ============================================================================
# ParadigmParameters
# ============================================================================


class TestParadigmParameters:
    def test_epm_has_motor_threshold(self):
        """EPM paradigm_parameters has motor_low_entries_threshold."""
        cat = load_catalog("epm")
        assert "motor_low_entries_threshold" in cat.paradigm_parameters.parameters
        spec = cat.paradigm_parameters.parameters["motor_low_entries_threshold"]
        assert spec.default == 8
        assert spec.unit == "count"

    def test_zero_maze_has_distance_threshold(self):
        """Zero maze paradigm_parameters has zm_low_distance_threshold."""
        cat = load_catalog("zero_maze")
        assert "zm_low_distance_threshold" in cat.paradigm_parameters.parameters
        spec = cat.paradigm_parameters.parameters["zm_low_distance_threshold"]
        assert spec.default == 10.0
        assert spec.unit == "cm"

    def test_ldb_has_transition_threshold(self):
        """LDB paradigm_parameters has signal_low_transition_threshold."""
        cat = load_catalog("light_dark_box")
        assert "signal_low_transition_threshold" in cat.paradigm_parameters.parameters
        spec = cat.paradigm_parameters.parameters["signal_low_transition_threshold"]
        assert spec.default == 4

    def test_oft_center_metrics_have_center_zone_param(self):
        """center_zone is on each center metric (metric-level), not paradigm-level."""
        cat = load_catalog("open_field")
        # paradigm_parameters should be empty (center_zone moved to metric-level)
        assert len(cat.paradigm_parameters.parameters) == 0
        # Check each center metric has center_zone
        center_metric_ids = {"center_time_ratio", "center_distance_ratio", "center_entry_count", "center_time", "center_distance"}
        for m in cat.default_metrics:
            if m.id in center_metric_ids:
                assert "center_zone" in m.parameters, f"{m.id} missing center_zone"
                assert m.parameters["center_zone"].default == "in_zone_center"
            else:
                assert "center_zone" not in m.parameters, f"{m.id} should not have center_zone"
        # Optional metrics must NOT have center_zone
        for m in cat.optional_metrics:
            assert "center_zone" not in m.parameters, f"optional {m.id} should not have center_zone"


# ============================================================================
# SharedParameters (from _common.yaml)
# ============================================================================


class TestSharedParameters:
    def test_load_common_with_shared_params(self):
        """_common.yaml has shared_parameters with velocity params."""
        common = load_common_catalog()
        params = common.shared_parameters.parameters
        assert "velocity_threshold" in params
        assert params["velocity_threshold"].default == 30.0
        assert "velocity_min_duration" in params
        assert params["velocity_min_duration"].default == 25

    def test_shared_has_sample_size(self):
        """sample_size_underpowered_threshold is in shared_parameters."""
        common = load_common_catalog()
        params = common.shared_parameters.parameters
        assert "sample_size_underpowered_threshold" in params
        assert params["sample_size_underpowered_threshold"].default == 5

    def test_shared_has_pendulum_params(self):
        """All 9 pendulum params are in shared_parameters."""
        common = load_common_catalog()
        params = common.shared_parameters.parameters
        pendulum_keys = [k for k in params if k.startswith("pendulum_")]
        assert len(pendulum_keys) == 10


# ============================================================================
# validate_catalog_consistency
# ============================================================================


class TestValidateCatalogConsistency:
    def test_parameters_ref_unknown_id(self):
        """Metric with parameters_ref=['fake_param'] → CatalogError."""
        common = load_common_catalog()
        # Create a mock Catalog with a metric that references a nonexistent shared param
        mock_metric = MetricEntry(
            id="test",
            script="test",
            requires_columns=[],
            output_unit="x",
            display_name_zh="x",
            unit_zh="x",
            one_liner="x",
            direction_for_anxiety=None,
            statistical_default="groupwise_compare",
            parameters_ref=["nonexistent_param"],
        )
        mock_cat = Catalog(
            paradigm="mock",
            ev19_templates=[],
            default_metrics=[mock_metric],
            optional_metrics=[],
            charts=[],
            statistics_default=None,
        )
        with pytest.raises(CatalogError, match="not in _common.yaml.shared_parameters"):
            validate_catalog_consistency(common, [("mock", mock_cat)])

    def test_duplicate_params_across_paradigms(self):
        """Two paradigms defining same param name + default → CatalogError."""
        common = load_common_catalog()
        spec = ParamSpec(default=10, unit="x", description="x", tunable_by_user=True, valid_range=None)
        cat_a = Catalog(
            paradigm="a",
            ev19_templates=[],
            default_metrics=[],
            optional_metrics=[],
            charts=[],
            statistics_default=None,
            paradigm_parameters=ParadigmParameters(parameters={"foo_threshold": spec}),
        )
        cat_b = Catalog(
            paradigm="b",
            ev19_templates=[],
            default_metrics=[],
            optional_metrics=[],
            charts=[],
            statistics_default=None,
            paradigm_parameters=ParadigmParameters(parameters={"foo_threshold": spec}),
        )
        with pytest.raises(CatalogError, match="should be promoted"):
            validate_catalog_consistency(common, [("a", cat_a), ("b", cat_b)])

    def test_different_defaults_not_duplicate(self):
        """Two paradigms defining same param name but different defaults → OK."""
        common = load_common_catalog()
        spec_a = ParamSpec(default=10, unit="x", description="x", tunable_by_user=True, valid_range=None)
        spec_b = ParamSpec(default=20, unit="x", description="x", tunable_by_user=True, valid_range=None)
        cat_a = Catalog(
            paradigm="a",
            ev19_templates=[],
            default_metrics=[],
            optional_metrics=[],
            charts=[],
            statistics_default=None,
            paradigm_parameters=ParadigmParameters(parameters={"foo_threshold": spec_a}),
        )
        cat_b = Catalog(
            paradigm="b",
            ev19_templates=[],
            default_metrics=[],
            optional_metrics=[],
            charts=[],
            statistics_default=None,
            paradigm_parameters=ParadigmParameters(parameters={"foo_threshold": spec_b}),
        )
        # Should not raise
        validate_catalog_consistency(common, [("a", cat_a), ("b", cat_b)])


# ============================================================================
# load_all_catalogs e2e
# ============================================================================


class TestLoadAllCatalogs:
    def test_e2e_consistency(self):
        """Load all 6 paradigm YAMLs + _common.yaml → consistency check passes."""
        common, cats = load_all_catalogs()
        assert len(cats) == 6
        assert len(common.shared_parameters.parameters) >= 13  # 2 velocity + 1 sample + 10 pendulum

    def test_all_paradigm_keys_present(self):
        """All 6 expected paradigms loaded."""
        common, cats = load_all_catalogs()
        names = {name for name, _ in cats}
        assert names == {"epm", "open_field", "light_dark_box", "forced_swim", "tail_suspension", "zero_maze"}
