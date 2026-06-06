"""Tests for column-semantics Sprint 1: assess_column_confidence, _apply_aliases,
and resolve_metrics with column_aliases.

Updated 2026-06-06: _apply_aliases now only removes __ignore__ columns;
concept matching is done on-the-fly by _missing_columns and
_build_zone_aliases_overrides. Tests verify parameters_in_use uses
PHYSICAL column names, not catalog concept names.
"""

from ethoinsight.catalog.resolve import ResolveError, _apply_aliases, resolve_metrics
from ethoinsight.utils import assess_column_confidence


class TestAssessColumnConfidence:
    """assess_column_confidence — deterministic column recognition."""

    def test_all_standard_columns_recognized(self):
        """Standard EthoVision columns (L1 7 + common) are all recognized."""
        standard = [
            "Trial time", "Recording time",
            "X center", "Y center",
            "X nose", "Y nose",
            "Distance moved", "Velocity",
            "Body area", "Elongation",
            "Result 1", "Result 2",
        ]
        result = assess_column_confidence(standard)
        assert len(result["unrecognized"]) == 0, f"Unexpected unrecognized: {result['unrecognized']}"
        assert len(result["recognized"]) == len(standard)

    def test_chinese_column_names_recognized_via_map(self):
        """Chinese column names in COLUMN_MAP are recognized."""
        standard = ["试用时间", "X 中心", "Y 中心", "移动距离", "速度", "Result 1"]
        result = assess_column_confidence(standard)
        assert len(result["unrecognized"]) == 0

    def test_custom_zone_columns_unrecognized_without_patterns(self):
        """Custom zone columns are unrecognized when no catalog patterns provided."""
        columns = ["中心区", "边缘区", "边缘区到中心区"]
        result = assess_column_confidence(columns)
        assert len(result["unrecognized"]) == 3
        assert {e["raw"] for e in result["unrecognized"]} == {"中心区", "边缘区", "边缘区到中心区"}

    def test_custom_zone_columns_recognized_with_patterns(self):
        """Custom zone columns ARE recognized when catalog patterns match the normalized name."""
        columns = ["中心区", "边缘区", "边缘区到中心区", "Trial time"]
        patterns = ["in_zone_center_*", "in_zone_border_*", "in_zone_*"]
        result = assess_column_confidence(columns, required_patterns=patterns)
        recognized_raw = {e["raw"] for e in result["recognized"]}
        assert "Trial time" in recognized_raw  # COLUMN_MAP hit

    def test_34_file_real_columns(self):
        """Simulate real 34-file OFT columns — zone columns unrecognized, standard recognized."""
        real_columns = [
            "Trial time", "Recording time",
            "X center", "Y center",
            "X nose", "Y nose",
            "X tail", "Y tail",
            "Distance moved", "Velocity",
            "Heading", "Body area",
            "Area change", "Elongation",
            "Result 1", "Result 2",
            "中心区", "边缘区", "边缘区到中心区",
        ]
        result = assess_column_confidence(real_columns)
        recognized_raw = {e["raw"] for e in result["recognized"]}
        unrecognized_raw = {e["raw"] for e in result["unrecognized"]}

        for std in ["Trial time", "Recording time", "X center", "Y center",
                     "Distance moved", "Velocity", "Result 1", "Result 2"]:
            assert std in recognized_raw, f"{std} should be recognized"

        assert "中心区" in unrecognized_raw
        assert "边缘区" in unrecognized_raw
        assert "边缘区到中心区" in unrecognized_raw

    def test_normalized_field_present(self):
        """Each entry has both raw and normalized fields."""
        columns = ["中心区", "Trial time"]
        result = assess_column_confidence(columns)
        for entry in result["recognized"] + result["unrecognized"]:
            assert "raw" in entry
            assert "normalized" in entry
            assert isinstance(entry["raw"], str)
            assert isinstance(entry["normalized"], str)

    def test_with_catalog_patterns_oft(self):
        """With OFT catalog patterns, 'center' (normalized from 中心区) should match."""
        columns = ["中心区", "边缘区", "Trial time"]
        oft_patterns = ["in_zone_center_*", "in_zone_*", "distance_moved", "x_center", "y_center", "trial_time"]
        result = assess_column_confidence(columns, required_patterns=oft_patterns)
        recognized_raw = {e["raw"] for e in result["recognized"]}
        assert "Trial time" in recognized_raw


class TestConceptMatching:
    """_concept_matches_pattern — concept keywords and full names matching zone patterns."""

    def test_concept_keyword_matches_zone_pattern(self):
        """概念关键词 'center' → 满足 'in_zone_center_*' 模式。"""
        from ethoinsight.catalog.resolve import _concept_matches_pattern

        # concept keyword → zone pattern (new path)
        assert _concept_matches_pattern("center", "in_zone_center_*")
        assert _concept_matches_pattern("open", "in_zone_open*")
        assert _concept_matches_pattern("open_arms", "in_zone_open_arms_*")
        assert _concept_matches_pattern("light", "in_zone_light*")

        # full concept name → zone pattern (also supported)
        assert _concept_matches_pattern("in_zone_center", "in_zone_center_*")
        assert _concept_matches_pattern("in_zone_center_aligned", "in_zone_center_*")
        assert _concept_matches_pattern("in_zone_open_arms_1", "in_zone_open*")

        # non-zone patterns: only fnmatch applies
        assert _concept_matches_pattern("distance_moved", "distance_moved")
        assert not _concept_matches_pattern("distance_moved", "velocity")


class TestApplyAliases:
    """_apply_aliases — now only removes __ignore__ columns, no renaming."""

    def test_ignore_column(self):
        """Column with alias None or __ignore__ is removed."""
        columns = ["center", "border", "x_center"]
        aliases = {"border": None}
        result = _apply_aliases(columns, aliases)
        assert "border" not in result
        assert "center" in result
        assert "x_center" in result

    def test_ignore_column_via_string(self):
        """__ignore__ string also removes the column."""
        columns = ["center", "border"]
        aliases = {"border": "__ignore__"}
        result = _apply_aliases(columns, aliases)
        assert "border" not in result
        assert "center" in result

    def test_columns_preserved_when_not_ignored(self):
        """Non-ignored columns keep their physical names (no renaming to concept names)."""
        columns = ["center", "边缘区", "x_center"]
        aliases = {"center": "in_zone_center", "边缘区": "in_zone_border"}
        result = _apply_aliases(columns, aliases)
        assert result == ["center", "边缘区", "x_center"]

    def test_no_op_when_no_match(self):
        """Columns not in aliases pass through unchanged."""
        columns = ["x_center", "y_center", "distance_moved"]
        aliases = {"center": "in_zone_center_point"}
        result = _apply_aliases(columns, aliases)
        assert result == columns

    def test_input_not_mutated(self):
        """Original list is not mutated."""
        columns = ["center", "border"]
        original = list(columns)
        _apply_aliases(columns, {"center": "in_zone_center_point"})
        assert columns == original


class TestResolveWithColumnAliases:
    """Integration: resolve_metrics with column_aliases — REAL data, REAL normalize.

    CRITICAL regression coverage: earlier drafts assumed 中心区 → "center" via slugify.
    That is FALSE: normalize_column_name("中心区") == "中心区" (Chinese passes through).
    These tests use the actual normalized columns the pipeline produces, and concept-keyword
    aliases (what the LLM writes), to prove the seam works end-to-end at the resolve layer.
    """

    # Real OFT columns after parse_header → normalize_columns (中心区/边缘区 unchanged).
    REAL_OFT_COLUMNS = [
        "trial_time", "recording_time",
        "x_center", "y_center",
        "area", "areachange", "elongation",
        "distance_moved", "velocity",
        "中心区", "边缘区到中心区", "边缘区", "result_1",
    ]

    def test_without_aliases_raises_on_custom_columns(self, tmp_path):
        """Without aliases, real custom zone columns (中心区) cause columns_missing for OFT."""
        try:
            resolve_metrics(
                paradigm="open_field",
                columns=self.REAL_OFT_COLUMNS,
                raw_files=["/mnt/user-data/uploads/test.txt"],
                workspace_dir=str(tmp_path),
                virtual_workspace_dir="/mnt/user-data/workspace",
            )
            assert False, "Should have raised ResolveError"
        except ResolveError as e:
            assert e.code == "columns_missing"
            assert any("in_zone_center" in p for p in e.details.get("missing_patterns", []))

    def test_with_concept_keyword_aliases_resolves(self, tmp_path):
        """REAL pipeline: concept-keyword aliases → metrics resolve + parameters_in_use 用物理列名。"""
        aliases = {
            "中心区": "center",   # concept keyword — matched by _concept_matches_pattern
            "边缘区": "border",   # concept keyword (OFT has no border metric; harmless)
            "边缘区到中心区": "__ignore__",
        }
        plan = resolve_metrics(
            paradigm="open_field",
            columns=self.REAL_OFT_COLUMNS,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases=aliases,
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "center_time_ratio" in metric_ids
        assert "center_distance_ratio" in metric_ids
        assert "center_entry_count" in metric_ids

        # New assertion (spec §4.3): parameters_in_use must use PHYSICAL column names
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "中心区", (
            f"center_zone should be physical '中心区', got {center_metric.parameters_in_use.get('center_zone')}"
        )

    def test_concrete_column_name_aliases_still_work(self, tmp_path):
        """Backward-compat: a concrete column-name target (in_zone_center_point) also works."""
        aliases = {
            "中心区": "in_zone_center_point",  # concrete name — used as-is (not a concept)
            "边缘区到中心区": "__ignore__",
        }
        plan = resolve_metrics(
            paradigm="open_field",
            columns=self.REAL_OFT_COLUMNS,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases=aliases,
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "center_time_ratio" in metric_ids

        # concrete name like "in_zone_center_point" fnmatch-matches "in_zone_center_*"
        # → qualifies as zone concept → physical override injected
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "中心区"

    def test_with_aliases_ignore_removes_column(self, tmp_path):
        """When a column is ignored, it should not cause issues."""
        aliases = {
            "中心区": "center",
            "边缘区到中心区": "__ignore__",
        }
        plan = resolve_metrics(
            paradigm="open_field",
            columns=self.REAL_OFT_COLUMNS,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases=aliases,
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "center_time_ratio" in metric_ids

        # Ignored column "边缘区到中心区" should not appear in columns
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "中心区"


class TestColumnAliasesParametersInUse:
    """column_aliases 产出的 parameters_in_use 必须是物理列名（spec §4.1）。"""

    def test_center_zone_uses_physical_column_name(self, tmp_path):
        """OFT: column_aliases → center_zone 应为物理列。"""
        columns = [
            "trial_time", "recording_time",
            "x_center", "y_center",
            "distance_moved", "velocity",
            "中心区", "边缘区", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="open_field",
            columns=columns,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"中心区": "center", "边缘区": "border"},
        )
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "中心区", (
            f"center_zone should be physical '中心区', got {center_metric.parameters_in_use.get('center_zone')}"
        )

    def test_anonymous_zone_is_wins_over_column_aliases(self, tmp_path):
        """anonymous_zone_is（显式用户指定）优先于 column_aliases 派生覆盖。"""
        columns = [
            "trial_time", "recording_time",
            "x_center", "y_center",
            "distance_moved", "velocity",
            "in_zone", "中心区", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="open_field",
            columns=columns,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"中心区": "center"},
            overrides={"anonymous_zone_is": "in_zone"},
        )
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "in_zone", (
            f"anonymous_zone_is should win, got {center_metric.parameters_in_use.get('center_zone')}"
        )

    def test_zero_maze_open_zones_list_param(self, tmp_path):
        """zero_maze 的 open_zones 是 list[str]（spec §4.2）。"""
        columns = [
            "trial_time", "x_center", "y_center",
            "distance_moved", "velocity",
            "open1", "open2", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="zero_maze",
            columns=columns,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"open1": "open", "open2": "open"},
        )
        zm_metric = next(m for m in plan.metrics if m.id == "open_zone_time_ratio")
        open_zones = zm_metric.parameters_in_use.get("open_zones", [])
        assert isinstance(open_zones, list), f"open_zones should be list, got {type(open_zones)}"
        assert set(open_zones) == {"open1", "open2"}, f"open_zones should be physical cols, got {open_zones}"

    def test_concrete_concept_name_also_produces_physical_override(self, tmp_path):
        """完整概念名 'in_zone_center' 也应产出物理列名覆盖。"""
        columns = [
            "trial_time", "recording_time",
            "x_center", "y_center",
            "distance_moved", "velocity",
            "中心区", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="open_field",
            columns=columns,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"中心区": "in_zone_center"},
        )
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "中心区"

    def test_non_zone_alias_does_not_inject_override(self, tmp_path):
        """非 zone 的 alias 不触发 zone override 注入。"""
        columns = [
            "trial_time", "recording_time",
            "x_center", "y_center",
            "distance_moved", "velocity",
            "in_zone", "result_1",
        ]
        plan = resolve_metrics(
            paradigm="open_field",
            columns=columns,
            raw_files=["/mnt/user-data/uploads/test.txt"],
            workspace_dir=str(tmp_path),
            virtual_workspace_dir="/mnt/user-data/workspace",
            column_aliases={"distance_moved": "velocity"},
            overrides={"anonymous_zone_is": "in_zone"},
        )
        center_metric = next(m for m in plan.metrics if m.id == "center_time_ratio")
        assert center_metric.parameters_in_use["center_zone"] == "in_zone"
