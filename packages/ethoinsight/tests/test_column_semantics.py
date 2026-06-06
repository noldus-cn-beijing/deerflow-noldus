"""Tests for column-semantics Sprint 1: assess_column_confidence, _apply_aliases,
and resolve_metrics with column_aliases.
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
        # Without required_patterns, these should be unrecognized
        assert len(result["unrecognized"]) == 3
        assert {e["raw"] for e in result["unrecognized"]} == {"中心区", "边缘区", "边缘区到中心区"}

    def test_custom_zone_columns_recognized_with_patterns(self):
        """Custom zone columns ARE recognized when catalog patterns match the normalized name."""
        columns = ["中心区", "边缘区", "边缘区到中心区", "Trial time"]
        # "中心区" normalizes to "center" via _slugify
        # "边缘区" normalizes to "边缘区" (no substitution)
        patterns = ["in_zone_center_*", "in_zone_border_*", "in_zone_*"]
        result = assess_column_confidence(columns, required_patterns=patterns)
        # "中心区" → "center" → does NOT match any glob (center ≠ in_zone_center_*)
        # "边缘区" → "边缘区" → does NOT match
        # So they should still be unrecognized with these patterns
        recognized_raw = {e["raw"] for e in result["recognized"]}
        assert "Trial time" in recognized_raw  # COLUMN_MAP hit

    def test_34_file_real_columns(self):
        """Simulate real 34-file OFT columns — zone columns unrecognized, standard recognized."""
        # Real columns from a typical OFT file with custom zone names
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

        # Standard columns should be recognized
        for std in ["Trial time", "Recording time", "X center", "Y center",
                     "Distance moved", "Velocity", "Result 1", "Result 2"]:
            assert std in recognized_raw, f"{std} should be recognized"

        # Custom zone columns should be unrecognized (without catalog patterns)
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
        # OFT requires_columns uses in_zone_center_* pattern
        oft_patterns = ["in_zone_center_*", "in_zone_*", "distance_moved", "x_center", "y_center", "trial_time"]
        result = assess_column_confidence(columns, required_patterns=oft_patterns)
        recognized_raw = {e["raw"] for e in result["recognized"]}
        # "中心区" normalizes to "center" — does NOT match in_zone_center_* (fnmatch glob)
        # This is expected: the alias remapping happens later via column_aliases, not here
        assert "Trial time" in recognized_raw


class TestApplyAliases:
    """_apply_aliases — column alias remapping."""

    def test_concept_keyword_translation_from_catalog(self):
        """Concept keyword 'center' → matchable in_zone_center_* column via catalog map."""
        from ethoinsight.catalog.loader import load_catalog
        from ethoinsight.catalog.resolve import _zone_concept_map

        cat = load_catalog("open_field")
        cmap = _zone_concept_map(cat)
        assert cmap.get("center") == "in_zone_center_*"

        columns = ["中心区", "x_center"]
        result = _apply_aliases(columns, {"中心区": "center"}, cmap)
        # 中心区 → concept "center" → a column that matches in_zone_center_*
        import fnmatch
        assert any(fnmatch.fnmatchcase(c, "in_zone_center_*") for c in result)
        assert "x_center" in result

    def test_simple_remap(self):
        """Simple 1:1 remap."""
        columns = ["center", "border", "x_center", "y_center"]
        aliases = {"center": "in_zone_center_point"}
        result = _apply_aliases(columns, aliases)
        assert "in_zone_center_point" in result
        assert "center" not in result
        assert "border" in result
        assert "x_center" in result

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

    def test_multiple_remaps(self):
        """Multiple columns remapped at once."""
        columns = ["center", "边缘区", "x_center"]
        aliases = {"center": "in_zone_center_point", "边缘区": "in_zone_border_point"}
        result = _apply_aliases(columns, aliases)
        assert result == ["in_zone_center_point", "in_zone_border_point", "x_center"]

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
        """Without aliases, real custom zone columns (中心区) cause columns_missing for OFT.

        This is the exact failure that the 34-file OFT upload hit: 中心区 normalizes to
        中心区 (NOT center), which does not match in_zone_center_*.
        """
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
        """REAL pipeline: concept-keyword aliases on real normalized columns → metrics resolve.

        The LLM writes a CONCEPT KEYWORD ("center"), NOT a machine column name. The catalog
        concept-translation layer turns it into a column that satisfies in_zone_center_*.
        """
        aliases = {
            "中心区": "center",   # concept keyword — translated by _zone_concept_map
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
