"""Tests for PARADIGMS registry and verify_paradigm_columns in ethoinsight.templates."""

import tempfile
from pathlib import Path

import pytest


class TestParadigmsRegistry:
    def test_registry_contains_18_paradigms(self):
        from ethoinsight.templates import PARADIGMS

        assert isinstance(PARADIGMS, dict)
        assert len(PARADIGMS) >= 18
        # shoaling must be "ready"
        assert PARADIGMS["shoaling"]["status"] == "ready"
        assert PARADIGMS["shoaling"]["cn"] == "斑马鱼鱼群行为"
        assert PARADIGMS["shoaling"]["subject"] == "fish"
        assert PARADIGMS["shoaling"]["category"] == "zebrafish"

    def test_list_paradigms_ready_only(self):
        from ethoinsight.templates import list_paradigms

        ready = list_paradigms(status="ready")
        assert len(ready) == 1
        assert ready[0]["name"] == "shoaling"

    def test_list_paradigms_all(self):
        from ethoinsight.templates import list_paradigms

        all_p = list_paradigms()
        assert len(all_p) >= 18

    def test_list_paradigms_by_category(self):
        from ethoinsight.templates import list_paradigms

        anxiety = list_paradigms(category="anxiety")
        names = {p["name"] for p in anxiety}
        assert "epm" in names
        assert "zero_maze" in names
        assert "light_dark" in names
        # category filtering excludes non-anxiety paradigms
        assert "shoaling" not in names
        assert "morris_water_maze" not in names

    def test_list_categories_returns_all_7(self):
        from ethoinsight.templates import list_categories

        cats = list_categories()
        assert len(cats) == 7
        assert cats[0]["name"] == "open_field"
        assert cats[0]["cn"] == "旷场及物体识别"

    def test_get_paradigm_returns_dict_with_new_fields(self):
        from ethoinsight.templates import get_paradigm

        p = get_paradigm("shoaling")
        assert p is not None
        assert p["cn"] == "斑马鱼鱼群行为"
        assert "expected_columns" in p
        assert "subject" in p
        assert "category" in p
        assert "ev19_arena_templates" in p

    def test_get_paradigm_unknown_returns_none(self):
        from ethoinsight.templates import get_paradigm

        assert get_paradigm("nonexistent_paradigm") is None


class TestVerifyParadigmColumns:
    def test_match_returns_true(self):
        from ethoinsight.templates import verify_paradigm_columns

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("X_center,Y_center,velocity,IID,NND,extra_col\n")
            f.write("1.0,2.0,3.0,4.0,5.0,6.0\n")
            tmp_path = f.name

        try:
            result = verify_paradigm_columns("shoaling", tmp_path)
            assert result is not None
            assert result["match"] is True
            assert len(result["missing"]) == 0
        finally:
            Path(tmp_path).unlink()

    def test_mismatch_returns_false_with_missing(self):
        from ethoinsight.templates import verify_paradigm_columns

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Col_A,Col_B,Col_C\n")
            f.write("1,2,3\n")
            tmp_path = f.name

        try:
            result = verify_paradigm_columns("shoaling", tmp_path)
            assert result is not None
            assert result["match"] is False
            assert len(result["missing"]) > 0
        finally:
            Path(tmp_path).unlink()

    def test_file_not_found_returns_none(self):
        from ethoinsight.templates import verify_paradigm_columns

        result = verify_paradigm_columns("shoaling", "/nonexistent/path.csv")
        assert result is None

    def test_unknown_paradigm_raises_valueerror(self):
        from ethoinsight.templates import verify_paradigm_columns

        with pytest.raises(ValueError, match="Unknown paradigm"):
            verify_paradigm_columns("nonexistent", "/some/path.csv")
