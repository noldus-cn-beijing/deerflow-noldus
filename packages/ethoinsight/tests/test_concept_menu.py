"""Stage 4 生成器纯函数单测 — concept_menu 模块。

测试归属：library 侧（packages/ethoinsight/tests/）。
只测自身逻辑（概念集正确性 + 渲染正确性 + 确定性）；skill md ↔ catalog 一致性
对账测试在 harness 侧 test_concept_menu_staleness.py。
"""

from ethoinsight.catalog.concept_menu import (
    _supported_paradigms,
    list_zone_concepts,
    render_answer_mapping_table,
    render_skill_list,
)
from ethoinsight.catalog.loader import _PARADIGM_ALIASES


class TestListZoneConcepts:
    """概念集正确性 — 依赖 Stage 2（全量）+ Stage 3（border/dark/closed）。"""

    def test_epm_no_center(self):
        concepts = list_zone_concepts("epm")
        assert "open_arms" in concepts
        assert "closed_arms" in concepts
        assert "center" not in concepts

    def test_oft_has_border_no_corner(self):
        concepts = list_zone_concepts("open_field")
        assert "center" in concepts
        assert "border" in concepts
        assert "corner" not in concepts

    def test_ldb_has_dark(self):
        concepts = list_zone_concepts("light_dark_box")
        assert "dark" in concepts
        assert "light" in concepts

    def test_zm_has_closed(self):
        concepts = list_zone_concepts("zero_maze")
        assert "closed" in concepts
        assert "open" in concepts

    def test_fst_empty(self):
        assert list_zone_concepts("forced_swim") == []

    def test_tst_empty(self):
        assert list_zone_concepts("tail_suspension") == []

    def test_deterministic(self):
        first = list_zone_concepts("oft")
        second = list_zone_concepts("oft")
        assert first == second
        assert isinstance(first, list)
        # 验证有序（sorted）
        assert first == sorted(first)


class TestSupportedParadigms:
    """范式清单不内联第三份 — 必须取自 _PARADIGM_ALIASES。"""

    def test_matches_alias_map(self):
        supported = set(_supported_paradigms())
        alias_keys = set(_PARADIGM_ALIASES.keys())
        assert supported == alias_keys


class TestRenderMarkdown:
    """渲染输出正确性。"""

    def test_skill_style_content(self):
        output = render_skill_list()
        # 含预期概念
        assert "EPM" in output
        assert "`open_arms`" in output
        assert "`closed_arms`" in output
        # EPM 不含 center
        assert "EPM: `center`" not in output
        # OFT 含 border、不含 corner
        assert "`border`" in output
        assert "OFT" in output
        assert "`corner`" not in output
        # 空集范式
        assert "无自定义分析区" in output
        assert "FST" in output
        assert "TST" in output

    def test_answer_mapping_style_content(self):
        output = render_answer_mapping_table()
        # 含预期概念
        assert "epm" in output
        assert "`open_arms`" in output
        assert "`closed_arms`" in output
        # EPM 不含 center
        assert "| epm | `center`" not in output
        # OFT 含 border、不含 corner
        assert "`border`" in output
        assert "open_field" in output
        assert "`corner`" not in output
        # 空集范式
        assert "forced_swim" in output
        assert "tail_suspension" in output
        assert "（无自定义分析区）" in output

    def test_both_styles_cover_all_six_paradigms(self):
        from ethoinsight.catalog.concept_menu import _PARADIGM_DISPLAY_SHORT

        for renderer, style in [(render_skill_list, "skill"), (render_answer_mapping_table, "answer_mapping")]:
            output = renderer()
            for p in _supported_paradigms():
                # skill 风格用简称、answer-mapping 风格用 academic key
                expected = _PARADIGM_DISPLAY_SHORT.get(p, p) if style == "skill" else p
                assert expected in output, f"{expected} (paradigm={p}) not found in {renderer.__name__} output"
