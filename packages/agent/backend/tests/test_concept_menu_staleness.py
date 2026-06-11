"""Stage 4 harness 侧 staleness / 覆盖度对账测试。

测试归属：harness 侧（packages/agent/backend/tests/）。
依赖方向 harness→library 顺方向，安全。

staleness 对账：整文件字节比对已提交 .generated.md ≡ 重新生成。
这是漂移回归探针（抓人手改生成文件不改 catalog），生成内容正确性由
library 侧 test_concept_menu.py + 本文件覆盖度断言承担。
"""

from __future__ import annotations

from pathlib import Path

from ethoinsight.catalog.concept_menu import (
    _supported_paradigms,
    list_zone_concepts,
    render_answer_mapping_table,
    render_skill_list,
)

# skills/custom/ 根目录（从 backend/tests/ 向上两层到 packages/agent/）
_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills" / "custom"
_CONFIRMATION_REF = (
    _SKILLS_ROOT / "ethoinsight-column-confirmation" / "references"
)
_GENERATED_SKILL = _CONFIRMATION_REF / "zone-concepts.generated.md"
_GENERATED_MAPPING = _CONFIRMATION_REF / "zone-concepts-mapping.generated.md"

_FIX_INSTRUCTION = (
    "运行 `make gen-references` 后重新提交以消除漂移"
)


class TestStalenessSkillStyle:
    """整文件字节比对 — 抓人手改生成文件不更新 catalog 的漂移。"""

    def test_generated_file_matches_render(self):
        committed = _GENERATED_SKILL.read_text(encoding="utf-8")
        regenerated = render_skill_list()
        assert committed == regenerated, (
            f"zone-concepts.generated.md 与重新生成结果不一致（人手改过生成文件？）。\n"
            f"{_FIX_INSTRUCTION}"
        )


class TestStalenessAnswerMappingStyle:
    def test_generated_file_matches_render(self):
        committed = _GENERATED_MAPPING.read_text(encoding="utf-8")
        regenerated = render_answer_mapping_table()
        assert committed == regenerated, (
            f"zone-concepts-mapping.generated.md 与重新生成结果不一致（人手改过生成文件？）。\n"
            f"{_FIX_INSTRUCTION}"
        )


class TestCoverage:
    """覆盖度断言 — 承担"内容正确"职责（staleness 只抓漂移）。"""

    def test_covers_all_six_paradigms(self):
        paradigms = _supported_paradigms()
        assert len(paradigms) == 6, (
            f"预期 6 个范式，实际 {len(paradigms)}: {paradigms}"
        )
        expected = {"forced_swim", "tail_suspension", "open_field",
                    "light_dark_box", "epm", "zero_maze"}
        assert set(paradigms) == expected

    def test_menu_has_no_center_in_epm(self):
        concepts = list_zone_concepts("epm")
        assert "center" not in concepts, (
            "EPM center 是双存时期的漂移项，不应出现在菜单中"
        )

    def test_menu_has_no_corner_in_oft(self):
        concepts = list_zone_concepts("open_field")
        assert "corner" not in concepts, (
            "OFT corner 是双存时期的漂移项，不应出现在菜单中"
        )

    def test_menu_has_border_in_oft(self):
        concepts = list_zone_concepts("open_field")
        assert "border" in concepts, (
            "OFT border 经 Fable 决策门 1 闭合为 binding=None 一等概念，"
            "必须出现在菜单中（Stage 3 产出）"
        )

    def test_menu_has_dark_in_ldb(self):
        concepts = list_zone_concepts("light_dark_box")
        assert "dark" in concepts, (
            "LDB dark 是 Stage 3 产出，必须出现在菜单中"
        )

    def test_menu_has_closed_in_zm(self):
        concepts = list_zone_concepts("zero_maze")
        assert "closed" in concepts, (
            "ZM closed 是 Stage 3 产出，必须出现在菜单中"
        )

    def test_empty_paradigms_have_no_concepts(self):
        for p in ("forced_swim", "tail_suspension"):
            concepts = list_zone_concepts(p)
            assert concepts == [], (
                f"{p} 应返回空列表，实际: {concepts}"
            )
