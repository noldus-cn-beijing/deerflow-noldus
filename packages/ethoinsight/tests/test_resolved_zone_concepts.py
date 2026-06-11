"""Stage 2 — resolved_zone_concepts 统一内部模型内容测试。

TDD red 先行：这些测试在实施前应该全红（Catalog 尚无 resolved_zone_concepts 字段）。
"""

from __future__ import annotations

import pytest

from ethoinsight.catalog.loader import load_catalog


# ============================================================================
# EPM: zone_concept_params 来源
# ============================================================================


def test_epm_resolved_from_zone_concept_params():
    """EPM 的 resolved_zone_concepts 应来自 zone_concept_params，恰 2 条。"""
    cat = load_catalog("epm")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 2, f"EPM should have exactly 2 resolved concepts, got {len(rcs)}: {list(rcs.keys())}"

    # open_arms
    assert "open_arms" in rcs
    open_rc = rcs["open_arms"]
    assert open_rc.concept == "open_arms"
    assert open_rc.binding is not None
    assert open_rc.binding.param == "open_arm_zones"
    assert open_rc.binding.wrap_list is True
    assert open_rc.source == "zone_concept_params"

    # closed_arms
    assert "closed_arms" in rcs
    closed_rc = rcs["closed_arms"]
    assert closed_rc.concept == "closed_arms"
    assert closed_rc.binding is not None
    assert closed_rc.binding.param == "closed_arm_zones"
    assert closed_rc.binding.wrap_list is True
    assert closed_rc.source == "zone_concept_params"


# ============================================================================
# OFT: anonymous_zone_override 来源
# ============================================================================


def test_oft_resolved_concepts():
    """OFT 的 resolved_zone_concepts 应包含 center（override）+ border（explicit_concept, binding=None）。"""
    cat = load_catalog("open_field")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 2, f"OFT should have exactly 2 resolved concepts, got {len(rcs)}: {list(rcs.keys())}"

    # center — from anonymous_zone_override
    assert "center" in rcs
    center_rc = rcs["center"]
    assert center_rc.concept == "center"
    assert center_rc.binding is not None
    assert center_rc.binding.param == "center_zone"
    assert center_rc.binding.wrap_list is False
    assert center_rc.source == "anonymous_zone_override"

    # border — explicit_concept, binding=None (Stage 3)
    assert "border" in rcs
    border_rc = rcs["border"]
    assert border_rc.concept == "border"
    assert border_rc.binding is None
    assert border_rc.source == "explicit_concept"


# ============================================================================
# LDB: anonymous_zone_override 来源
# ============================================================================


def test_ldb_resolved_concepts():
    """LDB 的 resolved_zone_concepts 应包含 light（override）+ dark（zone_concept_params）。"""
    cat = load_catalog("light_dark_box")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 2, f"LDB should have exactly 2 resolved concepts, got {len(rcs)}: {list(rcs.keys())}"

    # light — from anonymous_zone_override
    assert "light" in rcs
    light_rc = rcs["light"]
    assert light_rc.concept == "light"
    assert light_rc.binding is not None
    assert light_rc.binding.param == "light_zone"
    assert light_rc.binding.wrap_list is False
    assert light_rc.source == "anonymous_zone_override"

    # dark — from zone_concept_params (Stage 3)
    assert "dark" in rcs
    dark_rc = rcs["dark"]
    assert dark_rc.concept == "dark"
    assert dark_rc.binding is not None
    assert dark_rc.binding.param == "dark_zone"
    assert dark_rc.binding.wrap_list is False
    assert dark_rc.source == "zone_concept_params"


# ============================================================================
# ZM: anonymous_zone_override 来源
# ============================================================================


def test_zm_resolved_concepts():
    """ZM 的 resolved_zone_concepts 应包含 open（override）+ closed（zone_concept_params）。"""
    cat = load_catalog("zero_maze")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 2, f"ZM should have exactly 2 resolved concepts, got {len(rcs)}: {list(rcs.keys())}"

    # open — from anonymous_zone_override
    assert "open" in rcs
    open_rc = rcs["open"]
    assert open_rc.concept == "open"
    assert open_rc.binding is not None
    assert open_rc.binding.param == "open_zones"
    assert open_rc.binding.wrap_list is True
    assert open_rc.source == "anonymous_zone_override"

    # closed — from zone_concept_params (Stage 3)
    assert "closed" in rcs
    closed_rc = rcs["closed"]
    assert closed_rc.concept == "closed"
    assert closed_rc.binding is not None
    assert closed_rc.binding.param == "closed_zones"
    assert closed_rc.binding.wrap_list is True
    assert closed_rc.source == "zone_concept_params"


# ============================================================================
# derive 同源性：resolved 结果应与现场跑 derive 一致
# ============================================================================


def test_resolved_concept_param_matches_legacy_derive():
    """对 OFT/LDB/ZM，resolved_zone_concepts 里 override 来源条目的 concept
    等于直接对该范式跑 _derive_concept_from_zone_patterns 的返回值。"""
    from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns

    def _collect_zone_patterns(cat):
        zone_patterns: set[str] = set()
        entries = list(cat.default_metrics) + list(cat.optional_metrics) + list(cat.charts)
        for entry in entries:
            for pat in getattr(entry, "requires_columns", []) or []:
                if pat.startswith("in_zone") and "*" in pat:
                    zone_patterns.add(pat)
        return zone_patterns

    for paradigm, expected_concept, target_param in [
        ("open_field", "center", "center_zone"),
        ("light_dark_box", "light", "light_zone"),
        ("zero_maze", "open", "open_zones"),
    ]:
        cat = load_catalog(paradigm)
        zone_patterns = _collect_zone_patterns(cat)
        derived = _derive_concept_from_zone_patterns(zone_patterns, target_param)

        assert derived == expected_concept, (
            f"{paradigm}: derive gave {derived!r}, expected {expected_concept!r}"
        )
        assert derived in cat.resolved_zone_concepts, (
            f"{paradigm}: derived concept {derived!r} not in resolved_zone_concepts"
        )
        assert cat.resolved_zone_concepts[derived].concept == expected_concept


# ============================================================================
# 无 zone 的范式应返回空 dict
# ============================================================================


def test_resolved_default_empty_for_no_zone_paradigm():
    """对无 zone 字段的范式，resolved_zone_concepts 应为空 dict。"""
    for paradigm in ("forced_swim", "tail_suspension"):
        cat = load_catalog(paradigm)
        assert cat.resolved_zone_concepts == {}, (
            f"{paradigm} should have empty resolved_zone_concepts, "
            f"got {cat.resolved_zone_concepts}"
        )


# ============================================================================
# 越界守护：Stage 2 不产出 explicit_concept 来源
# ============================================================================


def test_stage2_does_not_emit_explicit_concept_source_unless_stage3():
    """Stage 2 只产出 zone_concept_params 和 anonymous_zone_override 两种来源。

    Stage 3 的 OFT border 使用 explicit_concept（binding=None），是这个规则的唯一例外。
    """
    for paradigm in ("epm", "open_field", "light_dark_box", "zero_maze", "forced_swim", "tail_suspension"):
        cat = load_catalog(paradigm)
        for rc in cat.resolved_zone_concepts.values():
            # Stage 3 OFT border 使用 explicit_concept 来源 — 这是有意为之
            if paradigm == "open_field" and rc.concept == "border":
                assert rc.source == "explicit_concept", (
                    f"OFT border should have source=explicit_concept, got {rc.source!r}"
                )
                assert rc.binding is None, "OFT border should have binding=None"
                continue
            assert rc.source in ("zone_concept_params", "anonymous_zone_override"), (
                f"{paradigm}: unexpected source {rc.source!r} for concept {rc.concept!r}"
            )


# ============================================================================
# 跨 stage 软缝回归（Stage 1 CNF × Stage 2 loader zone-pattern 收集）
# ============================================================================


def test_loader_zone_collection_tolerates_cnf_nested_requires_columns():
    """Stage 2 loader 的 zone_patterns 收集对 CNF 嵌套 requires_columns 不崩。

    背景（依赖图「软缝」）：Stage 1 把 requires_columns 升为 CNF（item 可为
    list[str] OR-组）。Stage 2 loader 规范化 anonymous_zone_override 时会迭代
    requires_columns 找 in_zone glob——若直接对 list 项调 .startswith 会 AttributeError。
    loader.py 已对该处做内联 flatten（item is list → 展开子 pattern）。本测试用
    直接构造的 MetricEntry（绕过本分支仍是 list[str] 的 loader 校验器）验证该 flatten
    逻辑：含嵌套组的 requires_columns 仍能正确抽出 in_zone glob、且不抛异常。

    注意：本分支的 loader 校验器尚不接受嵌套 yaml（那是 Stage 1 的改动），故无法走
    _parse_catalog 端到端；这里直接验证 loader 内联 flatten 表达式的等价逻辑。
    """
    from ethoinsight.catalog.schema import MetricEntry

    entries = [
        MetricEntry(
            id="m1",
            script="x",
            requires_columns=["velocity", ["in_zone_center_*", "in_zone_periphery_*"]],
            output_unit="ratio",
            display_name_zh="d",
            unit_zh="u",
            one_liner="o",
            direction_for_anxiety=None,
            statistical_default="groupwise_compare",
        ),
        MetricEntry(
            id="m2",
            script="y",
            requires_columns=["in_zone_open_*"],  # pure str item
            output_unit="count",
            display_name_zh="d",
            unit_zh="u",
            one_liner="o",
            direction_for_anxiety=None,
            statistical_default="groupwise_compare",
        ),
    ]

    # 复刻 loader.py 规范化段的 in_zone glob 收集（含内联 flatten），断言不崩 + 结果正确。
    zone_patterns: set[str] = set()
    for entry in entries:
        for item in getattr(entry, "requires_columns", []) or []:
            sub_patterns = item if isinstance(item, list) else [item]
            for pat in sub_patterns:
                if pat.startswith("in_zone") and "*" in pat:
                    zone_patterns.add(pat)

    assert zone_patterns == {"in_zone_center_*", "in_zone_periphery_*", "in_zone_open_*"}
