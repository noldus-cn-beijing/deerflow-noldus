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


def test_oft_resolved_from_override():
    """OFT 的 resolved_zone_concepts 应通过 derive 从 anonymous_zone_override 规范化。"""
    cat = load_catalog("open_field")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 1, f"OFT should have exactly 1 resolved concept, got {len(rcs)}: {list(rcs.keys())}"

    assert "center" in rcs
    center_rc = rcs["center"]
    assert center_rc.concept == "center"
    assert center_rc.binding is not None
    assert center_rc.binding.param == "center_zone"
    assert center_rc.binding.wrap_list is False
    assert center_rc.source == "anonymous_zone_override"


# ============================================================================
# LDB: anonymous_zone_override 来源
# ============================================================================


def test_ldb_resolved_from_override():
    """LDB 的 resolved_zone_concepts 应通过 derive 从 anonymous_zone_override 规范化。"""
    cat = load_catalog("light_dark_box")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 1, f"LDB should have exactly 1 resolved concept, got {len(rcs)}: {list(rcs.keys())}"

    assert "light" in rcs
    light_rc = rcs["light"]
    assert light_rc.concept == "light"
    assert light_rc.binding is not None
    assert light_rc.binding.param == "light_zone"
    assert light_rc.binding.wrap_list is False
    assert light_rc.source == "anonymous_zone_override"


# ============================================================================
# ZM: anonymous_zone_override 来源
# ============================================================================


def test_zm_resolved_from_override():
    """ZM 的 resolved_zone_concepts 应通过 derive 从 anonymous_zone_override 规范化。"""
    cat = load_catalog("zero_maze")
    rcs = cat.resolved_zone_concepts

    assert len(rcs) == 1, f"ZM should have exactly 1 resolved concept, got {len(rcs)}: {list(rcs.keys())}"

    assert "open" in rcs
    open_rc = rcs["open"]
    assert open_rc.concept == "open"
    assert open_rc.binding is not None
    assert open_rc.binding.param == "open_zones"
    assert open_rc.binding.wrap_list is True
    assert open_rc.source == "anonymous_zone_override"


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


def test_stage2_does_not_emit_explicit_concept_source():
    """Stage 2 只产出 zone_concept_params 和 anonymous_zone_override 两种来源。"""
    for paradigm in ("epm", "open_field", "light_dark_box", "zero_maze", "forced_swim", "tail_suspension"):
        cat = load_catalog(paradigm)
        for rc in cat.resolved_zone_concepts.values():
            assert rc.source in ("zone_concept_params", "anonymous_zone_override"), (
                f"{paradigm}: unexpected source {rc.source!r} for concept {rc.concept!r}"
            )
