"""Unit tests for EV19 facts table (62 variants)."""

from ethoinsight.ev19_facts import (
    EV19_VARIANTS,
    EV19_CATEGORIES,
    EV19_TEMPLATE_PARADIGM_MAP,
    SUPPORTED_PARADIGMS_V01,
    is_valid_ev19_template,
    get_template_facts,
    suggest_nearby_templates,
    get_default_template_for_paradigm,
    is_paradigm_template_compatible,
)


def test_ev19_variants_count():
    """62 variants are loaded."""
    assert len(EV19_VARIANTS) == 62


def test_ev19_categories_count():
    """20 unique categories."""
    assert len(EV19_CATEGORIES) == 20


def test_known_variant_is_valid():
    """A known variant ID passes validation."""
    assert is_valid_ev19_template("PlusMaze-AllZones") is True


def test_unknown_variant_is_invalid():
    """An unknown variant ID fails validation."""
    assert is_valid_ev19_template("Bogus-Template") is False


def test_get_template_facts_returns_full_record():
    """Facts include arena_template, zone_template, etc."""
    facts = get_template_facts("PlusMaze-AllZones")
    assert facts is not None
    assert facts["arena_template"] == "Elevated plus maze"
    assert facts["category"] == "PlusMaze"


def test_get_template_facts_returns_none_for_unknown():
    assert get_template_facts("Bogus") is None


def test_suggest_nearby_templates_for_typo():
    """Typo in PlusMaze suggests close matches."""
    suggestions = suggest_nearby_templates("PlusMze-AllZones")
    assert any("PlusMaze" in s for s in suggestions)


def test_paradigm_map_contains_known_paradigms():
    """epm and forced_swim are in paradigm map."""
    assert "epm" in EV19_TEMPLATE_PARADIGM_MAP
    assert "forced_swim" in EV19_TEMPLATE_PARADIGM_MAP


def test_paradigm_map_epm_includes_plusmaze_variants():
    """EPM should map to PlusMaze variants."""
    epm_templates = EV19_TEMPLATE_PARADIGM_MAP["epm"]
    assert "PlusMaze-AllZones" in epm_templates
    assert "PlusMaze-FewZones" in epm_templates
    assert "PlusMaze-NoZones" in epm_templates


def test_get_default_template_for_paradigm():
    """epm default = PlusMaze-AllZones."""
    assert get_default_template_for_paradigm("epm") == "PlusMaze-AllZones"


def test_get_default_template_for_unknown_paradigm_returns_none():
    assert get_default_template_for_paradigm("nonexistent") is None


def test_is_paradigm_template_compatible_known_pair():
    """PlusMaze-AllZones is compatible with epm."""
    assert is_paradigm_template_compatible("epm", "PlusMaze-AllZones") is True


def test_is_paradigm_template_compatible_mismatch():
    """PorsoltCylinder template is NOT compatible with epm."""
    assert is_paradigm_template_compatible("epm", "PorsoltCylinder-AllZones") is False


def test_is_paradigm_template_compatible_unknown_paradigm():
    """Unknown paradigm returns False."""
    assert is_paradigm_template_compatible("nonexistent", "PlusMaze-AllZones") is False


# ---------------------------------------------------------------------------
# TST (Tail Suspension Test) support — added 2026-06-01
# ---------------------------------------------------------------------------


def test_tail_suspension_is_in_supported_paradigms_v01():
    """tail_suspension must be in the v0.1 whitelist so Gate 1 recognizes it."""
    assert "tail_suspension" in SUPPORTED_PARADIGMS_V01


def test_tail_suspension_default_template_is_notemplate():
    """TST maps to NoTemplate (no zone-based analysis)."""
    assert get_default_template_for_paradigm("tail_suspension") == "NoTemplate"


def test_tail_suspension_paradigm_map_entry_exists():
    """Paradigm map has a tail_suspension entry."""
    assert "tail_suspension" in EV19_TEMPLATE_PARADIGM_MAP
    assert EV19_TEMPLATE_PARADIGM_MAP["tail_suspension"] == ["NoTemplate"]


def test_supported_paradigms_v01_has_six_paradigms():
    """v0.1 now supports 6 paradigms (5 original + TST)."""
    expected = {"epm", "open_field", "zero_maze", "light_dark_box", "forced_swim", "tail_suspension"}
    assert SUPPORTED_PARADIGMS_V01 == expected


# ---------------------------------------------------------------------------
# TST 不跨范式归类锁定（回归锚点） — 2026-06-04
#
# 行为学同事在 issue #72 指出「PorsoltCylinder-NoZones 和 NoTemplate 本质上是一个
# 东西，TST 用的就是强迫游泳模板」（机器结构层确实如此：都是单观察区、无分析区）。
# 但产品原则是「不跨范式复用/归类，哪怕范式间 overlap 也接受重复」：TST 是独立范式，
# 只归到结构中性的 NoTemplate，不挂到 FST 专属的 PorsoltCylinder 大类下。
# 这组测试把该边界钉死，防止未来有人照同事措辞把 PorsoltCylinder 加进 TST 候选
# （会让 identify 在真实无 zone 数据上变 ambiguous，多出一次无意义反问）。
# ---------------------------------------------------------------------------


def test_tail_suspension_not_classified_under_porsolt_nozones():
    """TST 不跨范式归到 FST 的 PorsoltCylinder-NoZones（产品原则：不跨范式归类）。"""
    assert is_paradigm_template_compatible("tail_suspension", "PorsoltCylinder-NoZones") is False


def test_tail_suspension_not_classified_under_porsolt_allzones():
    """TST 也不归到 PorsoltCylinder-AllZones（FST 潜水检测专属变体）。"""
    assert is_paradigm_template_compatible("tail_suspension", "PorsoltCylinder-AllZones") is False


def test_tail_suspension_map_has_single_unambiguous_candidate():
    """TST 候选列表保持单值，identify 在真实无 zone 数据上不应变 ambiguous。"""
    assert EV19_TEMPLATE_PARADIGM_MAP["tail_suspension"] == ["NoTemplate"]


def test_forced_swim_keeps_its_own_porsolt_templates():
    """对照：FST 仍独占 PorsoltCylinder 模板，未被 TST 改动波及。"""
    assert is_paradigm_template_compatible("forced_swim", "PorsoltCylinder-NoZones") is True
    assert is_paradigm_template_compatible("forced_swim", "PorsoltCylinder-AllZones") is True
    assert is_paradigm_template_compatible("forced_swim", "NoTemplate") is False
