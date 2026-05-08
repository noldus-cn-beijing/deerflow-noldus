"""Unit tests for EV19 facts table (62 variants)."""

from ethoinsight.ev19_facts import (
    EV19_VARIANTS,
    EV19_CATEGORIES,
    EV19_TEMPLATE_PARADIGM_MAP,
    is_valid_ev19_template,
    get_template_facts,
    suggest_nearby_templates,
    get_default_template_for_paradigm,
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
