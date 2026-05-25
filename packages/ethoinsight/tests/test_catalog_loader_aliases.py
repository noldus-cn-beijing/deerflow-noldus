"""Test paradigm key alias resolution in catalog/loader.load_catalog().

Background: identify_ev19_template / prep_metric_plan / metrics dispatcher /
skill docs all use academic-name paradigm keys (e.g. "forced_swim").
Catalog YAML files historically use shortened stems (fst.yaml). load_catalog
must accept BOTH and resolve correctly.
"""

from __future__ import annotations

import pytest

from ethoinsight.catalog.loader import CatalogError, load_catalog


# Academic name → expected catalog paradigm field value (which equals the stem)
_ACADEMIC_TO_FILENAME: dict[str, str] = {
    "forced_swim": "fst",
    "tail_suspension": "tst",
    "open_field": "oft",
    "light_dark_box": "ldb",
}

# Catalog keys that don't need aliasing (academic name == filename stem)
_ALREADY_ALIGNED = ["epm", "zero_maze", "shoaling"]


@pytest.mark.parametrize("academic,filename_stem", list(_ACADEMIC_TO_FILENAME.items()))
def test_load_catalog_accepts_academic_name(academic: str, filename_stem: str) -> None:
    """load_catalog(academic_name) should resolve to the abbreviated YAML file."""
    cat = load_catalog(academic)
    # Catalog's internal paradigm field still uses the filename stem (unchanged)
    assert cat.paradigm == filename_stem, (
        f"load_catalog('{academic}') should return catalog with paradigm='{filename_stem}', "
        f"got '{cat.paradigm}'"
    )


@pytest.mark.parametrize("academic,filename_stem", list(_ACADEMIC_TO_FILENAME.items()))
def test_load_catalog_still_accepts_filename_stem(academic: str, filename_stem: str) -> None:
    """Backward compat: passing the short filename stem must still work."""
    cat = load_catalog(filename_stem)
    assert cat.paradigm == filename_stem


@pytest.mark.parametrize("paradigm", _ALREADY_ALIGNED)
def test_load_catalog_already_aligned_paradigms(paradigm: str) -> None:
    """Paradigms where academic name == filename stem (epm/zero_maze/shoaling)."""
    cat = load_catalog(paradigm)
    assert cat.paradigm == paradigm


def test_load_catalog_unknown_paradigm_reports_alias_resolution() -> None:
    """Error message should mention both the input and the resolved filename
    so debugging is straightforward."""
    with pytest.raises(CatalogError) as exc:
        load_catalog("not_a_real_paradigm")
    msg = str(exc.value)
    assert "not_a_real_paradigm" in msg


def test_load_catalog_unknown_academic_alias_falls_through() -> None:
    """If an academic name isn't in the alias map, it's used as-is. Make sure
    that produces a clean unknown-paradigm error (not a KeyError or other crash)."""
    with pytest.raises(CatalogError) as exc:
        load_catalog("morris_water_maze")  # not in alias map, not a real catalog file
    assert "morris_water_maze" in str(exc.value)
