"""Tests for Stage 1 requires_columns CNF (Conjunctive Normal Form) support.

§5 of 2026-06-11-pr115-stage1-requires-columns-cnf-spec.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ============================================================================
# §5.1 loader – accept nested / reject malformed (red→green)
# ============================================================================


def test_loader_accepts_nested_requires_columns():
    """Loader accepts nested list[str] in requires_columns."""
    from ethoinsight.catalog.loader import _is_cnf_requires_columns

    assert _is_cnf_requires_columns(["velocity", ["in_zone_center", "in_zone_periphery"]]) is True
    # Pure str list still passes
    assert _is_cnf_requires_columns(["velocity", "distance"]) is True
    # Empty list is valid
    assert _is_cnf_requires_columns([]) is True


def test_loader_rejects_malformed_requires_columns():
    """Loader rejects malformed requires_columns values."""
    from ethoinsight.catalog.loader import _is_cnf_requires_columns

    # Non-list
    assert _is_cnf_requires_columns("velocity") is False
    assert _is_cnf_requires_columns(None) is False

    # Non-str / non-list items
    assert _is_cnf_requires_columns([123]) is False

    # Empty inner group
    assert _is_cnf_requires_columns([[]]) is False

    # Group containing non-str
    assert _is_cnf_requires_columns([["a", 1]]) is False

    # Empty string
    assert _is_cnf_requires_columns([""]) is False

    # Empty string in group
    assert _is_cnf_requires_columns([["a", ""]]) is False


def test_loader_rejects_nested_integration(tmp_path):
    """Loader CatalogError when requires_columns has malformed nested items (integration)."""
    from ethoinsight.catalog.loader import CatalogError, load_catalog

    yaml_content = """
paradigm: test_cnf
ev19_templates:
  - Test Template
default_metrics:
  - id: test_metric
    script: test.script
    requires_columns:
      - velocity
      - []
    output_unit: ratio
    display_name_zh: 测试
    unit_zh: 比例
    one_liner: test
    direction_for_anxiety: null
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
"""
    yaml_path = tmp_path / "test_cnf.yaml"
    yaml_path.write_text(yaml_content)
    with pytest.raises(CatalogError):
        load_catalog("test_cnf", catalog_dir=str(tmp_path))


def test_loader_accepts_nested_integration(tmp_path):
    """Loader successfully parses yaml with nested requires_columns (integration)."""
    from ethoinsight.catalog.loader import load_catalog

    yaml_content = """
paradigm: test_cnf
ev19_templates:
  - Test Template
default_metrics:
  - id: test_metric
    script: test.script
    requires_columns:
      - velocity
      - [in_zone_center, in_zone_periphery]
    output_unit: ratio
    display_name_zh: 测试
    unit_zh: 比例
    one_liner: test
    direction_for_anxiety: null
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
"""
    yaml_path = tmp_path / "test_cnf.yaml"
    yaml_path.write_text(yaml_content)
    cat = load_catalog("test_cnf", catalog_dir=str(tmp_path))
    metric = cat.default_metrics[0]
    assert metric.requires_columns[0] == "velocity"
    assert metric.requires_columns[1] == ["in_zone_center", "in_zone_periphery"]


# ============================================================================
# §5.2 _missing_columns CNF判定 (red→green)
# ============================================================================


def test_missing_columns_or_group_satisfied_by_either():
    """OR-group satisfied when any sub-pattern matches."""
    from ethoinsight.catalog.resolve import _missing_columns

    patterns = [["in_zone_center", "in_zone_periphery"]]
    available = ["in_zone_periphery"]
    missing = _missing_columns(patterns, available)
    assert missing == []


def test_missing_columns_or_group_all_absent():
    """OR-group preserved as list in missing when no sub-pattern matches."""
    from ethoinsight.catalog.resolve import _missing_columns

    patterns = [["in_zone_center", "in_zone_periphery"]]
    available = ["velocity"]
    missing = _missing_columns(patterns, available)
    assert missing == [["in_zone_center", "in_zone_periphery"]]


def test_missing_columns_mixed_cnf():
    """Mixed str + OR-group CNF: all must be satisfied."""
    from ethoinsight.catalog.resolve import _missing_columns

    patterns = ["velocity", ["a", "b"]]
    # Both satisfied
    available = ["velocity", "b"]
    assert _missing_columns(patterns, available) == []

    # Only str satisfied, OR-group absent
    available = ["velocity"]
    assert _missing_columns(patterns, available) == [["a", "b"]]

    # Neither satisfied
    available = ["x"]
    assert _missing_columns(patterns, available) == ["velocity", ["a", "b"]]


def test_missing_columns_or_group_via_alias():
    """OR-group sub-pattern matches via column_aliases concept matching."""
    from ethoinsight.catalog.resolve import _missing_columns

    # Use wildcard patterns so _concept_matches_pattern can extract keywords
    patterns = [["in_zone_center_*", "in_zone_periphery_*"]]
    available = ["中心区"]  # Chinese column name, not matching glob directly
    aliases = {"中心区": "center"}
    # "中心区" → normalize → "center" → alias → "center" → _concept_matches_pattern("center", "in_zone_center_*")
    missing = _missing_columns(patterns, available, column_aliases=aliases)
    assert missing == []


# ============================================================================
# §5.3 字节等价回归 (baseline guardrail, green-before-and-after)
# ============================================================================


def _resolve_snapshot(paradigm: str, columns: list[str]) -> dict:
    """Run the real resolve_metrics path and return a deterministic dict snapshot.

    - Uses the CORRECT signature: resolve_metrics(paradigm, columns, raw_files, workspace_dir).
    - Strips the volatile ``generated_at`` field (timestamp) for byte-stable comparison.
    - Captures a genuine domain ResolveError structurally (NOT a signature TypeError —
      a wrong call must fail loudly, never be swallowed into the baseline).
    """
    from ethoinsight.catalog.resolve import (
        ResolveError,
        plan_metrics_to_dict,
        resolve_metrics,
    )

    try:
        pm = resolve_metrics(paradigm, columns, raw_files=["/tmp/stub.csv"], workspace_dir="/tmp")
    except ResolveError as e:
        return {"resolve_error": str(e), "code": getattr(e, "code", None)}
    d = plan_metrics_to_dict(pm)
    d.pop("generated_at", None)
    return d


# Per-paradigm column sets covering each paradigm's requires_columns globs so the
# real resolve path produces a meaningful plan (exercises _missing_columns happy path).
# These MUST match the sets used to generate fixtures/cnf_baseline/*.json (on dev, pre-CNF).
_BASELINE_COLUMNS: dict[str, list[str]] = {
    "epm": ["in_zone_open_arms_1", "in_zone_closed_arms_1", "x_center", "y_center", "velocity", "distance_moved", "trial_time"],
    "oft": ["in_zone_center_1", "in_zone", "x_center", "y_center", "velocity", "distance_moved", "trial_time"],
    "ldb": ["in_zone_light", "in_zone_dark", "in_zone", "x_center", "y_center", "velocity", "trial_time"],
    "zero_maze": ["in_zone_open_1", "in_zone", "x_center", "y_center", "velocity", "distance_moved", "trial_time"],
    "fst": ["mobility_state_immobile", "velocity", "activity_intensity", "trial_time"],
    "tst": ["mobility_state_immobile", "velocity", "activity_intensity", "trial_time"],
}


def test_pure_list_of_str_resolve_unchanged():
    """All 6 paradigms (pure list[str] catalogs) resolve byte-identically to the
    pre-CNF baseline. This is the executable proof of Fable's "无损" criterion:
    the CNF change must not alter the str-only resolve path.

    Baselines in fixtures/cnf_baseline/*.json were generated on dev (pre-CNF) via
    the same _resolve_snapshot helper — see the test-fix handoff. They contain real
    PlanMetrics dicts (with generated_at stripped), NOT a TypeError error-string.
    """
    baseline_dir = Path(__file__).parent / "fixtures" / "cnf_baseline"

    for paradigm, columns in _BASELINE_COLUMNS.items():
        baseline_path = baseline_dir / f"{paradigm}.json"
        assert baseline_path.exists(), f"Baseline fixture missing: {baseline_path}"

        result = _resolve_snapshot(paradigm, columns)

        # Guard against the previous hollow-green failure mode: a baseline that is
        # just a signature TypeError string. A real baseline has a 'metrics' key
        # (success) or a 'resolve_error' (genuine domain error), never a bare 'error'.
        assert "error" not in result or "resolve_error" in result, (
            f"{paradigm}: _resolve_snapshot produced a non-domain error (likely a wrong "
            f"call signature) — refusing to compare a hollow baseline: {result}"
        )

        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)

        assert result == baseline, (
            f"Resolve output changed for {paradigm} (CNF broke the pure-str path)!\n"
            f"New: {json.dumps(result, default=str, ensure_ascii=False)}\n"
            f"Baseline: {json.dumps(baseline, default=str, ensure_ascii=False)}"
        )



@pytest.mark.parametrize(
    "patterns,available,column_aliases,expected_missing",
    [
        # Single str pattern
        (["velocity"], ["velocity"], None, []),
        (["velocity"], ["distance"], None, ["velocity"]),
        # Multiple str patterns
        (["velocity", "distance"], ["velocity"], None, ["distance"]),
        # All satisfied
        (["velocity", "distance"], ["velocity", "distance"], None, []),
        # None satisfied
        (["velocity", "distance"], ["x", "y"], None, ["velocity", "distance"]),
        # With wildcards
        (["in_zone*"], ["in_zone_center"], None, []),
        (["in_zone_open*"], ["in_zone_center_periphery"], None, ["in_zone_open*"]),
    ],
)
def test_missing_columns_str_path_unchanged(patterns, available, column_aliases, expected_missing):
    """Pure list-of-str _missing_columns result identical to pre-CNF golden values."""
    from ethoinsight.catalog.resolve import _missing_columns

    result = _missing_columns(patterns, available, column_aliases)
    assert result == expected_missing


def test_detect_anonymous_zone_str_path_unchanged():
    """_detect_anonymous_zone with pure str missing_patterns returns same result after flatten addition."""
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.resolve import _detect_anonymous_zone, ResolveError

    # Use a case where zone_unnamed should trigger (missing in_zone_center_* with bare in_zone available)
    # We need a paradigm with anonymous_zone_override; OFT has one
    cat = load_catalog("oft")

    # OFT: missing in_zone_center_* but bare in_zone exists + anonymous_zone_override present
    # This should return ResolveError (zone_unnamed)
    missing = ["in_zone_center_*"]
    available = ["in_zone", "velocity"]
    result = _detect_anonymous_zone(missing, available, {}, cat.anonymous_zone_override)
    assert isinstance(result, ResolveError)
    assert result.code == "zone_unnamed"

    # No zone pattern → None
    result = _detect_anonymous_zone(["velocity"], available, {}, cat.anonymous_zone_override)
    assert result is None

    # Zone pattern but no bare in_zone → None
    result = _detect_anonymous_zone(["in_zone_center_*"], ["velocity"], {}, cat.anonymous_zone_override)
    assert result is None

    # Zone pattern + bare in_zone + override resolved → True
    result = _detect_anonymous_zone(
        ["in_zone_center_*"], available, {"anonymous_zone_is": "center"}, cat.anonymous_zone_override
    )
    assert result is True


# ============================================================================
# §5.4 flatten + hard-crash fix (red→green)
# ============================================================================


def test_flatten_requires_columns():
    """_flatten_requires_columns correctly flattens CNF structure."""
    from ethoinsight.catalog.resolve import _flatten_requires_columns

    # Mixed str + list
    assert _flatten_requires_columns(["a", ["b", "c"], "d"]) == ["a", "b", "c", "d"]

    # Pure str list (shallow copy, order preserved)
    assert _flatten_requires_columns(["a", "b"]) == ["a", "b"]

    # Only groups
    assert _flatten_requires_columns([["a", "b"], ["c"]]) == ["a", "b", "c"]

    # None / [] → []
    assert _flatten_requires_columns(None) == []
    assert _flatten_requires_columns([]) == []

    # Empty list or None passed directly
    assert _flatten_requires_columns(None) == []


def test_detect_anonymous_zone_with_nested_missing():
    """_detect_anonymous_zone does not crash on nested (list) missing_patterns items."""
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.resolve import _detect_anonymous_zone, ResolveError

    cat = load_catalog("oft")

    # Nested missing: an OR-group as a single list item
    missing = [["in_zone_center", "in_zone_periphery"]]
    available = ["in_zone", "velocity"]
    result = _detect_anonymous_zone(missing, available, {}, cat.anonymous_zone_override)
    # Should not crash with AttributeError; should return ResolveError (zone_unnamed)
    assert isinstance(result, ResolveError)
    assert result.code == "zone_unnamed"


def test_build_zone_aliases_overrides_with_nested():
    """_build_zone_aliases_overrides does not crash on nested requires_columns entries."""
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.resolve import _build_zone_aliases_overrides

    # All 6 paradigms: no AttributeError from pat.startswith on list items
    for paradigm in ["epm", "oft", "ldb", "zero_maze", "fst", "tst"]:
        cat = load_catalog(paradigm)
        # With aliases that trigger zone pattern matching
        aliases = {"中心区": "center", "边缘区": "periphery"}
        result = _build_zone_aliases_overrides(aliases, cat, {})
        # Should return a dict (possibly empty), never crash
        assert isinstance(result, dict)
