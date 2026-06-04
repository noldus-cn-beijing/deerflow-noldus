"""Tests for 2026-06-04 zone override unification across three paradigms.

Covers:
- Bare in_zone with no override → zone_unnamed (OFT / zero_maze / LDB)
- Unified key anonymous_zone_is translation → real parameters
- List wrapping for zero_maze open_zones
- _detect_anonymous_zone with anonymous_zone_override parameter
- True missing columns not misidentified
- Override not leaking to non-zone metrics
- Loader: list default acceptance + anonymous_zone_override parsing
- EPM/FST/TST not affected (no anonymous_zone_override declared)
"""

from __future__ import annotations

import pytest


# ============================================================
# Column fixtures
# ============================================================

def _anonymous_zone_columns(*extra_cols: str) -> list[str]:
    """Generic anonymous zone fixture: bare in_zone but no named zone columns."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "in_zone",
        "distance_moved",
        *extra_cols,
    ]


def _no_in_zone_columns() -> list[str]:
    """No in_zone column at all."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "distance_moved",
    ]


def _named_oft_columns() -> list[str]:
    """OFT with proper in_zone_center_* columns."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "in_zone_center_center_point",
        "distance_moved",
    ]


# ============================================================
# zone_unnamed detection — three paradigms
# ============================================================

class TestZoneUnnamedThreeParadigms:
    """Bare in_zone with no override → zone_unnamed for OFT / zero_maze / LDB."""

    def test_oft_anonymous_zone_raises_zone_unnamed(self):
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=_anonymous_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "zone_unnamed"
        assert exc.value.details["found_column"] == "in_zone"
        assert any("in_zone_center" in p for p in exc.value.details["missing_patterns"])

    def test_zero_maze_anonymous_zone_raises_zone_unnamed(self):
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="zero_maze",
                columns=_anonymous_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "zone_unnamed"

    def test_ldb_anonymous_zone_raises_zone_unnamed(self):
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="ldb",
                columns=_anonymous_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "zone_unnamed"


# ============================================================
# Unified key translation — core mechanism
# ============================================================

class TestUnifiedKeyTranslation:
    """anonymous_zone_is → paradigm-specific real parameter."""

    def test_oft_unified_key_translates_to_center_zone(self):
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="oft",
            columns=_anonymous_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"anonymous_zone_is": "in_zone"},
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "center_time_ratio" in metric_ids
        assert "center_distance_ratio" in metric_ids
        assert "center_entry_count" in metric_ids
        assert "center_time" in metric_ids
        assert "center_distance" in metric_ids
        # center_zone should be in parameters_in_use for center metrics
        for m in plan.metrics:
            if m.id.startswith("center_"):
                assert m.parameters_in_use.get("center_zone") == "in_zone", (
                    f"{m.id} should have center_zone=in_zone, got {m.parameters_in_use}"
                )
            else:
                assert "center_zone" not in m.parameters_in_use, (
                    f"{m.id} must not have center_zone but got {m.parameters_in_use}"
                )

    def test_ldb_unified_key_translates_to_light_zone(self):
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="ldb",
            columns=_anonymous_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"anonymous_zone_is": "in_zone"},
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "light_time_ratio" in metric_ids
        assert "transition_count" in metric_ids
        assert "light_latency" in metric_ids
        # light_zone should be in parameters_in_use for light metrics
        for m in plan.metrics:
            assert m.parameters_in_use.get("light_zone") == "in_zone", (
                f"{m.id} should have light_zone=in_zone, got {m.parameters_in_use}"
            )

    def test_zero_maze_unified_key_translates_to_open_zones_list(self):
        """List wrapping: zero_maze open_zones must be a list[str]."""
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="zero_maze",
            columns=_anonymous_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"anonymous_zone_is": "in_zone"},
        )
        metric_ids = {m.id for m in plan.metrics}
        assert "open_zone_time_ratio" in metric_ids
        assert "open_zone_time" in metric_ids
        assert "open_zone_distance" in metric_ids
        assert "hesitation_count" in metric_ids
        # open_zones must be a list of str
        for m in plan.metrics:
            oz = m.parameters_in_use.get("open_zones")
            assert isinstance(oz, list), (
                f"{m.id}: open_zones must be list, got {type(oz).__name__}: {oz!r}"
            )
            assert oz == ["in_zone"], (
                f"{m.id}: open_zones should be ['in_zone'], got {oz!r}"
            )

    def test_zero_maze_open_zones_is_list_not_str(self):
        """Regression guard: str→list wrapping prevents char-by-char iteration bug."""
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="zero_maze",
            columns=_anonymous_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"anonymous_zone_is": "in_zone"},
        )
        for m in plan.metrics:
            oz = m.parameters_in_use.get("open_zones")
            assert not isinstance(oz, str), (
                f"{m.id}: open_zones should NOT be str (char-iteration bug), "
                f"got str: {oz!r}"
            )


# ============================================================
# Compute with list parameter — zero_maze
# ============================================================

class TestZeroMazeComputeWithListParam:
    """compute_open_zone_time_ratio accepts open_zones=["in_zone"] and produces real value."""

    def test_compute_open_zone_time_ratio_with_list(self):
        import pandas as pd
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = pd.DataFrame({
            "in_zone": [1, 1, 0, 0, 1],
            "trial_time": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        result = compute_open_zone_time_ratio(df, open_zones=["in_zone"])
        assert result is not None
        # 3 out of 5 in_zone=1 → ratio ≈ 0.6
        assert 0.5 < result < 0.7

    def test_compute_open_zone_time_with_list(self):
        import pandas as pd
        from ethoinsight.metrics.zero_maze import compute_open_zone_time

        df = pd.DataFrame({
            "in_zone": [1, 1, 0, 0, 1],
            "trial_time": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        result = compute_open_zone_time(df, open_zones=["in_zone"])
        assert result is not None
        assert result > 0


# ============================================================
# _detect_anonymous_zone unit tests — new signature
# ============================================================

class TestDetectAnonymousZoneNewSignature:
    """_detect_anonymous_zone with anonymous_zone_override parameter."""

    def test_paradigm_declares_azo_and_override_has_unified_key_returns_true(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone
        from ethoinsight.catalog.schema import AnonymousZoneOverride

        azo = AnonymousZoneOverride(target_param="center_zone", wrap_list=False)
        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={"anonymous_zone_is": "in_zone"},
            anonymous_zone_override=azo,
        )
        assert result is True

    def test_paradigm_declares_azo_but_no_unified_key_in_overrides_returns_error(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone
        from ethoinsight.catalog.schema import AnonymousZoneOverride

        azo = AnonymousZoneOverride(target_param="center_zone", wrap_list=False)
        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={},
            anonymous_zone_override=azo,
        )
        assert result is not None
        assert result.code == "zone_unnamed"

    def test_paradigm_not_declared_azo_returns_none(self):
        """Paradigm without anonymous_zone_override (e.g. EPM) → None (not triggered)."""
        from ethoinsight.catalog.resolve import _detect_anonymous_zone

        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={},
            anonymous_zone_override=None,
        )
        assert result is None

    def test_old_center_zone_override_no_longer_works(self):
        """Legacy center_zone override (without unified key) no longer resolves.
        Only anonymous_zone_is is recognized as the unified key."""
        from ethoinsight.catalog.resolve import _detect_anonymous_zone
        from ethoinsight.catalog.schema import AnonymousZoneOverride

        azo = AnonymousZoneOverride(target_param="center_zone", wrap_list=False)
        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={"center_zone": "in_zone"},  # old-style — no longer accepted
            anonymous_zone_override=azo,
        )
        # Should return error because center_zone override is NOT the unified key
        assert result is not None
        assert result.code == "zone_unnamed"

    def test_unified_key_with_zero_maze_wrap_list_true(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone
        from ethoinsight.catalog.schema import AnonymousZoneOverride

        azo = AnonymousZoneOverride(target_param="open_zones", wrap_list=True)
        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_open*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={"anonymous_zone_is": "in_zone"},
            anonymous_zone_override=azo,
        )
        assert result is True


# ============================================================
# True missing columns — no false zone_unnamed
# ============================================================

class TestTrueMissingColumns:
    """Non-zone missing columns still raise columns_missing."""

    def test_no_in_zone_at_all_raises_columns_missing(self):
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=_no_in_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "columns_missing"

    def test_named_zone_but_missing_distance_raises_columns_missing(self):
        from ethoinsight.catalog import resolve, ResolveError

        cols = ["time", "trial_time", "x_center", "y_center", "in_zone_center_center_point"]
        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=cols,
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "columns_missing"


# ============================================================
# Override not leaking
# ============================================================

class TestOverrideNotLeaking:
    """Override should only affect metrics that declare the zone parameter."""

    def test_oft_unified_key_does_not_leak_to_optional_metrics(self):
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="oft",
            columns=_anonymous_zone_columns("Elongation"),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"anonymous_zone_is": "in_zone"},
            include=["body_elongation_stats"],
        )
        for m in plan.metrics:
            if m.id.startswith("center_"):
                assert m.parameters_in_use.get("center_zone") == "in_zone"
            else:
                assert "center_zone" not in m.parameters_in_use, (
                    f"{m.id} must not have center_zone but got {m.parameters_in_use}"
                )


# ============================================================
# EPM/FST/TST not affected
# ============================================================

class TestNoFalsePositive:
    """Paradigms without anonymous_zone_override declaration are unaffected."""

    def test_epm_with_named_zones_resolves_normally(self):
        from ethoinsight.catalog import resolve

        cols = [
            "time", "x_center", "y_center",
            "in_zone_open_arms_center_point",
            "in_zone_closed_arms_center_point",
            "distance_moved",
        ]
        plan = resolve(
            paradigm="epm",
            columns=cols,
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
        assert len(plan.metrics) > 0

    def test_fst_resolves_normally(self):
        from ethoinsight.catalog import resolve

        cols = ["time", "trial_time", "x_center", "y_center", "velocity", "mobility_state"]
        plan = resolve(
            paradigm="forced_swim",
            columns=cols,
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
        assert len(plan.metrics) > 0

    def test_tst_resolves_normally(self):
        from ethoinsight.catalog import resolve

        cols = ["time", "trial_time", "x_center", "y_center", "velocity", "mobility_state"]
        plan = resolve(
            paradigm="tail_suspension",
            columns=cols,
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
        assert len(plan.metrics) > 0


# ============================================================
# Loader: list default + anonymous_zone_override parsing
# ============================================================

class TestLoaderAnonymousZoneOverride:
    """Loader correctly parses anonymous_zone_override and accepts list defaults."""

    def test_oft_catalog_has_anonymous_zone_override(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("oft")
        assert cat.anonymous_zone_override is not None
        assert cat.anonymous_zone_override.target_param == "center_zone"
        assert cat.anonymous_zone_override.wrap_list is False

    def test_zero_maze_catalog_has_anonymous_zone_override(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("zero_maze")
        assert cat.anonymous_zone_override is not None
        assert cat.anonymous_zone_override.target_param == "open_zones"
        assert cat.anonymous_zone_override.wrap_list is True

    def test_ldb_catalog_has_anonymous_zone_override(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("ldb")
        assert cat.anonymous_zone_override is not None
        assert cat.anonymous_zone_override.target_param == "light_zone"
        assert cat.anonymous_zone_override.wrap_list is False

    def test_epm_catalog_has_no_anonymous_zone_override(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("epm")
        assert cat.anonymous_zone_override is None

    def test_fst_catalog_has_no_anonymous_zone_override(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("forced_swim")
        assert cat.anonymous_zone_override is None

    def test_zero_maze_open_zones_param_is_list_default(self):
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("zero_maze")
        metric = next(m for m in cat.default_metrics if m.id == "open_zone_time_ratio")
        pspec = metric.parameters["open_zones"]
        assert isinstance(pspec.default, list)
        assert pspec.default == ["in_zone_open"]

    def test_load_all_catalogs_does_not_trigger_duplicate_detection(self):
        """Zone params are per-metric (not paradigm_parameters), avoiding unhashable list error."""
        from ethoinsight.catalog.loader import load_all_catalogs

        common, catalogs = load_all_catalogs()
        assert common is not None
        assert len(catalogs) >= 5  # epm, oft, ldb, fst, tst, zero_maze


# ============================================================
# Error hint (backend side — conditional on deerflow import)
# ============================================================

class TestErrorHintsUpdated:
    """zone_unnamed hint uses unified key and positive language."""

    def test_zone_unnamed_hint_has_unified_key(self):
        pytest.importorskip("deerflow.tools.builtins.prep_metric_plan_tool")
        from deerflow.tools.builtins.prep_metric_plan_tool import _ERROR_HINTS

        hint = _ERROR_HINTS["zone_unnamed"]
        assert "anonymous_zone_is" in hint
        assert "ask_clarification" in hint
        assert "parameter_overrides" in hint
        assert "in_zone" in hint

    def test_zone_unnamed_hint_has_no_negative_language(self):
        """Positive language: no "不要" or "禁止" (deepseek positive prompt rule)."""
        pytest.importorskip("deerflow.tools.builtins.prep_metric_plan_tool")
        from deerflow.tools.builtins.prep_metric_plan_tool import _ERROR_HINTS

        hint = _ERROR_HINTS["zone_unnamed"]
        assert "不要" not in hint
        assert "禁止" not in hint

    def test_zone_unnamed_hint_mentions_evidence(self):
        """Hint should guide lead to inspect_uploaded_file for anonymous_zone_evidence."""
        pytest.importorskip("deerflow.tools.builtins.prep_metric_plan_tool")
        from deerflow.tools.builtins.prep_metric_plan_tool import _ERROR_HINTS

        hint = _ERROR_HINTS["zone_unnamed"]
        assert "inspect_uploaded_file" in hint or "anonymous_zone_evidence" in hint

    def test_zone_unnamed_hint_no_longer_mentions_center_zone_key(self):
        """Old center_zone-specific override key should be gone from hint."""
        pytest.importorskip("deerflow.tools.builtins.prep_metric_plan_tool")
        from deerflow.tools.builtins.prep_metric_plan_tool import _ERROR_HINTS

        hint = _ERROR_HINTS["zone_unnamed"]
        assert "center_zone" not in hint


# ============================================================
# Backward compatibility: old OFT test expectations still pass
# with the new unified key (only the key name changed)
# ============================================================

class TestOFTWithNamedZoneColumnsStillWorks:
    """Named zone columns resolve normally without override."""

    def test_oft_named_zone_resolves(self):
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="oft",
            columns=_named_oft_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
        assert len(plan.metrics) > 0
