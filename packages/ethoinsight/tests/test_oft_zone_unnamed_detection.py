"""Tests for PR-1: anonymous zone detection + center_zone override + failure classification.

Covers:
- zone_unnamed detection (OFT/LDB/zero_maze with bare in_zone but no named zone columns)
- No false positive for EPM (has named zone columns)
- center_zone override allows resolution
- True missing columns (non-zone) still raises columns_missing
- compute_center_time/compute_center_distance accept center_zone parameter
"""

from __future__ import annotations

import pytest


# ============================================================
# Helper: column fixtures for different paradigms
# ============================================================

def _oft_anonymous_zone_columns() -> list[str]:
    """OFT data with bare in_zone but NO in_zone_center_* columns."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "in_zone",
        "distance_moved",
    ]


def _oft_named_zone_columns() -> list[str]:
    """OFT data with proper in_zone_center_* columns."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "in_zone_center_center_point",
        "distance_moved",
    ]


def _ldb_anonymous_zone_columns() -> list[str]:
    """LDB data with bare in_zone but NO in_zone_light* columns."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
        "in_zone",
        "distance_moved",
    ]


def _oft_no_in_zone_at_all() -> list[str]:
    """OFT data without any in_zone column — true missing."""
    return [
        "time",
        "trial_time",
        "x_center",
        "y_center",
    ]


# ============================================================
# zone_unnamed detection
# ============================================================

class TestAnonymousZoneDetection:
    """Detect bare in_zone when named zone columns are missing."""

    def test_oft_anonymous_zone_raises_zone_unnamed(self):
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=_oft_anonymous_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "zone_unnamed"
        assert "in_zone" in str(exc.value)
        # Details should include found_column and missing_patterns
        assert exc.value.details["found_column"] == "in_zone"
        assert any("in_zone_center" in p for p in exc.value.details["missing_patterns"])

    def test_ldb_anonymous_zone_raises_zone_unnamed(self):
        """Paradigm-agnostic: LDB with bare in_zone also triggers zone_unnamed."""
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="ldb",
                columns=_ldb_anonymous_zone_columns(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "zone_unnamed"

    def test_oft_with_named_zone_resolves_normally(self):
        """When in_zone_center_* exists, resolve should succeed (no zone_unnamed)."""
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="oft",
            columns=_oft_named_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
        assert len(plan.metrics) > 0

    def test_oft_anonymous_zone_with_center_zone_override_resolves(self):
        """When user has confirmed center_zone=in_zone via override, resolve succeeds."""
        from ethoinsight.catalog import resolve

        plan = resolve(
            paradigm="oft",
            columns=_oft_anonymous_zone_columns(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"center_zone": "in_zone"},
        )
        metric_ids = {m.id for m in plan.metrics}
        # All 5 center metrics should be present
        assert "center_time_ratio" in metric_ids
        assert "center_distance_ratio" in metric_ids
        assert "center_entry_count" in metric_ids
        assert "center_time" in metric_ids
        assert "center_distance" in metric_ids
        # center_zone should appear in parameters_in_use of center metrics only
        for m in plan.metrics:
            if m.id.startswith("center_"):
                assert m.parameters_in_use.get("center_zone") == "in_zone", (
                    f"{m.id} should have center_zone=in_zone"
                )

    def test_override_with_optional_metric_does_not_inject_center_zone(self):
        """center_zone override must NOT leak into optional metrics' parameters_in_use."""
        from ethoinsight.catalog import resolve

        # Include an optional metric that does NOT accept center_zone
        plan = resolve(
            paradigm="oft",
            columns=_oft_anonymous_zone_columns() + ["Elongation"],
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            overrides={"center_zone": "in_zone"},
            include=["body_elongation_stats"],
        )
        for m in plan.metrics:
            if m.id.startswith("center_"):
                assert m.parameters_in_use.get("center_zone") == "in_zone"
            else:
                # Optional metrics must NOT have center_zone injected
                assert "center_zone" not in m.parameters_in_use, (
                    f"{m.id} must not have center_zone but got {m.parameters_in_use}"
                )


# ============================================================
# True missing columns (non-zone)
# ============================================================

class TestTrueMissingColumns:
    """Non-zone missing columns still raise columns_missing, not zone_unnamed."""

    def test_missing_distance_moved_raises_columns_missing(self):
        """Missing distance_moved (not a zone pattern) → columns_missing."""
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
        assert "distance_moved" in str(exc.value)

    def test_missing_x_center_no_zone_raises_columns_missing(self):
        """No in_zone at all, missing x_center → columns_missing (not zone_unnamed)."""
        from ethoinsight.catalog import resolve, ResolveError

        cols = ["time", "trial_time", "y_center"]
        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=cols,
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "columns_missing"

    def test_oft_no_zone_columns_at_all_raises_columns_missing(self):
        """No in_zone column at all → columns_missing (bare in_zone doesn't exist)."""
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=_oft_no_in_zone_at_all(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        assert exc.value.code == "columns_missing"

    def test_columns_missing_message_is_user_friendly(self):
        """columns_missing error message should be in Chinese and actionable."""
        from ethoinsight.catalog import resolve, ResolveError

        with pytest.raises(ResolveError) as exc:
            resolve(
                paradigm="oft",
                columns=_oft_no_in_zone_at_all(),
                raw_files=["/tmp/raw.txt"],
                workspace_dir="/tmp/workspace",
            )
        msg = str(exc.value)
        assert "数据缺少" in msg
        assert "实验" in msg


# ============================================================
# EPM no false positive
# ============================================================

class TestEPMNoFalsePositive:
    """EPM has named zone columns (in_zone_open_arms_*, in_zone_closed_arms_*),
    so it should never trigger zone_unnamed."""

    def test_epm_resolves_normally(self):
        from ethoinsight.catalog import resolve

        cols = [
            "time",
            "x_center",
            "y_center",
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


# ============================================================
# compute_center_time / compute_center_distance accept center_zone
# ============================================================

class TestCenterZoneParameterPassThrough:
    """compute_center_time and compute_center_distance accept center_zone kwarg."""

    def test_compute_center_time_accepts_center_zone(self):
        import pandas as pd
        from ethoinsight.metrics.oft import compute_center_time

        df = pd.DataFrame({
            "in_zone": [1, 1, 0, 0, 1],
            "trial_time": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        # Should not raise TypeError
        result = compute_center_time(df, center_zone="in_zone")
        assert result is not None
        assert result > 0

    def test_compute_center_distance_accepts_center_zone(self):
        import pandas as pd
        from ethoinsight.metrics.oft import compute_center_distance

        df = pd.DataFrame({
            "in_zone": [1, 1, 0, 0, 1],
            "distance_moved": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # Should not raise TypeError
        result = compute_center_distance(df, center_zone="in_zone")
        assert result is not None
        assert result > 0

    def test_compute_center_time_raises_typeerror_with_unexpected_kwarg(self):
        """Before the fix, compute_center_time would raise TypeError with center_zone.
        This test confirms the fix prevents that."""
        import pandas as pd
        from ethoinsight.metrics.oft import compute_center_time

        df = pd.DataFrame({
            "in_zone": [1, 0, 1],
            "trial_time": [0.0, 1.0, 2.0],
        })
        # center_zone="in_zone" should work (not TypeError)
        result = compute_center_time(df, center_zone="in_zone")
        assert result is not None


# ============================================================
# _detect_anonymous_zone helper
# ============================================================

class TestDetectAnonymousZoneHelper:
    """Unit tests for _detect_anonymous_zone function."""

    def test_zone_pattern_with_bare_in_zone_returns_error(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone

        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={},
        )
        assert result is not None
        assert result.code == "zone_unnamed"

    def test_zone_pattern_without_bare_in_zone_returns_none(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone

        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "x_center"],
            overrides={},
        )
        assert result is None

    def test_non_zone_pattern_returns_none(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone

        result = _detect_anonymous_zone(
            missing_patterns=["distance_moved"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={},
        )
        assert result is None

    def test_zone_pattern_with_center_zone_override_returns_true(self):
        from ethoinsight.catalog.resolve import _detect_anonymous_zone

        result = _detect_anonymous_zone(
            missing_patterns=["in_zone_center_*"],
            available_columns=["time", "in_zone", "x_center"],
            overrides={"center_zone": "in_zone"},
        )
        # Override resolves the zone → returns True (proceed)
        assert result is True


# ============================================================
# Error hint lookup (backend-side, only tested when deerflow is available)
# ============================================================

class TestErrorHints:
    """Verify zone_unnamed is in _ERROR_HINTS (requires deerflow harness)."""

    def test_zone_unnamed_in_error_hints(self):
        pytest.importorskip("deerflow.tools.builtins.prep_metric_plan_tool")
        from deerflow.tools.builtins.prep_metric_plan_tool import _ERROR_HINTS

        assert "zone_unnamed" in _ERROR_HINTS
        hint = _ERROR_HINTS["zone_unnamed"]
        assert "in_zone" in hint
        assert "ask_clarification" in hint
        assert "parameter_overrides" in hint
