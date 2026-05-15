"""Tests for LDB (Light-Dark Box) metric functions."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from ethoinsight.metrics.ldb import (
    compute_light_time_ratio,
    compute_transition_count,
    compute_light_latency,
)


def _make_ldb_df(n_frames=100, *, light_pattern=None, seed=42):
    """Synthetic LDB DataFrame with light/dark zone columns and trial_time."""
    rng = np.random.default_rng(seed)
    if light_pattern is None:
        light_pattern = [1] * 30 + [0] * 30 + [1] * 10 + [0] * 30  # starts in light
    pattern = light_pattern[:n_frames] + [0] * max(0, n_frames - len(light_pattern))
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames) * 0.04,
            "x_center": rng.uniform(-10, 10, n_frames),
            "y_center": rng.uniform(-10, 10, n_frames),
            "in_zone_light": pattern,
            "in_zone_dark": [1 - v for v in pattern],
        }
    )


def _build_parsed_ldb_data(
    subjects: dict[str, pd.DataFrame],
    total_files: int = 1,
    duration_s: float = 600.0,
) -> dict:
    """Minimal parsed_data dict matching parse.parse_batch output for LDB."""
    all_dfs = []
    for name, df in subjects.items():
        df = df.copy()
        df["subject"] = name
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    return {
        "subjects": subjects,
        "all_data": combined,
        "file_list": [f"subj_{i}.txt" for i in range(len(subjects))],
        "summary": {
            "total_files": total_files,
            "total_subjects": len(subjects),
            "total_rows": sum(len(df) for df in subjects.values()),
            "duration_seconds": duration_s,
        },
    }


# ============================================================================
# TestComputeLightTimeRatio
# ============================================================================


class TestComputeLightTimeRatio:
    """Tests for compute_light_time_ratio."""

    def test_basic_ratio(self):
        # pattern: 30 light + 30 dark + 10 light + 30 dark = 40 light / 100 total = 0.40
        df = _make_ldb_df(n_frames=100)
        result = compute_light_time_ratio(df)
        assert result == pytest.approx(0.40)

    def test_all_in_light(self):
        df = _make_ldb_df(n_frames=50, light_pattern=[1] * 50)
        result = compute_light_time_ratio(df)
        assert result == pytest.approx(1.0)

    def test_all_in_dark(self):
        df = _make_ldb_df(n_frames=50, light_pattern=[0] * 50)
        result = compute_light_time_ratio(df)
        assert result == pytest.approx(0.0)

    def test_missing_light_column_returns_none(self):
        df = pd.DataFrame({"trial_time": [0, 1, 2], "in_zone_dark": [1, 0, 1]})
        result = compute_light_time_ratio(df)
        assert result is None

    def test_custom_column_name(self):
        df = pd.DataFrame(
            {
                "trial_time": np.arange(10) * 0.04,
                "my_light_zone": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            }
        )
        result = compute_light_time_ratio(df, light_zone="my_light_zone")
        assert result == pytest.approx(3 / 10)


# ============================================================================
# TestComputeTransitionCount
# ============================================================================


class TestComputeTransitionCount:
    """Tests for compute_transition_count."""

    def test_basic_transition_count(self):
        # pattern: 30 light → 30 dark → 10 light → 30 dark
        # transitions (light perspective): 1→0 at frame 30, 0→1 at frame 60, 1→0 at frame 70
        # = 3 transitions in light column → same count via dark column (mirror)
        # dark: 0→1 at 30, 1→0 at 60, 0→1 at 70 → 3 transitions (counting 0→1 in dark)
        # But we count transitions using light column 0→1 transitions + dark column 0→1 transitions...
        # Actually per spec: count 0→1 in EITHER light or dark = total zone crossings
        # Light 0→1: frame 60 (1 transition)
        # Dark 0→1: frame 30 (1 transition), frame 70 (1 transition)
        # Total = 3 transitions — but not counting first frame
        # Actually let's verify what the default pattern produces:
        # light=[1]*30+[0]*30+[1]*10+[0]*30 (100 frames total)
        # light 0→1: at index 60 (one transition)
        # dark 0→1: at index 30 and index 70 (two transitions)
        # Each crossing of the boundary is one transition in either direction
        # The number of zone crossings = light 0→1 + dark 0→1 = 1 + 2 = 3
        df = _make_ldb_df(n_frames=100)
        result = compute_transition_count(df)
        assert result == 3

    def test_no_transitions(self):
        # Always in light — no transitions
        df = _make_ldb_df(n_frames=50, light_pattern=[1] * 50)
        result = compute_transition_count(df)
        assert result == 0

    def test_alternating_produces_correct_transitions(self):
        # [1, 0, 1, 0, 1, 0] — 3 dark entries + 2 light entries = 5 transitions
        # light 0→1: indices 2, 4 → 2 transitions
        # dark 0→1: indices 1, 3, 5 → 3 transitions
        # total = 5
        pattern = [1, 0, 1, 0, 1, 0]
        df = _make_ldb_df(n_frames=6, light_pattern=pattern)
        result = compute_transition_count(df)
        assert result == 5

    def test_missing_columns_returns_none(self):
        df = pd.DataFrame({"trial_time": [0, 1, 2], "x_center": [1, 2, 3]})
        result = compute_transition_count(df)
        assert result is None

    def test_single_transition_light_to_dark(self):
        # light then dark: [1, 1, 1, 0, 0, 0]
        # light 0→1: none (starts in 1, never returns)
        # dark 0→1: at index 3 → 1 transition
        pattern = [1, 1, 1, 0, 0, 0]
        df = _make_ldb_df(n_frames=6, light_pattern=pattern)
        result = compute_transition_count(df)
        assert result == 1


# ============================================================================
# TestComputeLightLatency
# ============================================================================


class TestComputeLightLatency:
    """Tests for compute_light_latency."""

    def test_latency_when_starts_in_dark(self):
        # animal is in dark for 20 frames then enters light
        # trial_time = frame * 0.04, so first light frame = 20 → t = 0.80s
        pattern = [0] * 20 + [1] * 80
        df = _make_ldb_df(n_frames=100, light_pattern=pattern)
        result = compute_light_latency(df)
        assert result == pytest.approx(20 * 0.04)

    def test_latency_when_starts_in_light(self):
        # animal starts in light zone — latency should be 0.0s (frame 0)
        df = _make_ldb_df(n_frames=100)  # pattern starts with [1]*30
        result = compute_light_latency(df)
        assert result == pytest.approx(0.0)

    def test_never_enters_light_returns_none(self):
        df = _make_ldb_df(n_frames=50, light_pattern=[0] * 50)
        result = compute_light_latency(df)
        assert result is None

    def test_no_light_column_returns_none(self):
        df = pd.DataFrame({"trial_time": [0, 1, 2], "in_zone_dark": [1, 0, 1]})
        result = compute_light_latency(df)
        assert result is None

    def test_latency_without_trial_time_returns_frame_index(self):
        # Without trial_time, returns frame index of first light entry
        n = 30
        pattern = [0] * 10 + [1] * 20
        df = pd.DataFrame(
            {
                "x_center": np.zeros(n),
                "in_zone_light": pattern,
                "in_zone_dark": [1 - v for v in pattern],
            }
        )
        result = compute_light_latency(df)
        assert result == 10  # frame index, not time


# ============================================================================
# TestComputeParadigmMetricsLdb
# ============================================================================


class TestComputeParadigmMetricsLdb:
    """Tests for compute_paradigm_metrics with paradigm='light_dark_box'."""

    def test_includes_all_ldb_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_ldb_df(n_frames=200)
        parsed = _build_parsed_ldb_data({"Subject_1": df})
        result = compute_paradigm_metrics(parsed, "light_dark_box")

        subj = result["per_subject"]["Subject_1"]
        expected_metrics = [
            "distance_moved",
            "velocity_stats",
            "light_time_ratio",
            "transition_count",
            "light_latency",
        ]
        for m in expected_metrics:
            assert m in subj, f"Missing metric: {m}"

    def test_light_time_ratio_value_correct(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        # pattern: 40 light frames in 100 total = 0.40
        df = _make_ldb_df(n_frames=100)
        parsed = _build_parsed_ldb_data({"S1": df})
        result = compute_paradigm_metrics(parsed, "light_dark_box")
        assert result["per_subject"]["S1"]["light_time_ratio"] == pytest.approx(0.40)

    def test_data_quality_warning_low_n(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_ldb_df(n_frames=100)
        parsed = _build_parsed_ldb_data({"Subject_1": df})
        groups = {"treatment": ["Subject_1"]}
        result = compute_paradigm_metrics(parsed, "light_dark_box", groups=groups)

        warnings = result.get("data_quality_warnings", [])
        # n=1 < 5 → warning
        n_warnings = [w for w in warnings if "统计功效不足" in w.get("message", "")]
        assert len(n_warnings) >= 1

    def test_data_quality_warning_low_transitions(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        # animal stays in light the whole time → transitions = 0 < 4
        df = _make_ldb_df(n_frames=200, light_pattern=[1] * 200)
        parsed = _build_parsed_ldb_data({"Subject_1": df})
        result = compute_paradigm_metrics(parsed, "light_dark_box")

        warnings = result.get("data_quality_warnings", [])
        transition_warnings = [
            w for w in warnings if w.get("metric") == "transition_count"
        ]
        assert len(transition_warnings) >= 1

    def test_group_summary_has_ldb_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df1 = _make_ldb_df(n_frames=200, seed=1)
        df2 = _make_ldb_df(n_frames=200, seed=2)
        parsed = _build_parsed_ldb_data({"S1": df1, "S2": df2})
        groups = {"control": ["S1", "S2"]}
        result = compute_paradigm_metrics(parsed, "light_dark_box", groups=groups)

        assert "control" in result["group_summary"]
        ctrl = result["group_summary"]["control"]
        for m in ["light_time_ratio", "light_latency"]:
            assert m in ctrl, f"Missing group metric: {m}"
