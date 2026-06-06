"""Tests for Zero Maze metric functions.

Covers compute_open_zone_time_ratio / compute_open_zone_time /
compute_open_zone_distance / compute_hesitation_count and
compute_paradigm_metrics(paradigm="zero_maze") dispatch + data quality warnings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Helpers
# ============================================================================


def _make_zm_df(
    n_frames: int = 100,
    *,
    open_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic Zero Maze DataFrame with open/closed zone columns.

    By default the first 25 frames are in open zone, the remaining 75 in closed.
    """
    rng = np.random.default_rng(seed)
    if open_pattern is None:
        open_pattern = [1] * 25 + [0] * 75

    # Pad or trim to n_frames
    pat = (open_pattern + [0] * max(0, n_frames - len(open_pattern)))[:n_frames]

    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames) * 0.04,
            "x_center": rng.uniform(-10, 10, n_frames),
            "y_center": rng.uniform(-10, 10, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "in_zone_open_1": [v for v in pat],
            "in_zone_closed_1": [1 - v for v in pat],
        }
    )


def _build_parsed_zm_data(
    subjects: dict[str, pd.DataFrame],
    total_files: int = 1,
    duration_s: float = 600.0,
) -> dict:
    """Minimal parsed_data dict matching parse.parse_batch output for Zero Maze."""
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
# compute_open_zone_time_ratio tests
# ============================================================================


class TestComputeOpenZoneTimeRatio:
    """Tests for compute_open_zone_time_ratio."""

    def test_25_percent_open_returns_0_25(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = _make_zm_df(n_frames=100)  # 25 open / 75 closed
        result = compute_open_zone_time_ratio(df)
        assert result == pytest.approx(0.25)

    def test_all_open_returns_1(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = _make_zm_df(n_frames=50, open_pattern=[1] * 50)
        result = compute_open_zone_time_ratio(df)
        assert result == pytest.approx(1.0)

    def test_none_open_returns_0(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = _make_zm_df(n_frames=50, open_pattern=[0] * 50)
        result = compute_open_zone_time_ratio(df)
        assert result == pytest.approx(0.0)

    def test_no_open_columns_returns_none(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = pd.DataFrame({"trial_time": [0, 1, 2], "x_center": [1, 2, 3]})
        result = compute_open_zone_time_ratio(df)
        assert result is None

    def test_explicit_zones_override_detection(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        df = _make_zm_df(n_frames=100)
        # Override to use only in_zone_open_1
        result = compute_open_zone_time_ratio(df, open_zones=["in_zone_open_1"])
        assert result == pytest.approx(0.25)

    def test_multiple_open_columns_merged_with_or(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time_ratio

        n = 100
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "in_zone_open_1": [1] * 25 + [0] * 75,
                "in_zone_open_2": [0] * 75 + [1] * 25,
            }
        )
        result = compute_open_zone_time_ratio(df)
        # 50 of 100 frames are open (non-overlapping)
        assert result == pytest.approx(0.5)


# ============================================================================
# compute_open_zone_time tests
# ============================================================================


class TestComputeOpenZoneTime:
    """Tests for compute_open_zone_time."""

    def test_time_in_seconds(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time

        df = _make_zm_df(n_frames=100)  # 25 open frames × 0.04s
        result = compute_open_zone_time(df)
        assert result == pytest.approx(25 * 0.04)

    def test_no_trial_time_falls_back_to_frame_count(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time

        n = 60
        df = pd.DataFrame(
            {
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "in_zone_open_1": [1] * 20 + [0] * 40,
            }
        )
        result = compute_open_zone_time(df)
        assert result == 20  # falls back to frame count

    def test_no_open_columns_returns_none(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time

        df = pd.DataFrame({"trial_time": [0, 1, 2]})
        result = compute_open_zone_time(df)
        assert result is None

    def test_zero_open_frames_returns_zero(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_time

        df = _make_zm_df(n_frames=50, open_pattern=[0] * 50)
        result = compute_open_zone_time(df)
        assert result == pytest.approx(0.0)


# ============================================================================
# compute_open_zone_distance tests
# ============================================================================


class TestComputeOpenZoneDistance:
    """Tests for compute_open_zone_distance."""

    def test_distance_matches_open_frames(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_distance

        rng = np.random.default_rng(0)
        n = 100
        dist = rng.uniform(1, 3, n)
        open_flags = np.array([1] * 25 + [0] * 75)
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "distance_moved": dist,
                "in_zone_open_1": open_flags,
                "in_zone_closed_1": 1 - open_flags,
            }
        )
        result = compute_open_zone_distance(df)
        expected = dist[:25].sum()
        assert result == pytest.approx(expected)

    def test_no_distance_moved_returns_none(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_distance

        df = _make_zm_df(n_frames=50)
        df = df.drop(columns=["distance_moved"])
        result = compute_open_zone_distance(df)
        assert result is None

    def test_no_open_columns_returns_none(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_distance

        df = pd.DataFrame(
            {
                "trial_time": [0, 1, 2],
                "distance_moved": [1.0, 2.0, 3.0],
            }
        )
        result = compute_open_zone_distance(df)
        assert result is None

    def test_zero_total_distance_returns_zero(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_distance

        n = 50
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "distance_moved": np.zeros(n),
                "in_zone_open_1": [1] * 25 + [0] * 25,
            }
        )
        result = compute_open_zone_distance(df)
        assert result == 0.0

    def test_all_in_open_zone_equals_total_distance(self):
        from ethoinsight.metrics.zero_maze import compute_open_zone_distance

        n = 50
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "distance_moved": np.ones(n),
                "in_zone_open_1": np.ones(n, dtype=int),
            }
        )
        result = compute_open_zone_distance(df)
        assert result == pytest.approx(50.0)


# ============================================================================
# compute_hesitation_count tests
# ============================================================================


class TestComputeHesitationCount:
    """Tests for compute_hesitation_count."""

    def test_zero_hesitations_long_open_stay(self):
        """Animal stays in open long enough → not a hesitation."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        # closed(20) → open(20 frames, >= min_gap=5) → closed(rest)
        n = 80
        open_pat = [0] * 20 + [1] * 20 + [0] * 40
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "in_zone_open_1": open_pat,
                "in_zone_closed_1": [1 - v for v in open_pat],
            }
        )
        result = compute_hesitation_count(df)
        assert result == 0

    def test_one_hesitation_brief_open(self):
        """Single brief (< min_gap=5 frames) open excursion from closed."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        # closed(20) → open(3 frames < 5) → closed(rest) → 1 hesitation
        n = 80
        open_pat = [0] * 20 + [1] * 3 + [0] * 57
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "in_zone_open_1": open_pat,
                "in_zone_closed_1": [1 - v for v in open_pat],
            }
        )
        result = compute_hesitation_count(df)
        assert result == 1

    def test_multiple_hesitations(self):
        """Three brief open excursions → 3 hesitations."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        n = 120
        open_pat = (
            [0] * 10 + [1] * 2 + [0] * 10 + [1] * 2 + [0] * 10 + [1] * 2 + [0] * 84
        )
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "in_zone_open_1": open_pat,
                "in_zone_closed_1": [1 - v for v in open_pat],
            }
        )
        result = compute_hesitation_count(df)
        assert result == 3

    def test_no_open_columns_returns_none(self):
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        df = pd.DataFrame({"trial_time": [0, 1, 2]})
        result = compute_hesitation_count(df)
        assert result is None

    def test_always_closed_returns_zero(self):
        """Never enters open zone → no hesitations."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        df = _make_zm_df(n_frames=50, open_pattern=[0] * 50)
        result = compute_hesitation_count(df)
        assert result == 0

    def test_custom_min_gap(self):
        """Hesitation threshold adjustable via min_gap_frames."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        # With min_gap=10: open(7 frames) → hesitation
        # With min_gap=5:  open(7 frames) → NOT hesitation (7 >= 5)
        n = 80
        open_pat = [0] * 20 + [1] * 7 + [0] * 53
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "in_zone_open_1": open_pat,
                "in_zone_closed_1": [1 - v for v in open_pat],
            }
        )
        result_default = compute_hesitation_count(df, min_gap_frames=5)
        result_high = compute_hesitation_count(df, min_gap_frames=10)
        assert result_default == 0  # 7 >= 5, not a hesitation
        assert result_high == 1  # 7 < 10, counts as hesitation

    def test_explicit_zones_parameter(self):
        """Using explicit zone parameters."""
        from ethoinsight.metrics.zero_maze import compute_hesitation_count

        n = 60
        open_pat = [0] * 20 + [1] * 2 + [0] * 38
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "in_zone_open_1": open_pat,
                "in_zone_closed_1": [1 - v for v in open_pat],
            }
        )
        result = compute_hesitation_count(
            df,
            open_zones=["in_zone_open_1"],
            closed_zones=["in_zone_closed_1"],
        )
        assert result == 1


# ============================================================================
# compute_paradigm_metrics with paradigm="zero_maze"
# ============================================================================


class TestComputeParadigmMetricsZeroMaze:
    """Tests for compute_paradigm_metrics with paradigm='zero_maze'."""

    def test_includes_all_zm_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_zm_df(n_frames=200)
        parsed = _build_parsed_zm_data({"Subject_1": df})
        result = compute_paradigm_metrics(parsed, "zero_maze")

        subj = result["per_subject"]["Subject_1"]
        expected_metrics = [
            "distance_moved",
            "open_zone_time_ratio",
            "open_zone_time",
            "open_zone_distance",
            "hesitation_count",
        ]
        for m in expected_metrics:
            assert m in subj, f"Missing metric: {m}"

    def test_open_zone_time_ratio_value(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_zm_df(n_frames=100)  # 25% open
        parsed = _build_parsed_zm_data({"S1": df})
        result = compute_paradigm_metrics(parsed, "zero_maze")
        ratio = result["per_subject"]["S1"]["open_zone_time_ratio"]
        assert ratio == pytest.approx(0.25)

    def test_group_summary_has_zm_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df1 = _make_zm_df(n_frames=200, seed=1)
        df2 = _make_zm_df(n_frames=200, seed=2)
        parsed = _build_parsed_zm_data({"Ctrl_1": df1, "Ctrl_2": df2})
        groups = {"control": ["Ctrl_1", "Ctrl_2"]}
        result = compute_paradigm_metrics(parsed, "zero_maze", groups=groups)

        assert "control" in result["group_summary"]
        ctrl = result["group_summary"]["control"]
        for m in ["open_zone_time_ratio", "open_zone_time", "distance_moved"]:
            assert m in ctrl, f"Missing group metric: {m}"

    def test_data_quality_warning_low_n(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_zm_df(n_frames=200)
        parsed = _build_parsed_zm_data({"Subject_1": df})
        groups = {"treatment": ["Subject_1"]}
        result = compute_paradigm_metrics(parsed, "zero_maze", groups=groups)

        warnings = result.get("data_quality_warnings", [])
        critical = [w for w in warnings if w.get("severity") == "critical"]
        assert len(critical) >= 1

    def test_zm_warning_low_distance(self):
        """Total distance too low → warning about motor suppression confound."""
        from ethoinsight.metrics import compute_paradigm_metrics

        n = 100
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n) * 0.04,
                "x_center": np.zeros(n),
                "y_center": np.zeros(n),
                "distance_moved": np.full(n, 0.1),  # very low movement
                "in_zone_open_1": [1] * 25 + [0] * 75,
                "in_zone_closed_1": [0] * 25 + [1] * 75,
            }
        )
        parsed = _build_parsed_zm_data({"S1": df})
        result = compute_paradigm_metrics(parsed, "zero_maze")

        warnings = result.get("data_quality_warnings", [])
        dist_warnings = [w for w in warnings if "distance" in w.get("metric", "")]
        assert len(dist_warnings) >= 1

    def test_zm_no_distance_warning_with_good_data(self):
        """Good movement data → no distance-related warning."""
        from ethoinsight.metrics import compute_paradigm_metrics

        dfs = {f"S{i}": _make_zm_df(n_frames=300, seed=i) for i in range(1, 6)}
        parsed = _build_parsed_zm_data(dfs)
        result = compute_paradigm_metrics(parsed, "zero_maze")

        warnings = result.get("data_quality_warnings", [])
        dist_warnings = [w for w in warnings if "distance_moved" in w.get("metric", "")]
        assert len(dist_warnings) == 0
