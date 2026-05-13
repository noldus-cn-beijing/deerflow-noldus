"""Tests for EPM (Elevated Plus Maze) metric functions.

Covers compute_open_arm_entry_count / _entry_ratio / _time / total_entry_count
and compute_paradigm_metrics(paradigm="epm") dispatch + data quality warnings.

(Note: 2026-05-11 拆分自原 test_template_epm.py。templates/epm.py 脚本模板
路径已被 SOTA 架构决策否决（见
docs/handoffs/2026-05-11-paradigm-sota-architecture-grill-handoff.md），
原 TestEpmTemplateModule / TestEpmTemplateSoftGate 测试块已删。)
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Helpers
# ============================================================================


def _make_epm_df(
    n_frames: int = 100,
    *,
    open_arm_cols: list[str] | None = None,
    closed_arm_cols: list[str] | None = None,
    center_cols: list[str] | None = None,
    open_arm_pattern: list[int] | None = None,
    closed_arm_pattern: list[int] | None = None,
    center_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking EPM trajectory data.

    Zone columns are 0/1. By default creates alternating open/closed arm
    occupancy to produce realistic entry counts.
    """
    rng = np.random.default_rng(seed)
    if open_arm_cols is None:
        open_arm_cols = ["in_zone_open_arm_1"]
    if closed_arm_cols is None:
        closed_arm_cols = ["in_zone_closed_arm_1"]
    if center_cols is None:
        center_cols = ["in_zone_center-point"]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    # Default: 5 open-arm entries (0→1 transitions)
    if open_arm_pattern is None:
        open_vals = np.zeros(n_frames, dtype=int)
        entries = [10, 25, 45, 65, 85]  # start frames for open-arm bouts
        for start in entries:
            open_vals[start : start + 8] = 1
        open_arm_pattern = open_vals

    for col in open_arm_cols:
        df[col] = open_arm_pattern

    # Default: 5 closed-arm entries (non-overlapping with open arm)
    if closed_arm_pattern is None:
        closed_vals = np.zeros(n_frames, dtype=int)
        entries = [18, 35, 55, 75, 93]
        for start in entries:
            closed_vals[start : start + 6] = 1
        closed_arm_pattern = closed_vals

    for col in closed_arm_cols:
        df[col] = closed_arm_pattern

    # Center is always 1 when neither open nor closed
    if center_pattern is None:
        center_vals = np.ones(n_frames, dtype=int)
        for i in range(n_frames):
            if open_arm_pattern[i] == 1 or closed_arm_pattern[i] == 1:
                center_vals[i] = 0
        center_pattern = center_vals

    for col in center_cols:
        df[col] = center_pattern

    return df


def _build_parsed_epm_data(
    subjects: dict[str, pd.DataFrame],
    total_files: int = 1,
    duration_s: float = 600.0,
) -> dict:
    """Minimal parsed_data dict matching parse.parse_batch output for EPM."""
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
# EPM metric function tests
# ============================================================================


class TestComputeOpenArmEntryCount:
    """Tests for compute_open_arm_entry_count."""

    def test_counts_five_entries(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        df = _make_epm_df(n_frames=100)
        result = compute_open_arm_entry_count(df)
        assert result == 5

    def test_no_open_arm_columns_returns_none(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        df = pd.DataFrame({"trial_time": [0, 1, 2], "x_center": [1, 2, 3]})
        result = compute_open_arm_entry_count(df)
        assert result is None

    def test_empty_dataframe_returns_zero(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        df = _make_epm_df(n_frames=0, open_arm_pattern=np.array([], dtype=int))
        result = compute_open_arm_entry_count(df)
        assert result == 0

    def test_single_frame_returns_one(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        df = _make_epm_df(n_frames=1, open_arm_pattern=np.array([1], dtype=int))
        result = compute_open_arm_entry_count(df)
        assert result == 1  # first frame counts as entry

    def test_always_in_open_arm_returns_one_entry(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        vals = np.ones(50, dtype=int)
        df = _make_epm_df(
            n_frames=50,
            open_arm_pattern=vals,
            closed_arm_pattern=np.zeros(50, dtype=int),
            center_pattern=np.zeros(50, dtype=int),
        )
        result = compute_open_arm_entry_count(df)
        assert result == 1  # one entry at frame 0

    def test_multiple_open_arm_columns_merged(self):
        from ethoinsight.metrics import compute_open_arm_entry_count

        # Two open arm columns, each with different entry timings
        n = 100
        oa1 = np.zeros(n, dtype=int)
        oa2 = np.zeros(n, dtype=int)
        oa1[10:20] = 1
        oa2[30:40] = 1
        oa1[50:55] = 1
        oa2[70:80] = 1

        df = _make_epm_df(
            n_frames=n,
            open_arm_cols=["in_zone_open_arm_1", "in_zone_open_arm_2"],
            open_arm_pattern=None,  # will be overridden below
        )
        # Override the defaults set by _make_epm_df
        df["in_zone_open_arm_1"] = oa1
        df["in_zone_open_arm_2"] = oa2
        df["in_zone_closed_arm_1"] = np.zeros(n, dtype=int)
        df["in_zone_center-point"] = np.ones(n, dtype=int)

        result = compute_open_arm_entry_count(df)
        # Combined: entry at 10, 30, 50, 70 = 4 transitions
        assert result == 4


class TestComputeOpenArmEntryRatio:
    """Tests for compute_open_arm_entry_ratio."""

    def test_ratio_five_of_nine(self):
        from ethoinsight.metrics import compute_open_arm_entry_count
        from ethoinsight.metrics import compute_open_arm_entry_ratio
        from ethoinsight.metrics import compute_total_entry_count

        # 5 open entries out of 9 total → ratio ≈ 0.556
        n = 150
        oa = np.zeros(n, dtype=int)
        ca = np.zeros(n, dtype=int)
        # Open: 5 entries
        for start in [5, 25, 45, 65, 85]:
            oa[start : start + 6] = 1
        # Closed: 4 entries
        for start in [12, 32, 52, 72]:
            ca[start : start + 6] = 1

        df = _make_epm_df(
            n_frames=n,
            open_arm_pattern=oa,
            closed_arm_pattern=ca,
            center_pattern=np.ones(n, dtype=int),
        )
        open_count = compute_open_arm_entry_count(df)
        total_count = compute_total_entry_count(df)
        ratio = compute_open_arm_entry_ratio(df)

        assert open_count == 5
        assert total_count == 9
        assert ratio == pytest.approx(5 / 9)

    def test_no_entries_returns_none(self):
        from ethoinsight.metrics import compute_open_arm_entry_ratio

        df = _make_epm_df(
            n_frames=100,
            open_arm_pattern=np.zeros(100, dtype=int),
            closed_arm_pattern=np.zeros(100, dtype=int),
            center_pattern=np.ones(100, dtype=int),
        )
        result = compute_open_arm_entry_ratio(df)
        assert result is None  # 0/0 is undefined

    def test_all_open_entries_returns_one(self):
        from ethoinsight.metrics import compute_open_arm_entry_ratio

        n = 50
        oa = np.zeros(n, dtype=int)
        oa[10:20] = 1
        oa[30:40] = 1
        df = _make_epm_df(
            n_frames=n,
            open_arm_pattern=oa,
            closed_arm_pattern=np.zeros(n, dtype=int),
            center_pattern=np.ones(n, dtype=int),
        )
        result = compute_open_arm_entry_ratio(df)
        assert result == pytest.approx(1.0)


class TestComputeOpenArmTime:
    """Tests for compute_open_arm_time."""

    def test_time_in_seconds(self):
        from ethoinsight.metrics import compute_open_arm_time

        df = _make_epm_df(n_frames=100)
        result = compute_open_arm_time(df)
        # 5 bouts × 8 frames × 0.04s/frame = 1.6s
        assert result == pytest.approx(5 * 8 * 0.04)

    def test_no_trial_time_returns_frames(self):
        from ethoinsight.metrics import compute_open_arm_time

        n = 100
        oa = np.zeros(n, dtype=int)
        oa[10:30] = 1
        df = pd.DataFrame(
            {
                "x_center": np.arange(n, dtype=float),
                "y_center": np.arange(n, dtype=float),
                "in_zone_open_arm_1": oa,
            }
        )
        result = compute_open_arm_time(df)
        assert result == 20  # falls back to frame count

    def test_no_open_arm_columns_returns_none(self):
        from ethoinsight.metrics import compute_open_arm_time

        df = pd.DataFrame({"trial_time": [0, 1, 2]})
        result = compute_open_arm_time(df)
        assert result is None


class TestComputeTotalEntryCount:
    """Tests for compute_total_entry_count."""

    def test_counts_all_arm_entries(self):
        from ethoinsight.metrics import compute_total_entry_count

        df = _make_epm_df(n_frames=100)  # default: 5 open + 5 closed = 10 entries
        result = compute_total_entry_count(df)
        assert result == 10

    def test_no_zone_columns_returns_none(self):
        from ethoinsight.metrics import compute_total_entry_count

        df = pd.DataFrame({"trial_time": [0, 1, 2], "x_center": [1, 2, 3]})
        result = compute_total_entry_count(df)
        assert result is None

    def test_only_center_zone_excluded(self):
        from ethoinsight.metrics import compute_total_entry_count

        n = 50
        center = np.ones(n, dtype=int)
        # Center-only should not count as arm entries → returns None (no arm columns)
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "in_zone_center-point": center,
            }
        )
        result = compute_total_entry_count(df)
        assert result is None


# ============================================================================
# EPM paradigm metrics dispatcher tests
# ============================================================================


class TestComputeParadigmMetricsEpm:
    """Tests for compute_paradigm_metrics with paradigm='epm'."""

    def test_includes_all_epm_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_epm_df(n_frames=200)
        parsed = _build_parsed_epm_data({"Subject_1": df})
        result = compute_paradigm_metrics(parsed, "epm")

        subj = result["per_subject"]["Subject_1"]
        expected_metrics = [
            "distance_moved",
            "velocity_stats",
            "open_arm_time_ratio",
            "open_arm_entry_ratio",
            "open_arm_entry_count",
            "open_arm_time",
            "total_entry_count",
        ]
        for m in expected_metrics:
            assert m in subj, f"Missing metric: {m}"
            assert subj[m] is not None, f"Metric {m} should not be None"

    def test_group_summary_has_epm_metrics(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df1 = _make_epm_df(n_frames=200, seed=1)
        df2 = _make_epm_df(n_frames=200, seed=2)
        parsed = _build_parsed_epm_data({"Subj_Ctrl_1": df1, "Subj_Ctrl_2": df2})

        groups = {"control": ["Subj_Ctrl_1", "Subj_Ctrl_2"]}
        result = compute_paradigm_metrics(parsed, "epm", groups=groups)

        assert "control" in result["group_summary"]
        ctrl = result["group_summary"]["control"]
        for m in [
            "open_arm_time_ratio",
            "open_arm_entry_ratio",
            "open_arm_entry_count",
            "total_entry_count",
        ]:
            assert m in ctrl, f"Missing group metric: {m}"

    def test_data_quality_warning_low_n(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_epm_df(n_frames=200)
        parsed = _build_parsed_epm_data({"Subject_1": df})
        groups = {"treatment": ["Subject_1"]}
        result = compute_paradigm_metrics(parsed, "epm", groups=groups)

        warnings = result.get("data_quality_warnings", [])
        # n=1 < 3 → critical warning
        critical = [w for w in warnings if w.get("severity") == "critical"]
        assert len(critical) >= 1

    def test_epm_specific_warning_low_total_entries(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        # Only 1 entry total (very low)
        n = 100
        oa = np.zeros(n, dtype=int)
        oa[10:20] = 1  # single open arm bout
        df = _make_epm_df(
            n_frames=n,
            open_arm_pattern=oa,
            closed_arm_pattern=np.zeros(n, dtype=int),
            center_pattern=np.ones(n, dtype=int),
        )
        parsed = _build_parsed_epm_data({"Subject_1": df})
        result = compute_paradigm_metrics(parsed, "epm")

        warnings = result.get("data_quality_warnings", [])
        entry_warnings = [w for w in warnings if w.get("metric") == "total_entry_count"]
        assert len(entry_warnings) >= 1

    def test_epm_paradigm_no_warning_with_good_data(self):
        from ethoinsight.metrics import compute_paradigm_metrics

        df = _make_epm_df(n_frames=300)  # 5 open + 5 closed = 10 entries, OK
        parsed = _build_parsed_epm_data(
            {
                "S1": df,
                "S2": _make_epm_df(n_frames=300, seed=7),
                "S3": _make_epm_df(n_frames=300, seed=8),
                "S4": _make_epm_df(n_frames=300, seed=9),
                "S5": _make_epm_df(n_frames=300, seed=10),
            }
        )
        result = compute_paradigm_metrics(parsed, "epm")

        warnings = result.get("data_quality_warnings", [])
        # With n>=5 per group and reasonable entries, should have no EPM-specific warnings
        epm_warnings = [w for w in warnings if w.get("metric") == "total_entry_count"]
        assert len(epm_warnings) == 0


# ============================================================================
# EPM template module tests
# ============================================================================
