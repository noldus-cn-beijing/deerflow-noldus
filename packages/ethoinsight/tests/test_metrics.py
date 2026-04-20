"""Tests for ethoinsight.metrics and ethoinsight.charts.

Uses real demo data from the zebrafish shoaling paradigm and EPM.
"""

from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ethoinsight import charts, metrics, parse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Resolve project root by finding noldus-insight in the path
def _find_project_root() -> str:
    """Find the noldus-insight project root."""
    # Try common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "demo-data", "DemoData"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "demo-data", "DemoData"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "demo-data", "DemoData"),
    ]
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isdir(norm):
            return norm
    # Fallback: walk up from __file__ looking for demo-data
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidate = os.path.join(d, "demo-data", "DemoData")
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    return ""


DEMO_DIR = _find_project_root()

SHOALING_DIR = os.path.join(DEMO_DIR, "斑马鱼鱼群行为")
EPM_DIR = os.path.join(DEMO_DIR, "高架十字迷宫")


def _shoaling_base_files() -> list[str]:
    """Return the 5 base shoaling trajectory files (no numeric suffix)."""
    pattern = os.path.join(SHOALING_DIR, "轨迹-*Subject ?.txt")
    files = sorted(glob.glob(pattern))
    return files


@pytest.fixture(scope="module")
def shoaling_parsed():
    """Parse the 5 base shoaling files."""
    files = _shoaling_base_files()
    if not files:
        pytest.skip("Shoaling demo data not found")
    return parse.parse_batch(files)


@pytest.fixture(scope="module")
def shoaling_single_df():
    """Parse a single shoaling trajectory file."""
    files = _shoaling_base_files()
    if not files:
        pytest.skip("Shoaling demo data not found")
    return parse.parse_trajectory(files[0])


@pytest.fixture(scope="module")
def epm_parsed():
    """Parse base EPM files."""
    pattern = os.path.join(EPM_DIR, "轨迹-*Subject ?.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip("EPM demo data not found")
    return parse.parse_batch(files)


# ---------------------------------------------------------------------------
# Test: compute_distance_moved
# ---------------------------------------------------------------------------


class TestComputeDistanceMoved:
    def test_positive_total(self, shoaling_single_df):
        result = metrics.compute_distance_moved(shoaling_single_df)
        assert result is not None
        assert result > 0

    def test_returns_none_for_missing_column(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        assert metrics.compute_distance_moved(df) is None


# ---------------------------------------------------------------------------
# Test: compute_velocity_stats
# ---------------------------------------------------------------------------


class TestComputeVelocityStats:
    def test_returns_dict(self, shoaling_single_df):
        result = metrics.compute_velocity_stats(shoaling_single_df)
        assert result is not None
        assert set(result.keys()) == {"mean", "std", "max", "min", "median"}

    def test_values_reasonable(self, shoaling_single_df):
        result = metrics.compute_velocity_stats(shoaling_single_df)
        assert result["mean"] > 0
        assert result["max"] >= result["mean"]
        assert result["min"] <= result["mean"]

    def test_returns_none_for_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert metrics.compute_velocity_stats(df) is None


# ---------------------------------------------------------------------------
# Test: compute_inter_individual_distance
# ---------------------------------------------------------------------------


class TestComputeIID:
    def test_returns_dataframe(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        iid = metrics.compute_inter_individual_distance(subjects)
        assert iid is not None
        assert isinstance(iid, pd.DataFrame)
        assert "trial_time" in iid.columns
        assert "mean_iid" in iid.columns
        assert len(iid) > 0

    def test_mean_iid_positive(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        iid = metrics.compute_inter_individual_distance(subjects)
        assert iid["mean_iid"].mean() > 0

    def test_needs_at_least_2_subjects(self):
        df = pd.DataFrame({
            "trial_time": [0, 1, 2],
            "x_center": [1.0, 2.0, 3.0],
            "y_center": [1.0, 2.0, 3.0],
        })
        result = metrics.compute_inter_individual_distance({"s1": df})
        assert result is None


# ---------------------------------------------------------------------------
# Test: compute_nearest_neighbor_distance
# ---------------------------------------------------------------------------


class TestComputeNND:
    def test_returns_dataframe(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        nnd = metrics.compute_nearest_neighbor_distance(subjects)
        assert nnd is not None
        assert "nnd" in nnd.columns
        assert "subject" in nnd.columns

    def test_nnd_less_or_equal_iid(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        iid = metrics.compute_inter_individual_distance(subjects)
        nnd = metrics.compute_nearest_neighbor_distance(subjects)
        # Mean NND should be <= mean IID (nearest is closer than average)
        mean_nnd = nnd["nnd"].mean()
        mean_iid = iid["mean_iid"].mean()
        assert mean_nnd <= mean_iid


# ---------------------------------------------------------------------------
# Test: compute_group_polarity
# ---------------------------------------------------------------------------


class TestComputeGroupPolarity:
    def test_returns_dataframe(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        pol = metrics.compute_group_polarity(subjects)
        assert pol is not None
        assert "polarity" in pol.columns
        assert len(pol) > 0

    def test_polarity_in_range(self, shoaling_parsed):
        subjects = shoaling_parsed["subjects"]
        pol = metrics.compute_group_polarity(subjects)
        assert pol["polarity"].min() >= 0
        assert pol["polarity"].max() <= 1.0 + 1e-9  # numerical tolerance


# ---------------------------------------------------------------------------
# Test: compute_open_arm_time_ratio (EPM)
# ---------------------------------------------------------------------------


class TestComputeOpenArmTimeRatio:
    def test_epm_open_arm_ratio(self, epm_parsed):
        subjects = epm_parsed["subjects"]
        # At least one subject should have open arm zone data
        found = False
        for name, df in subjects.items():
            ratio = metrics.compute_open_arm_time_ratio(df)
            if ratio is not None:
                found = True
                assert 0 <= ratio <= 1
        if not found:
            pytest.skip("No EPM files with open arm zone columns")


# ---------------------------------------------------------------------------
# Test: compute_paradigm_metrics
# ---------------------------------------------------------------------------


class TestComputeParadigmMetrics:
    def test_shoaling_full(self, shoaling_parsed):
        groups = {
            "control": ["Subject 1", "Subject 2"],
            "treatment": ["Subject 3", "Subject 4", "Subject 5"],
        }
        m = metrics.compute_paradigm_metrics(
            shoaling_parsed, "shoaling", groups=groups,
        )
        assert m["paradigm"] == "shoaling"
        assert "per_subject" in m
        assert "group_summary" in m
        assert "timeseries" in m
        assert "metadata" in m
        # Check structure
        assert len(m["per_subject"]) == 5
        assert "control" in m["group_summary"]
        assert "treatment" in m["group_summary"]
        # Check metrics exist
        assert "distance_moved" in m["per_subject"]["Subject 1"]

    def test_metadata(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        assert m["metadata"]["n_subjects"] == 5
        assert m["metadata"]["n_files"] == 5

    def test_group_summary_has_stats(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        grp = m["group_summary"]["all"]
        assert "distance_moved" in grp
        info = grp["distance_moved"]
        assert "mean" in info
        assert "std" in info
        assert "n" in info
        assert "values" in info


# ---------------------------------------------------------------------------
# Test: save_to_csv
# ---------------------------------------------------------------------------


class TestSaveToCSV:
    def test_write_and_read(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            result = metrics.save_to_csv(m, path)
            assert result == path
            assert os.path.exists(path)
            df = pd.read_csv(path)
            assert "subject" in df.columns
            assert len(df) == 5  # 5 subjects
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: charts
# ---------------------------------------------------------------------------


class TestCharts:
    def test_box_plot(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = charts.box_plot(m, ["distance_moved"], output_path=path)
            assert result == path
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_bar_chart(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = charts.bar_chart(m, ["distance_moved"], output_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_trajectory_plot(self, shoaling_parsed):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = charts.trajectory_plot(
                shoaling_parsed["all_data"], output_path=path,
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_timeseries_plot(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        iid = m["timeseries"].get("inter_individual_distance")
        if iid is None:
            pytest.skip("No IID timeseries")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = charts.timeseries_plot(iid, y_col="mean_iid", output_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_violin_plot(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = charts.violin_plot(m, ["distance_moved"], output_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Test: data boundary defences (commit 2 — fake per-subject IID/polarity bug)
# ---------------------------------------------------------------------------


class TestShoalingGroupMetricsNotFakedPerSubject:
    """Regression tests for the zero-variance IID/polarity bug.

    The previous implementation broadcast a single group-level mean_iid /
    mean_polarity scalar to every subject's per_subject dict, producing
    identical values across fish and thus zero between-subject variance.
    IID and polarity are group-level metrics and must not appear as
    per-subject scalars.
    """

    def test_mean_iid_not_in_per_subject(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        for subject, sdict in m["per_subject"].items():
            assert "mean_iid" not in sdict, (
                f"{subject} has fake per-subject mean_iid; "
                "IID is a group-level metric."
            )

    def test_mean_polarity_not_in_per_subject(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        for subject, sdict in m["per_subject"].items():
            assert "mean_polarity" not in sdict, (
                f"{subject} has fake per-subject mean_polarity; "
                "polarity is a group-level metric."
            )

    def test_group_level_metrics_populated_for_valid_shoaling(self, shoaling_parsed):
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        glm = m.get("group_level_metrics", {})
        assert isinstance(glm.get("mean_iid"), float)
        assert glm["mean_iid"] > 0
        assert isinstance(glm.get("mean_polarity"), float)
        assert 0 <= glm["mean_polarity"] <= 1.0 + 1e-9

    def test_mean_nnd_still_per_subject(self, shoaling_parsed):
        """NND is genuinely per-subject and must still appear."""
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling")
        nnd_values = [
            sdict.get("mean_nnd")
            for sdict in m["per_subject"].values()
            if sdict.get("mean_nnd") is not None
        ]
        assert len(nnd_values) > 0, "mean_nnd should remain per-subject"
        # NND per-subject should NOT all be identical (different fish, different neighbours)
        assert len(set(nnd_values)) > 1, "mean_nnd should vary across subjects"


class TestShoalingSingleSubjectBoundary:
    """When only 1 subject is tracked, group metrics must be flagged inapplicable."""

    def test_single_subject_marks_iid_inapplicable(self):
        df = pd.DataFrame({
            "trial_time": [0.0, 0.1, 0.2],
            "x_center": [1.0, 2.0, 3.0],
            "y_center": [1.0, 2.0, 3.0],
            "distance_moved": [0.0, 1.4, 1.4],
            "velocity": [0.0, 14.0, 14.0],
        })
        parsed = {
            "subjects": {"only_fish": df},
            "summary": {"total_files": 1, "duration_seconds": 0.2},
        }
        m = metrics.compute_paradigm_metrics(parsed, "shoaling")
        glm = m.get("group_level_metrics", {})
        assert isinstance(glm.get("mean_iid"), dict)
        assert glm["mean_iid"].get("applicable") is False
        assert "reason" in glm["mean_iid"]
        assert isinstance(glm.get("mean_polarity"), dict)
        assert glm["mean_polarity"].get("applicable") is False

    def test_single_subject_emits_quality_warning(self):
        df = pd.DataFrame({
            "trial_time": [0.0, 0.1, 0.2],
            "x_center": [1.0, 2.0, 3.0],
            "y_center": [1.0, 2.0, 3.0],
            "distance_moved": [0.0, 1.4, 1.4],
            "velocity": [0.0, 14.0, 14.0],
        })
        parsed = {
            "subjects": {"only_fish": df},
            "summary": {"total_files": 1, "duration_seconds": 0.2},
        }
        m = metrics.compute_paradigm_metrics(parsed, "shoaling")
        warnings = m.get("data_quality_warnings", [])
        assert any(
            "1 subject" in w.get("message", "") or "mean_iid" in w.get("metric", "")
            for w in warnings
        ), f"Expected single-subject warning, got {warnings}"


class TestSmallSampleWarning:
    """n<3 in any group should emit a critical data_quality_warning."""

    def test_n_equals_2_group_emits_critical_warning(self, shoaling_parsed):
        groups = {
            "control": ["Subject 1", "Subject 2"],  # n=2
            "treatment": ["Subject 3", "Subject 4", "Subject 5"],  # n=3
        }
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling", groups=groups)
        warnings = m.get("data_quality_warnings", [])
        critical = [w for w in warnings if w.get("severity") == "critical"]
        assert critical, f"Expected at least one critical warning for n=2 group, got {warnings}"
        assert any(
            "control" in w.get("message", "") and "n=2" in w.get("message", "")
            for w in critical
        )

    def test_all_groups_n_ge_3_no_critical_warning(self, shoaling_parsed):
        groups = {
            "control": ["Subject 1", "Subject 2", "Subject 3"],
            "treatment": ["Subject 4", "Subject 5"],  # n=2 — still critical for this group
        }
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling", groups=groups)
        # Sanity: treatment has n=2 so there IS a critical warning
        warnings = m.get("data_quality_warnings", [])
        assert any(w.get("severity") == "critical" for w in warnings)

    def test_pipeline_does_not_block_on_small_sample(self, shoaling_parsed):
        """Warnings should not short-circuit metric computation."""
        groups = {"small": ["Subject 1", "Subject 2"]}
        m = metrics.compute_paradigm_metrics(shoaling_parsed, "shoaling", groups=groups)
        # Group summary should still contain stats even though n<3
        assert "small" in m["group_summary"]
        assert "distance_moved" in m["group_summary"]["small"]
        assert m["group_summary"]["small"]["distance_moved"]["n"] == 2

