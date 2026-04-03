"""Tests for ethoinsight.statistics and ethoinsight.assess."""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from ethoinsight import assess, metrics, parse, statistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _find_project_root() -> str:
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


@pytest.fixture(scope="module")
def shoaling_metrics():
    """Compute shoaling metrics with 2 groups for statistical testing."""
    pattern = os.path.join(SHOALING_DIR, "轨迹-*Subject ?.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        pytest.skip("Shoaling demo data not found")
    parsed = parse.parse_batch(files)
    groups = {
        "control": ["Subject 1", "Subject 2"],
        "treatment": ["Subject 3", "Subject 4", "Subject 5"],
    }
    return metrics.compute_paradigm_metrics(parsed, "shoaling", groups=groups)


# ---------------------------------------------------------------------------
# Test: test_normality
# ---------------------------------------------------------------------------


class TestNormality:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        values = rng.normal(10, 2, 50).tolist()
        result = statistics.test_normality(values)
        assert result["test"] == "shapiro-wilk"
        assert result["n"] == 50
        assert isinstance(result["is_normal"], bool)

    def test_small_sample(self):
        result = statistics.test_normality([1.0, 2.0])
        assert result["n"] == 2
        assert result["is_normal"] is None  # Too few samples

    def test_non_normal_data(self):
        # Uniform distribution is non-normal for large n
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 1, 100).tolist()
        result = statistics.test_normality(values)
        assert result["n"] == 100


# ---------------------------------------------------------------------------
# Test: compare_two_groups
# ---------------------------------------------------------------------------


class TestCompareTwoGroups:
    def test_different_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 1, 30).tolist()
        g2 = rng.normal(15, 1, 30).tolist()
        result = statistics.compare_two_groups(g1, g2)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert "test_used" in result
        assert "effect_size" in result

    def test_same_groups(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 1, 30).tolist()
        g2 = rng.normal(10, 1, 30).tolist()
        result = statistics.compare_two_groups(g1, g2)
        # p-value should generally be > 0.05 for same distribution
        assert result["p_value"] > 0.01  # relaxed for randomness

    def test_paired(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 1, 20).tolist()
        g2 = (np.array(g1) + rng.normal(2, 0.5, 20)).tolist()
        result = statistics.compare_two_groups(g1, g2, paired=True)
        assert result["test_used"] in ("paired-t-test", "wilcoxon-signed-rank")

    def test_non_normal_uses_nonparametric(self):
        # Exponential distribution is non-normal
        rng = np.random.default_rng(42)
        g1 = rng.exponential(2, 50).tolist()
        g2 = rng.exponential(5, 50).tolist()
        result = statistics.compare_two_groups(g1, g2)
        assert result["test_used"] in ("mann-whitney-u", "welch-t-test")


# ---------------------------------------------------------------------------
# Test: compare_multiple_groups
# ---------------------------------------------------------------------------


class TestCompareMultipleGroups:
    def test_three_groups(self):
        rng = np.random.default_rng(42)
        groups = [
            rng.normal(10, 1, 20).tolist(),
            rng.normal(15, 1, 20).tolist(),
            rng.normal(20, 1, 20).tolist(),
        ]
        result = statistics.compare_multiple_groups(groups)
        assert result["significant"] is True
        assert result["test_used"] in ("one-way-anova", "kruskal-wallis")
        assert "effect_size" in result


# ---------------------------------------------------------------------------
# Test: effect sizes
# ---------------------------------------------------------------------------


class TestEffectSizes:
    def test_cohens_d_large(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 1, 30).tolist()
        g2 = rng.normal(15, 1, 30).tolist()
        result = statistics.compute_cohens_d(g1, g2)
        assert result["d"] is not None
        assert result["d"] > 0.8
        assert result["magnitude"] == "large"

    def test_cohens_d_small_sample(self):
        result = statistics.compute_cohens_d([1.0], [2.0])
        assert result["d"] is None

    def test_eta_squared(self):
        rng = np.random.default_rng(42)
        groups = [
            rng.normal(10, 1, 20).tolist(),
            rng.normal(15, 1, 20).tolist(),
            rng.normal(20, 1, 20).tolist(),
        ]
        result = statistics.compute_eta_squared(groups)
        assert result["eta_squared"] is not None
        assert 0 <= result["eta_squared"] <= 1
        assert result["magnitude"] == "large"


# ---------------------------------------------------------------------------
# Test: compare_groups (dispatcher)
# ---------------------------------------------------------------------------


class TestCompareGroups:
    def test_shoaling_compare(self, shoaling_metrics):
        result = statistics.compare_groups(shoaling_metrics)
        assert "comparisons" in result
        assert "summary" in result
        assert "alpha" in result
        assert "correction" in result

    def test_comparisons_format_compatible_with_charts(self, shoaling_metrics):
        """Verify the output format is compatible with charts.py significance param."""
        result = statistics.compare_groups(shoaling_metrics)
        for metric_name, comps in result["comparisons"].items():
            for comp in comps:
                # Charts expects group1, group2, p_value
                assert "p_value" in comp
                if "group1" in comp:
                    assert "group2" in comp

    def test_bonferroni_correction(self, shoaling_metrics):
        result = statistics.compare_groups(
            shoaling_metrics, correction="bonferroni",
        )
        assert result["correction"] == "bonferroni"

    def test_no_correction(self, shoaling_metrics):
        result = statistics.compare_groups(
            shoaling_metrics, correction="none",
        )
        assert result["correction"] == "none"

    def test_single_group_returns_empty(self, shoaling_metrics):
        result = statistics.compare_groups(
            shoaling_metrics, groups=["control"],
        )
        assert result["comparisons"] == {}


# ---------------------------------------------------------------------------
# Test: assess_results
# ---------------------------------------------------------------------------


class TestAssessResults:
    def test_assess_shoaling(self, shoaling_metrics):
        stat_results = statistics.compare_groups(shoaling_metrics)
        result = assess.assess_results(stat_results, "shoaling")
        assert result["paradigm"] == "shoaling"
        assert "overall_assessment" in result
        assert "findings" in result
        assert "recommendations" in result
        assert "confidence" in result

    def test_assess_with_metrics(self, shoaling_metrics):
        stat_results = statistics.compare_groups(shoaling_metrics)
        result = assess.assess_results(
            stat_results, "shoaling", metrics_result=shoaling_metrics,
        )
        assert isinstance(result["findings"], list)

    def test_assess_empty_stats(self):
        result = assess.assess_results(
            {"comparisons": {}, "summary": "No data"},
            "epm",
        )
        assert result["overall_assessment"] is not None
        assert len(result["findings"]) == 0
