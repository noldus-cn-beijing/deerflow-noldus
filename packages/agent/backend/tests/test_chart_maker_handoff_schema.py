"""Unit tests for ChartMakerHandoff schema (Sprint 0)."""

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import ChartMakerHandoff, FailedChart


class TestChartMakerMinimumFields:
    """paradigm + summary required; others have defaults."""

    def test_minimum_fields(self):
        # status="failed" so the completed→non-empty-chart_files validator (Spec S3) does
        # not fire — this test verifies field defaults, not the completed contract.
        h = ChartMakerHandoff(status="failed", paradigm="fst", summary="generated charts", analysis_config_id="PENDING_SPRINT_4.5")
        assert h.status == "failed"
        assert h.paradigm == "fst"
        assert h.summary == "generated charts"
        assert h.chart_files == []
        assert h.failed_charts == []
        assert h.analysis_config_id == "PENDING_SPRINT_4.5"

    def test_all_fields(self):
        h = ChartMakerHandoff(
            status="partial",
            paradigm="epm",
            summary="some charts",
            chart_files=["/mnt/user-data/outputs/heatmap.png"],
            failed_charts=[FailedChart(chart_id="trajectory", reason="no data")],
            analysis_config_id="abc123",
        )
        assert h.status == "partial"
        assert len(h.chart_files) == 1
        assert h.failed_charts[0].chart_id == "trajectory"


class TestChartMakerPathValidation:
    """chart_files must be under /mnt/user-data/outputs/."""

    def test_valid_outputs_path(self):
        h = ChartMakerHandoff(
            paradigm="fst",
            summary="ok",
            chart_files=["/mnt/user-data/outputs/plot.png"],
            analysis_config_id="x",
        )
        assert len(h.chart_files) == 1

    def test_workspace_path_rejected(self):
        with pytest.raises(ValidationError, match="workspace"):
            ChartMakerHandoff(
                paradigm="fst",
                summary="bad",
                chart_files=["/mnt/user-data/workspace/x.png"],
                analysis_config_id="x",
            )

    def test_arbitrary_path_rejected(self):
        with pytest.raises(ValidationError, match="outputs"):
            ChartMakerHandoff(
                paradigm="fst",
                summary="bad",
                chart_files=["/tmp/plot.png"],
                analysis_config_id="x",
            )

    def test_empty_chart_files_ok_for_non_completed(self):
        # Empty chart_files is legal only for partial/failed (seal tool contract:
        # 全部失败时为 failed). The completed→non-empty validator (Spec S3) rejects
        # completed + empty as a dumb-failure, so use failed/partial here.
        h = ChartMakerHandoff(status="failed", paradigm="fst", summary="no charts", analysis_config_id="x")
        assert h.chart_files == []
        h_partial = ChartMakerHandoff(status="partial", paradigm="fst", summary="some failed", analysis_config_id="x")
        assert h_partial.chart_files == []


class TestFailedChart:
    """FailedChart: chart_id + reason required."""

    def test_valid(self):
        fc = FailedChart(chart_id="trajectory_heatmap", reason="no data")
        assert fc.chart_id == "trajectory_heatmap"

    def test_missing_chart_id(self):
        with pytest.raises(ValidationError):
            FailedChart(reason="no data")


class TestRemainingCharts:
    """P5 (spec 2026-06-17-chart-budget-by-type): remaining_charts 降级指纹字段。

    chart 预算按类型而非数量后，aggregate 全画、per_subject 代表性子集入 charts_budget_remaining；
    chart-maker 把它透传进 handoff 的 remaining_charts[]，让 lead/用户知道还能画更多个体图。
    """

    def test_default_empty(self):
        """无降级时 remaining_charts=[]（向后兼容旧 handoff）。"""
        h = ChartMakerHandoff(status="failed", paradigm="fst", summary="ok", analysis_config_id="x")
        assert h.remaining_charts == []

    def test_populated_from_budget_truncation(self):
        """预算截断 → remaining_charts 非空，复用 FailedChart 结构。"""
        h = ChartMakerHandoff(
            status="partial",
            paradigm="epm",
            summary="aggregate 全画 + 代表性 per_subject",
            chart_files=["/mnt/user-data/outputs/plot_box_open_arm.png"],
            remaining_charts=[
                FailedChart(chart_id="trajectory", reason="chart_budget_truncated"),
                FailedChart(chart_id="heatmap", reason="chart_budget_truncated"),
            ],
            analysis_config_id="x",
        )
        assert len(h.remaining_charts) == 2
        assert h.remaining_charts[0].chart_id == "trajectory"
        assert "truncated" in h.remaining_charts[1].reason

    def test_remaining_does_not_block_completed_validator(self):
        """completed + chart_files 非空仍合法；remaining_charts 与 completed 不冲突
        （completed 校验只看 chart_files 非空，降级指纹独立）。"""
        h = ChartMakerHandoff(
            status="completed",
            paradigm="epm",
            summary="aggregate 优先画完",
            chart_files=["/mnt/user-data/outputs/plot_box_open_arm.png"],
            remaining_charts=[FailedChart(chart_id="trajectory", reason="chart_budget_truncated")],
            analysis_config_id="x",
        )
        assert h.status == "completed"
        assert len(h.remaining_charts) == 1
