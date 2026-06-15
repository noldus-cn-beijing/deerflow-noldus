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
