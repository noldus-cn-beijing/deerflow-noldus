"""Contract tests for subagent handoff JSON shapes.

Validates that the Pydantic handoff schemas in deerflow.subagents.handoff_schemas
accept valid inputs and reject invalid ones. LLM behaviour and full pipeline
integration are NOT tested here — the real-data pipeline contract lives alongside
the ethoinsight algorithmic tests.

See docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md section 5 (L1 —
Handoff contract enforcement).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Avoid circular import via deerflow.subagents.__init__
_executor_mock = MagicMock()
_executor_mock.SubagentExecutor = MagicMock
_executor_mock.SubagentResult = MagicMock
_executor_mock.SubagentStatus = MagicMock
_executor_mock.MAX_CONCURRENT_SUBAGENTS = 3
sys.modules.setdefault("deerflow.subagents.executor", _executor_mock)

from deerflow.subagents.handoff_schemas import (  # noqa: E402
    CodeExecutorHandoff,
    DataAnalystHandoff,
    DataQualityWarning,
    ReportWriterHandoff,
)


class TestCodeExecutorHandoffSchema:
    def test_minimal_completed_accepts(self):
        handoff = CodeExecutorHandoff(status="completed", summary="ok", paradigm="fst", analysis_config_id="test-config-id")
        assert handoff.status == "completed"
        assert handoff.errors == []
        assert handoff.data_quality_warnings == []

    def test_rejects_invalid_status(self):
        with pytest.raises(Exception):
            CodeExecutorHandoff(status="unknown", summary="ok", paradigm="fst", analysis_config_id="test-config-id")  # type: ignore[arg-type]

    def test_rejects_missing_status(self):
        with pytest.raises(Exception):
            CodeExecutorHandoff(summary="ok", paradigm="fst", analysis_config_id="test-config-id")  # type: ignore[call-arg]

    def test_accepts_metrics_summary_with_extra_fields(self):
        """Real ethoinsight output has mean/std/n/p25/p75 — extras must pass."""
        payload = {
            "status": "completed",
            "summary": "ok",
            "paradigm": "fst",
            "analysis_config_id": "test-config-id",
            "metrics_summary": {
                "control": {
                    "iid": {"mean": 45.2, "std": 12.3, "n": 6, "p25": 35.0, "p75": 55.0}
                }
            },
        }
        h = CodeExecutorHandoff(**payload)
        stat = h.metrics_summary["control"]["iid"]
        assert stat.mean == 45.2
        assert stat.applicable is True  # default

    def test_accepts_inapplicable_metric(self):
        payload = {
            "status": "completed",
            "summary": "ok",
            "paradigm": "fst",
            "analysis_config_id": "test-config-id",
            "metrics_summary": {
                "control": {
                    "iid": {
                        "applicable": False,
                        "reason": "group metric requires >=2 simultaneously tracked subjects",
                    }
                }
            },
        }
        h = CodeExecutorHandoff(**payload)
        stat = h.metrics_summary["control"]["iid"]
        assert stat.applicable is False
        assert "group metric" in (stat.reason or "")

    def test_data_quality_warnings_roundtrip(self):
        payload = {
            "status": "partial",
            "summary": "warnings",
            "paradigm": "fst",
            "analysis_config_id": "test-config-id",
            "data_quality_warnings": [
                {"severity": "critical", "metric": "all", "message": "n<3 in group control", "code": "SAMPLE.TOO_SMALL", "blocks_downstream": True},
                {"severity": "warning", "metric": "iid", "message": "zero variance", "code": "SIGNAL.TRACKING_LOST"},
            ],
        }
        h = CodeExecutorHandoff(**payload)
        assert len(h.data_quality_warnings) == 2
        assert h.data_quality_warnings[0].severity == "critical"
        assert h.data_quality_warnings[0].code == "SAMPLE.TOO_SMALL"

    def test_rejects_invalid_confidence(self):
        with pytest.raises(Exception):
            CodeExecutorHandoff(status="completed", summary="ok", paradigm="fst", analysis_config_id="test-config-id", confidence=1.5)

    def test_current_production_shape_parses(self):
        """Mirrors the dict currently written by the SOTA glue-script."""
        production_sample = {
            "status": "completed",
            "summary": "Analyzed 5 files, 5 subjects, paradigm: shoaling",
            "paradigm": "shoaling",
            "analysis_config_id": "test-config-id",
            "output_files": {
                "metrics": "/mnt/user-data/outputs/metrics.csv",
                "statistics": "/mnt/user-data/outputs/statistics.json",
                "charts": ["/mnt/user-data/outputs/iid_boxplot.png"],
            },
            "metrics_summary": {
                "control": {"iid": {"mean": 40.0, "std": 10.0, "n": 5}}
            },
            "statistics": {"comparisons": [], "summary": {}},
            "assessment": {"phenotypes": []},
            "metadata": {"paradigm": "shoaling", "n_files": 5, "groups": {"control": ["S1"]}},
            "errors": [],
        }
        h = CodeExecutorHandoff(**production_sample)
        assert h.status == "completed"
        assert h.metadata["paradigm"] == "shoaling"


class TestDataAnalystHandoffSchema:
    def test_minimal_completed_accepts(self):
        h = DataAnalystHandoff(status="completed", analysis_config_id="test-config-id")
        assert h.status == "completed"
        assert h.key_findings == []
        assert h.outlier_findings == []
        assert h.method_warnings == []

    def test_status_is_required(self):
        with pytest.raises(Exception):
            DataAnalystHandoff(analysis_config_id="test-config-id")  # type: ignore[call-arg]

    def test_failed_with_errors(self):
        h = DataAnalystHandoff(
            status="failed",
            analysis_config_id="test-config-id",
            errors=["timeout reading handoff_code_executor.json"],
        )
        assert h.status == "failed"
        assert len(h.errors) == 1


class TestReportWriterHandoffSchema:
    def test_minimal_completed_accepts(self):
        h = ReportWriterHandoff(
            status="completed",
            analysis_config_id="test-config-id",
            report_path="/mnt/user-data/outputs/report.md",
        )
        assert h.sections_written == []

    def test_rejects_missing_path(self):
        with pytest.raises(Exception):
            ReportWriterHandoff(status="completed", analysis_config_id="test-config-id")  # type: ignore[call-arg]


class TestDataQualityWarning:
    def test_rejects_unknown_severity(self):
        with pytest.raises(Exception):
            DataQualityWarning(severity="meh", metric="iid", message="x", code="SAMPLE.TOO_SMALL")  # type: ignore[arg-type]
