"""Tests for DataQualityWarning normalisation in handoff_schemas.

Layer 2 of seal-robustness design: the model_validator on DataQualityWarning
conservatively normalises common LLM typos before strict validation.

Covers:
1. code: underscore → DOT for known prefixes (SAMPLE/MOTOR/SIGNAL/METHOD)
2. code: pure semantic errors left as-is (no known prefix match)
3. metric: None → "all"
4. evidence: str → {"note": <original>}
5. evidence: non-dict non-str → {}
6. chart/report handoff schemas not affected
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import (
    WARNING_CODE_PREFIXES,
    CodeExecutorHandoff,
    DataQualityWarning,
    ReportWriterHandoff,
)


class TestCodeUnderscoreNormalization:
    """code field: underscore → DOT for known prefixes."""

    def test_sample_underscore_to_dot(self):
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE_TOO_SMALL",
        )
        assert w.code == "SAMPLE.TOO_SMALL"

    def test_motor_underscore_to_dot(self):
        w = DataQualityWarning(
            severity="warning",
            metric="distance_moved",
            message="motor issue",
            code="MOTOR_THRESHOLD_HIGH",
        )
        assert w.code == "MOTOR.THRESHOLD_HIGH"

    def test_signal_underscore_to_dot(self):
        w = DataQualityWarning(
            severity="info",
            metric="immobility",
            message="signal",
            code="SIGNAL_NOISY",
        )
        assert w.code == "SIGNAL.NOISY"

    def test_method_underscore_to_dot(self):
        w = DataQualityWarning(
            severity="info",
            metric="all",
            message="method",
            code="METHOD_DEPRECATED",
        )
        assert w.code == "METHOD.DEPRECATED"

    def test_pure_semantic_error_rejected(self):
        """Codes that don't start with a known prefix fail validation
        (field_validator rejects them after normalisation leaves them untouched)."""
        with pytest.raises(ValidationError):
            DataQualityWarning(
                severity="critical",
                metric="all",
                message="insufficient sample",
                code="insufficient_sample",
            )

    def test_already_dot_format_untouched(self):
        """Already-DOT format passes through unchanged."""
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="ok",
            code="SAMPLE.TOO_SMALL",
        )
        assert w.code == "SAMPLE.TOO_SMALL"

    def test_no_underscore_rejected(self):
        """Codes without underscores and without DOT are rejected by field_validator."""
        with pytest.raises(ValidationError):
            DataQualityWarning(
                severity="info",
                metric="all",
                message="ok",
                code="SAMPLETOOSMALL",
            )

    def test_multi_underscore_only_first_replaced(self):
        """Only the first underscore (segmenting prefix from name) is replaced."""
        w = DataQualityWarning(
            severity="info",
            metric="all",
            message="ok",
            code="SAMPLE_TOO_SMALL_DETAIL",
        )
        # replace("_", ".", 1) → SAMPLE.TOO_SMALL_DETAIL
        assert w.code == "SAMPLE.TOO_SMALL_DETAIL"


class TestMetricNoneNormalization:
    """metric field: None → "all"."""

    def test_metric_none_becomes_all(self):
        w = DataQualityWarning(
            severity="critical",
            metric=None,  # type: ignore[arg-type]
            message="too small",
            code="SAMPLE.TOO_SMALL",
        )
        assert w.metric == "all"

    def test_metric_string_passes_through(self):
        w = DataQualityWarning(
            severity="critical",
            metric="immobility",
            message="too small",
            code="SAMPLE.TOO_SMALL",
        )
        assert w.metric == "immobility"


class TestEvidenceNormalization:
    """evidence field: str → {"note": <original>}; non-dict non-str → {}."""

    def test_evidence_str_wrapped(self):
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE.TOO_SMALL",
            evidence="n_per_group=1, required=2",
        )
        assert w.evidence == {"note": "n_per_group=1, required=2"}

    def test_evidence_dict_passes_through(self):
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE.TOO_SMALL",
            evidence={"n_per_group": 1, "required": 2},
        )
        assert w.evidence == {"n_per_group": 1, "required": 2}

    def test_evidence_int_becomes_empty_dict(self):
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE.TOO_SMALL",
            evidence=42,  # type: ignore[arg-type]
        )
        assert w.evidence == {}

    def test_evidence_list_becomes_empty_dict(self):
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE.TOO_SMALL",
            evidence=[1, 2, 3],  # type: ignore[arg-type]
        )
        assert w.evidence == {}

    def test_evidence_default_empty_dict(self):
        """Default evidence is empty dict when not provided."""
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="too small",
            code="SAMPLE.TOO_SMALL",
        )
        assert w.evidence == {}


class TestWarningCodePrefixes:
    """Verify the prefix set is importable and correct."""

    def test_contains_expected_prefixes(self):
        assert "SAMPLE" in WARNING_CODE_PREFIXES
        assert "MOTOR" in WARNING_CODE_PREFIXES
        assert "SIGNAL" in WARNING_CODE_PREFIXES
        assert "METHOD" in WARNING_CODE_PREFIXES
        assert len(WARNING_CODE_PREFIXES) == 4


class TestCodeExecutorHandoffNotAffected:
    """Verify CodeExecutorHandoff still parses correctly with normalised warnings."""

    def test_roundtrip_with_normalised_warning(self):
        h = CodeExecutorHandoff(
            status="partial",
            summary="warnings test",
            paradigm="fst",
            analysis_config_id="test-config-1",
            data_quality_warnings=[
                {
                    "severity": "critical",
                    "metric": None,  # normalised to "all"
                    "message": "n_per_group=1",
                    "code": "SAMPLE_TOO_SMALL",  # normalised to SAMPLE.TOO_SMALL
                    "evidence": "n_per_group=1",  # wrapped to {"note": "..."}
                },
            ],
        )
        assert h.data_quality_warnings[0].metric == "all"
        assert h.data_quality_warnings[0].code == "SAMPLE.TOO_SMALL"
        assert h.data_quality_warnings[0].evidence == {"note": "n_per_group=1"}

    def test_existing_test_still_passes(self):
        """Verify existing production_shape test still works with required fields."""
        payload = {
            "status": "partial",
            "summary": "warnings",
            "paradigm": "epm",
            "analysis_config_id": "test-config-2",
            "data_quality_warnings": [
                {"severity": "critical", "metric": "all", "message": "n<3 in group control", "code": "SAMPLE.TOO_SMALL"},
                {"severity": "warning", "metric": "iid", "message": "zero variance", "code": "SIGNAL.LOW_VARIANCE"},
            ],
        }
        h = CodeExecutorHandoff(**payload)
        assert len(h.data_quality_warnings) == 2
        assert h.data_quality_warnings[0].severity == "critical"


class TestReportWriterHandoffNotAffected:
    """Verify ReportWriterHandoff is not affected by normalisation."""

    def test_normal_handoff(self):
        h = ReportWriterHandoff(
            status="completed",
            report_path="/mnt/user-data/outputs/report.md",
        )
        assert h.status == "completed"
        assert h.report_path == "/mnt/user-data/outputs/report.md"


class TestNoFalsePositives:
    """Edge cases: normalisation must not corrupt valid data."""

    def test_valid_warning_unchanged(self):
        w = DataQualityWarning(
            severity="warning",
            metric="distance_moved",
            message="high variance",
            code="SAMPLE.TOO_SMALL",
            evidence={"cv": 0.8},
        )
        assert w.severity == "warning"
        assert w.metric == "distance_moved"
        assert w.message == "high variance"
        assert w.code == "SAMPLE.TOO_SMALL"
        assert w.evidence == {"cv": 0.8}

    def test_blocks_downstream_preserved(self):
        """blocks_downstream field survives round-trip."""
        w = DataQualityWarning(
            severity="critical",
            metric="all",
            message="blocks",
            code="SAMPLE.TOO_SMALL",
            blocks_downstream=True,
        )
        assert w.blocks_downstream is True
