"""Unit tests for DataQualityWarning schema (Sprint 0)."""

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import DataQualityWarning


class TestDataQualityWarningCodeTaxonomy:
    """code field must start with SAMPLE / MOTOR / SIGNAL / METHOD / LEGACY."""

    def test_valid_sample_code(self):
        w = DataQualityWarning(severity="warning", metric="n", message="too small", code="SAMPLE.TOO_SMALL")
        assert w.code == "SAMPLE.TOO_SMALL"

    def test_valid_motor_code(self):
        w = DataQualityWarning(severity="warning", metric="velocity", message="low", code="MOTOR.LOW_VELOCITY")
        assert w.code == "MOTOR.LOW_VELOCITY"

    def test_valid_signal_code(self):
        w = DataQualityWarning(severity="info", metric="signal", message="noisy", code="SIGNAL.NOISY")
        assert w.code == "SIGNAL.NOISY"

    def test_valid_method_code(self):
        w = DataQualityWarning(severity="warning", metric="test", message="nonparam", code="METHOD.NONPARAMETRIC")
        assert w.code == "METHOD.NONPARAMETRIC"

    def test_valid_legacy_code(self):
        w = DataQualityWarning(severity="critical", metric="all", message="uncategorized", code="LEGACY.UNCATEGORIZED")
        assert w.code == "LEGACY.UNCATEGORIZED"

    def test_invalid_code_namespace(self):
        with pytest.raises(ValidationError, match="FOO\\.BAR"):
            DataQualityWarning(severity="warning", metric="x", message="y", code="FOO.BAR")

    def test_invalid_code_no_dot(self):
        with pytest.raises(ValidationError, match="BADCODE"):
            DataQualityWarning(severity="warning", metric="x", message="y", code="BADCODE")

    def test_empty_code_rejected(self):
        with pytest.raises(ValidationError):
            DataQualityWarning(severity="warning", metric="x", message="y", code="")


class TestDataQualityWarningEvidence:
    """evidence field: dict[str, Any], default empty dict."""

    def test_evidence_dict(self):
        w = DataQualityWarning(
            severity="warning",
            metric="velocity",
            message="slow",
            code="MOTOR.LOW_VELOCITY",
            evidence={"velocity_median_mm_s": 5.2, "threshold_mm_s": 30.0},
        )
        assert w.evidence == {"velocity_median_mm_s": 5.2, "threshold_mm_s": 30.0}

    def test_evidence_default_empty(self):
        w = DataQualityWarning(severity="info", metric="x", message="y", code="SAMPLE.OK")
        assert w.evidence == {}

    def test_evidence_rejects_non_dict(self):
        with pytest.raises(ValidationError):
            DataQualityWarning(
                severity="info",
                metric="x",
                message="y",
                code="SAMPLE.OK",
                evidence=5.2,
            )


class TestDataQualityWarningBlocksDownstream:
    """blocks_downstream: bool, default False."""

    def test_default_false(self):
        w = DataQualityWarning(severity="info", metric="x", message="y", code="SAMPLE.OK")
        assert w.blocks_downstream is False

    def test_explicit_true(self):
        w = DataQualityWarning(
            severity="critical",
            metric="x",
            message="y",
            code="SAMPLE.TOO_SMALL",
            blocks_downstream=True,
        )
        assert w.blocks_downstream is True

    def test_critical_not_auto_block(self):
        """Severity critical does NOT auto-set blocks_downstream; explicit field only."""
        w = DataQualityWarning(
            severity="critical",
            metric="x",
            message="y",
            code="SAMPLE.TOO_SMALL",
        )
        assert w.blocks_downstream is False
