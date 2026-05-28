"""Tests for GateSignals Pydantic model in subagent handoff_schemas.

GateSignals is the structured decision payload lead reads from subagent's
final AIMessage (Step 1.5 quality-gate decisions without inflating context).
"""

import pytest
from pydantic import ValidationError


def test_gate_signals_default_construction_is_safe():
    """Empty GateSignals must construct cleanly with safe defaults."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals()
    assert g.data_quality == {}
    assert g.statistical_validity == "ok"
    assert g.errors_count == 0


def test_gate_signals_full_construction():
    """All fields populated; ensure structured types come through."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals(
        data_quality={
            "critical_count": 1,
            "warning_count": 2,
            "critical_items": ["IID 为常数（单鱼模式）"],
        },
        statistical_validity="warning",
        errors_count=0,
    )
    assert g.data_quality["critical_count"] == 1
    assert g.data_quality["critical_items"][0].startswith("IID")
    assert g.statistical_validity == "warning"


def test_gate_signals_rejects_invalid_validity():
    """statistical_validity is a constrained literal."""
    from deerflow.subagents.handoff_schemas import GateSignals

    with pytest.raises(ValidationError):
        GateSignals(statistical_validity="unknown")  # noqa: PIE837


def test_gate_signals_allows_extra_fields():
    """Future-proof: extra keys allowed (extra='allow' in model_config)."""
    from deerflow.subagents.handoff_schemas import GateSignals

    g = GateSignals.model_validate({
        "data_quality": {},
        "statistical_validity": "ok",
        "errors_count": 0,
        "future_field": "ignored gracefully",
    })
    # extra='allow' makes future_field accessible
    assert g.statistical_validity == "ok"


def test_code_executor_handoff_has_optional_gate_signals():
    """CodeExecutorHandoff must accept gate_signals=None (optional in file)."""
    from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

    h = CodeExecutorHandoff(status="completed", summary="ok", paradigm="fst", analysis_config_id="test-config-id")
    assert h.gate_signals is None


def test_code_executor_handoff_accepts_gate_signals():
    from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, GateSignals

    h = CodeExecutorHandoff(
        status="completed",
        summary="ok",
        paradigm="fst",
        analysis_config_id="test-config-id",
        gate_signals=GateSignals(statistical_validity="warning"),
    )
    assert h.gate_signals.statistical_validity == "warning"
