"""Tests for Sprint 5: DataQualityGuardrailProvider.

TDD tests for:
1. Blocks task() to data-analyst/chart-maker/report-writer on critical+blocks_downstream
2. Allows task() to knowledge-assistant
3. Auto mode: provider not mounted (verified via agent.py logic, not here)
4. Acknowledged quality: allows task()
5. Deny message contains structured warning payload
"""

import json
from pathlib import Path

import pytest

from deerflow.guardrails.data_quality_provider import DataQualityGuardrailProvider
from deerflow.guardrails.provider import GuardrailRequest


def _make_request(tool_name: str, tool_input: dict | None = None) -> GuardrailRequest:
    """Create a minimal GuardrailRequest."""
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input=tool_input or {},
    )


def _write_handoff(workspace: str, warnings: list[dict]) -> None:
    """Write handoff_code_executor.json with data_quality_warnings."""
    handoff = {
        "status": "completed",
        "summary": "ok",
        "paradigm": "epm",
        "data_quality_warnings": warnings,
    }
    Path(workspace).mkdir(parents=True, exist_ok=True)
    (Path(workspace) / "handoff_code_executor.json").write_text(
        json.dumps(handoff), encoding="utf-8"
    )


def _write_context(workspace: str, gate_completed: list[str] | None = None) -> None:
    """Write experiment-context.json."""
    ctx = {"paradigm": "epm", "gate_completed": gate_completed or ["gate1_paradigm"]}
    Path(workspace).mkdir(parents=True, exist_ok=True)
    (Path(workspace) / "experiment-context.json").write_text(
        json.dumps(ctx), encoding="utf-8"
    )


class TestDataQualityGuardrailProvider:
    """Test the DataQualityGuardrailProvider."""

    def test_allows_non_task_tools(self, tmp_path):
        """Non-task() calls always pass through."""
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("ask_clarification", {"question": "test"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_allows_task_to_knowledge_assistant(self, tmp_path):
        """task(knowledge-assistant) always passes — not in blocked set."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "bad data", "code": "SAMPLE.TOO_SMALL", "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "knowledge-assistant"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_allows_task_to_code_executor(self, tmp_path):
        """task(code-executor) always passes — not in blocked set."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "bad data", "code": "SAMPLE.TOO_SMALL", "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "code-executor"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_blocks_data_analyst_on_critical_blocks_downstream(self, tmp_path):
        """task(data-analyst) blocked when critical+blocks_downstream warning exists."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "样本量不足", "code": "SAMPLE.TOO_SMALL",
             "metric": "all", "blocks_downstream": True, "evidence": {"n_per_group": 1}}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert decision.policy_id == "data-quality-guardrail"
        # Deny message must mention the warning
        msg = decision.reasons[0].message
        assert "SAMPLE.TOO_SMALL" in msg
        assert "ask_clarification" in msg

    def test_blocks_chart_maker(self, tmp_path):
        """task(chart-maker) blocked on critical+blocks_downstream."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "样本量不足", "code": "SAMPLE.TOO_SMALL",
             "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "chart-maker"})
        decision = provider.evaluate(request)
        assert decision.allow is False

    def test_blocks_report_writer(self, tmp_path):
        """task(report-writer) blocked on critical+blocks_downstream."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "样本量不足", "code": "SAMPLE.TOO_SMALL",
             "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "report-writer"})
        decision = provider.evaluate(request)
        assert decision.allow is False

    def test_allows_when_critical_but_not_blocks_downstream(self, tmp_path):
        """Critical warning without blocks_downstream=True does NOT block."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "bad data", "code": "SAMPLE.TOO_SMALL",
             "blocks_downstream": False}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_allows_when_no_critical_warnings(self, tmp_path):
        """Warning-level warnings don't block."""
        _write_handoff(str(tmp_path), [
            {"severity": "warning", "message": "注意样本偏小", "code": "SAMPLE.SMALL",
             "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_allows_when_no_handoff(self, tmp_path):
        """No handoff file → no warnings → allow."""
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_allows_when_quality_acknowledged(self, tmp_path):
        """Acknowledged quality gate → allow even with blocking warnings."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "bad data", "code": "SAMPLE.TOO_SMALL",
             "blocks_downstream": True}
        ])
        _write_context(str(tmp_path), gate_completed=["gate1_paradigm", "gate2_quality_acknowledged"])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_deny_message_contains_structured_payload(self, tmp_path):
        """Deny metadata must carry structured warning payloads."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "样本量不足", "code": "SAMPLE.TOO_SMALL",
             "metric": "all", "blocks_downstream": True,
             "evidence": {"n_per_group": 1, "threshold": 3}}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "warnings" in decision.metadata
        warnings = decision.metadata["warnings"]
        assert len(warnings) == 1
        assert warnings[0]["code"] == "SAMPLE.TOO_SMALL"
        assert warnings[0]["blocks_downstream"] is True
        assert warnings[0]["evidence"]["n_per_group"] == 1

    def test_allows_when_workspace_unresolvable(self):
        """No workspace path → fail-open (don't block)."""
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: None)
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_multiple_blocking_warnings(self, tmp_path):
        """Multiple blocking warnings all appear in deny message."""
        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "warn1", "code": "SAMPLE.TOO_SMALL",
             "blocks_downstream": True},
            {"severity": "critical", "message": "warn2", "code": "MOTOR.LOW_VELOCITY",
             "blocks_downstream": True},
            {"severity": "warning", "message": "not blocking", "code": "SAMPLE.SMALL",
             "blocks_downstream": True},
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert len(decision.metadata["warnings"]) == 2  # only critical+blocks_downstream

    def test_async_eval_matches_sync(self, tmp_path):
        """aevaluate returns same result as evaluate."""
        import asyncio

        _write_handoff(str(tmp_path), [
            {"severity": "critical", "message": "bad", "code": "X", "blocks_downstream": True}
        ])
        provider = DataQualityGuardrailProvider(workspace_resolver=lambda: str(tmp_path))
        request = _make_request("task", {"subagent_type": "data-analyst"})
        sync_result = provider.evaluate(request)
        async_result = asyncio.run(provider.aevaluate(request))
        assert sync_result.allow == async_result.allow
        assert sync_result.policy_id == async_result.policy_id
