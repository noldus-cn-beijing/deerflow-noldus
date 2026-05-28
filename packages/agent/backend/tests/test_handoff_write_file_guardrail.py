"""Unit tests for ScriptInvocationOnlyProvider write_file handoff blocking (Sprint 0)."""

import pytest

from deerflow.guardrails.provider import GuardrailRequest
from deerflow.guardrails.script_invocation_only_provider import ScriptInvocationOnlyProvider


def _make_request(tool_name: str, tool_input: dict, agent_id: str) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input=tool_input,
        agent_id=agent_id,
        thread_id="test-thread",
        is_subagent=True,
        timestamp="2026-01-01T00:00:00Z",
    )


_provider = ScriptInvocationOnlyProvider()


class TestWriteFileHandoffBlocking:
    """4 handoff subagents cannot write_file handoff_*.json."""

    @pytest.mark.parametrize("subagent", ["code-executor", "data-analyst", "chart-maker", "report-writer"])
    def test_handoff_write_denied(self, subagent):
        req = _make_request(
            "write_file",
            {"path": "/mnt/user-data/workspace/handoff_code_executor.json", "content": "{}"},
            f"subagent:{subagent}",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is False
        assert decision.reasons[0].code == "handoff.write_file_forbidden"

    def test_non_handoff_write_allowed(self):
        req = _make_request(
            "write_file",
            {"path": "/mnt/user-data/workspace/plan_metrics.json", "content": "{}"},
            "subagent:code-executor",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is True

    def test_non_handoff_subagent_write_allowed(self):
        req = _make_request(
            "write_file",
            {"path": "/mnt/user-data/workspace/handoff_code_executor.json", "content": "{}"},
            "subagent:general-purpose",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is True

    def test_lead_agent_write_allowed(self):
        req = _make_request(
            "write_file",
            {"path": "/mnt/user-data/workspace/handoff_code_executor.json", "content": "{}"},
            "lead-agent",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is True


class TestBashGatingUnchanged:
    """Existing bash whitelisting still works."""

    def test_script_call_allowed(self):
        req = _make_request(
            "bash",
            {"command": "python -m ethoinsight.scripts.fst.compute_immobility --input x --output y"},
            "subagent:code-executor",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is True

    def test_arbitrary_bash_denied(self):
        req = _make_request(
            "bash",
            {"command": "pip install numpy"},
            "subagent:code-executor",
        )
        decision = _provider.evaluate(req)
        assert decision.allow is False
