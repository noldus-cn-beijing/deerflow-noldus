"""Tests for PathSequenceProvider — dispatch order enforcement (堵洞1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from deerflow.guardrails.path_sequence_provider import (
    PathSequenceBridge,
    PathSequenceProvider,
    _lead_messages,
    _lead_workspace,
)
from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture()
def provider():
    return PathSequenceProvider()


@pytest.fixture()
def reset_contextvars():
    """Reset contextvars before each test."""
    _lead_messages.set(None)
    _lead_workspace.set(None)
    yield
    _lead_messages.set(None)
    _lead_workspace.set(None)


def _make_request(tool_name: str, subagent_type: str | None = None) -> GuardrailRequest:
    tool_input = {}
    if subagent_type:
        tool_input["subagent_type"] = subagent_type
    return GuardrailRequest(tool_name=tool_name, tool_input=tool_input)


def _set_intent(intent: str) -> None:
    """Set _lead_messages with a real AIMessage containing [intent]."""
    msg = AIMessage(content=f"[intent] {intent}")
    _lead_messages.set([msg])


def _set_workspace(tmp_path: Path) -> None:
    _lead_workspace.set(str(tmp_path))


def _create_handoff(workspace: Path, subagent_name: str) -> None:
    """Create a handoff file for a subagent in the workspace."""
    handoff_file = workspace / f"handoff_{subagent_name}.json"
    handoff_file.write_text(json.dumps({"status": "completed"}))


class TestPathSequenceProviderBasic:
    """Basic allow/deny logic."""

    def test_non_task_tool_always_allowed(self, provider, reset_contextvars):
        """Non-task tools are never intercepted."""
        req = _make_request("read_file")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_no_intent_declared_allows(self, provider, reset_contextvars):
        """No [intent] line → fail open."""
        _lead_messages.set(None)
        req = _make_request("task", "chart-maker")
        assert provider.evaluate(req).allow is True

    def test_unknown_intent_allows(self, provider, reset_contextvars):
        """Unknown intent → fail open."""
        _set_intent("UNKNOWN_INTENT")
        req = _make_request("task", "chart-maker")
        assert provider.evaluate(req).allow is True

    def test_subagent_not_in_path_allows(self, provider, reset_contextvars):
        """Subagent not in this intent's path → allow."""
        _set_intent("CHART")  # CHART only has chart-maker
        req = _make_request("task", "knowledge-assistant")
        assert provider.evaluate(req).allow is True

    def test_no_workspace_allows(self, provider, reset_contextvars):
        """No workspace → fail open."""
        _set_intent("E2E_FULL")
        req = _make_request("task", "chart-maker")
        assert provider.evaluate(req).allow is True


class TestPathSequenceEnforcement:
    """Verify dispatch order is enforced correctly."""

    def test_code_executor_always_allowed_first(self, provider, reset_contextvars, tmp_path):
        """code-executor is always the first dispatch step → always allowed."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        req = _make_request("task", "code-executor")
        assert provider.evaluate(req).allow is True

    def test_data_analyst_without_code_executor_denied(self, provider, reset_contextvars, tmp_path):
        """Skipping code-executor to dispatch data-analyst → denied."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        req = _make_request("task", "data-analyst")
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert "code-executor" in decision.reasons[0].message

    def test_data_analyst_with_code_executor_allowed(self, provider, reset_contextvars, tmp_path):
        """After code-executor completes → data-analyst allowed."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        _create_handoff(tmp_path, "code_executor")
        req = _make_request("task", "data-analyst")
        assert provider.evaluate(req).allow is True

    def test_chart_maker_without_predecessors_denied(self, provider, reset_contextvars, tmp_path):
        """Skipping to chart-maker without code-executor and data-analyst → denied."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False
        # Should mention both missing predecessors
        msg = decision.reasons[0].message
        assert "code-executor" in msg
        assert "data-analyst" in msg

    def test_chart_maker_with_all_predecessors_allowed(self, provider, reset_contextvars, tmp_path):
        """After code-executor and data-analyst → chart-maker allowed."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        _create_handoff(tmp_path, "code_executor")
        _create_handoff(tmp_path, "data_analyst")
        req = _make_request("task", "chart-maker")
        assert provider.evaluate(req).allow is True

    def test_chart_intent_allows_chart_maker_directly(self, provider, reset_contextvars, tmp_path):
        """CHART intent: chart-maker has no predecessors → always allowed."""
        _set_intent("CHART")
        _set_workspace(tmp_path)
        req = _make_request("task", "chart-maker")
        assert provider.evaluate(req).allow is True

    def test_report_intent_allows_report_writer_directly(self, provider, reset_contextvars, tmp_path):
        """REPORT intent: report-writer has no predecessors → always allowed."""
        _set_intent("REPORT")
        _set_workspace(tmp_path)
        req = _make_request("task", "report-writer")
        assert provider.evaluate(req).allow is True

    def test_qa_fact_allows_knowledge_assistant_directly(self, provider, reset_contextvars, tmp_path):
        """QA_FACT intent: knowledge-assistant has no predecessors → always allowed."""
        _set_intent("QA_FACT")
        _set_workspace(tmp_path)
        req = _make_request("task", "knowledge-assistant")
        assert provider.evaluate(req).allow is True


class TestDenyMessageFormat:
    """Verify deny messages follow the '请改用 X 因为 Y 然后做 Z' structure."""

    def test_deny_contains_clear_instruction(self, provider, reset_contextvars, tmp_path):
        """Deny message must contain intent name and missing predecessor."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False
        msg = decision.reasons[0].message
        assert "E2E_FULL" in msg
        assert "chart-maker" in msg
        assert "请先" in msg
        # Must specify the immediate predecessor to call
        assert "task(" in msg


class TestAskStepNotBlocked:
    """Ask steps (ask_clarification) are not dispatch steps — should not be blocked."""

    def test_clarify_intent_no_dispatch_blocking(self, provider, reset_contextvars, tmp_path):
        """CLARIFY intent only has an ask step, no dispatch → task() never blocked."""
        _set_intent("CLARIFY")
        _set_workspace(tmp_path)
        req = _make_request("task", "any-subagent")
        assert provider.evaluate(req).allow is True


class TestPathSequenceBridge:
    """Test the bridge middleware sets contextvars correctly."""

    def test_bridge_sets_messages_and_workspace(self, tmp_path):
        bridge = PathSequenceBridge()

        # Create a mock ToolCallRequest
        from unittest.mock import MagicMock

        mock_request = MagicMock()
        mock_request.state = {
            "messages": [AIMessage(content="[intent] E2E_FULL")],
            "thread_data": {"workspace_path": str(tmp_path)},
        }

        # Reset contextvars
        _lead_messages.set(None)
        _lead_workspace.set(None)

        # Call the bridge
        handler_called = False

        def handler(req):
            nonlocal handler_called
            handler_called = True
            # Verify contextvars are set
            assert _lead_messages.get() is not None
            assert _lead_workspace.get() == str(tmp_path)
            return MagicMock()

        bridge.wrap_tool_call(mock_request, handler)
        assert handler_called
