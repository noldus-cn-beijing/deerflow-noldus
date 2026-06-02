"""Tests for InspectGateGuardrailProvider (layer 3b).

Covers:
  - Non-ask_clarification calls → always allow
  - No context (fail-open) → allow
  - No uploaded files → allow
  - Has uploads + identify ToolMessage in history → allow
  - Has uploads + no identify ToolMessage → deny with directive
  - Deny message contains clear directive
  - Bridge middleware sets contextvars correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deerflow.guardrails.inspect_gate_provider import (
    InspectGateBridgeMiddleware,
    InspectGateGuardrailProvider,
    _inspect_gate_messages,
    _inspect_gate_uploaded_files,
    set_inspect_gate_context,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_request(tool_name: str, tool_input: dict | None = None) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input=tool_input or {},
    )


class TestInspectGateGuardrailProvider:
    """Unit tests for InspectGateGuardrailProvider."""

    def setup_method(self) -> None:
        """Reset contextvars before each test."""
        set_inspect_gate_context(None, None)

    def test_non_ask_clarification_always_allowed(self) -> None:
        provider = InspectGateGuardrailProvider()
        set_inspect_gate_context([], ["file.txt"])
        req = _make_request("task", {"subagent_type": "code-executor"})
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_no_context_fail_open(self) -> None:
        provider = InspectGateGuardrailProvider()
        # Context is None (default)
        req = _make_request("ask_clarification")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_no_uploaded_files_allowed(self) -> None:
        provider = InspectGateGuardrailProvider()
        set_inspect_gate_context([], [])
        req = _make_request("ask_clarification")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_has_uploads_no_identify_denies(self) -> None:
        provider = InspectGateGuardrailProvider()
        messages = [AIMessage(content="I think this is FST")]
        set_inspect_gate_context(messages, ["fst.txt"])
        req = _make_request("ask_clarification", {"question": "FST or TST?"})
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert decision.reasons[0].code == "ethoinsight.identify_required_before_clarification"
        # Verify deny message has clear directive
        msg = decision.reasons[0].message
        assert "identify_ev19_template" in msg
        assert "请先调用" in msg

    def test_has_uploads_with_identify_allowed(self) -> None:
        provider = InspectGateGuardrailProvider()
        ai = AIMessage(content="", tool_calls=[{"name": "identify_ev19_template", "args": {}, "id": "tc1"}])
        tm = ToolMessage(content='{"status": "ok"}', tool_call_id="tc1", name="identify_ev19_template")
        messages = [ai, tm, AIMessage(content="got it")]
        set_inspect_gate_context(messages, ["fst.txt"])
        req = _make_request("ask_clarification")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_async_delegates_to_sync(self) -> None:
        import asyncio

        provider = InspectGateGuardrailProvider()
        set_inspect_gate_context([], ["fst.txt"])
        req = _make_request("ask_clarification")
        decision = asyncio.get_event_loop().run_until_complete(provider.aevaluate(req))
        assert decision.allow is False

    def test_policy_id_set(self) -> None:
        provider = InspectGateGuardrailProvider()
        set_inspect_gate_context([AIMessage(content="x")], ["fst.txt"])
        req = _make_request("ask_clarification")
        decision = provider.evaluate(req)
        assert decision.policy_id == "inspect-gate-guardrail"


class TestInspectGateBridgeMiddleware:
    """Unit tests for InspectGateBridgeMiddleware."""

    def setup_method(self) -> None:
        set_inspect_gate_context(None, None)

    def test_bridge_sets_context_from_state(self) -> None:
        bridge = InspectGateBridgeMiddleware()
        messages = [AIMessage(content="test")]
        uploaded = ["file.txt"]

        request = MagicMock()
        request.state = {"messages": messages, "uploaded_files": uploaded}
        request.tool_call = {"name": "ask_clarification", "args": {}, "id": "tc1"}

        handler_called = False

        def handler(req):
            nonlocal handler_called
            handler_called = True
            # Verify context is set when handler runs
            assert _inspect_gate_messages.get() == messages
            assert _inspect_gate_uploaded_files.get() == uploaded
            return MagicMock()

        bridge.wrap_tool_call(request, handler)
        assert handler_called

    def test_bridge_handles_none_state(self) -> None:
        bridge = InspectGateBridgeMiddleware()
        request = MagicMock()
        request.state = None
        request.tool_call = {"name": "ask_clarification", "args": {}, "id": "tc1"}

        def handler(req):
            return MagicMock()

        # Should not raise
        result = bridge.wrap_tool_call(request, handler)
        assert result is not None
