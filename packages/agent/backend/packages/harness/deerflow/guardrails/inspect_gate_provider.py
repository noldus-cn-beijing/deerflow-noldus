"""InspectGateGuardrailProvider — guardrail that blocks ask_clarification when identify_ev19_template hasn't been called.

This is the safety-net (layer 3b). ParadigmIdentificationGateMiddleware (layer 3a)
fires first in after_model; this guardrail catches any agent that manages to bypass
the middleware and directly calls ask_clarification without first calling identify.

Pattern follows ev19_template_provider.py: Bridge middleware sets a ContextVar from
request.state, and the provider reads it during evaluate().

Key behaviors:
  - Only intercepts ask_clarification calls
  - Only blocks when uploaded_files are present AND no identify ToolMessage in history
  - Deny message contains clear directive (per feedback_deny_messages_must_direct)
  - fail-open: state/messages unavailable → allow
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)

_IDENTIFY_TOOL = "identify_ev19_template"
_ASK_CLARIFICATION_TOOL = "ask_clarification"

# ContextVar bridge — set by InspectGateBridgeMiddleware before GuardrailMiddleware evaluates
_inspect_gate_messages: ContextVar[list | None] = ContextVar("inspect_gate_messages", default=None)
_inspect_gate_uploaded_files: ContextVar[list | None] = ContextVar("inspect_gate_uploaded_files", default=None)


def set_inspect_gate_context(messages: list | None, uploaded_files: list | None) -> None:
    """Set the context for the current tool call evaluation."""
    _inspect_gate_messages.set(messages)
    _inspect_gate_uploaded_files.set(uploaded_files)


def _has_identify_tool_message(messages: list) -> bool:
    """Check if there's a ToolMessage from identify_ev19_template in messages."""
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == _IDENTIFY_TOOL:
            return True
    return False


class InspectGateGuardrailProvider:
    """Block ask_clarification when identify_ev19_template hasn't been called with uploaded data.

    Agent sees the deny reason and is expected to call identify_ev19_template before retrying.
    """

    name = "inspect-gate-guardrail"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only intercept ask_clarification
        if request.tool_name != _ASK_CLARIFICATION_TOOL:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        messages = _inspect_gate_messages.get()
        uploaded_files = _inspect_gate_uploaded_files.get()

        # fail-open: no context available
        if messages is None or uploaded_files is None:
            logger.debug("InspectGateGuardrailProvider: context unavailable, allowing ask_clarification")
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # No uploaded files → not our concern
        if not uploaded_files:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # identify was already called → allow
        if _has_identify_tool_message(messages):
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Has uploaded data + no identify call → DENY
        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="ethoinsight.identify_required_before_clarification",
                    message=(
                        "检测到你要反问,但本轮有上传数据且尚未真实调用 identify_ev19_template "
                        "(messages 中无其 ToolMessage)。请先调用 identify_ev19_template(uploaded_files, "
                        "user_message),用工具返回的真实 status/clarification_question 再决定反问。"
                        "不要基于推测的模板候选反问。"
                    ),
                )
            ],
            policy_id="inspect-gate-guardrail",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class InspectGateBridgeMiddleware(AgentMiddleware[AgentState]):
    """Sets the inspect gate contextvars from thread state before GuardrailMiddleware runs.

    Must be placed BEFORE GuardrailMiddleware(InspectGateGuardrailProvider) in the chain.
    State is accessed via request.state (set by ThreadDataMiddleware).
    """

    def __init__(self) -> None:
        super().__init__()

    def _extract_and_set_context(self, request: ToolCallRequest) -> None:
        """Extract messages + uploaded_files from request.state and set contextvars."""
        state = request.state
        if state is None:
            return
        messages = state.get("messages") if isinstance(state, dict) else None
        uploaded_files = state.get("uploaded_files") if isinstance(state, dict) else None
        set_inspect_gate_context(messages, uploaded_files)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._extract_and_set_context(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_and_set_context(request)
        return await handler(request)
