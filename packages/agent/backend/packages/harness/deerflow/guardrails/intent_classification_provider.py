"""IntentClassificationGuardrailProvider — 强制 lead 在派遣前声明意图。

Spec §6.1: lead 在第一个非 read_file tool call 之前必须输出
`[intent] <INTENT_NAME>` 行,否则拦截并注入 reminder。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

logger = logging.getLogger(__name__)


_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)

_VALID_INTENTS = frozenset({
    "E2E_FULL", "E2E_FULL_ASKVIZ", "E2E_MIN", "CHART", "REPORT",
    "QA_FACT", "QA_KNOWLEDGE", "CLARIFY",
})

_INTENT_LINE_RE = re.compile(r"\[intent\]\s+([A-Z0-9_]+)", re.MULTILINE)


def _extract_declared_intents(messages: list | None) -> set[str]:
    if not messages:
        return set()
    declared: set[str] = set()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if not isinstance(content, str):
            if isinstance(content, list):
                content = "\n".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                content = str(content)
        for match in _INTENT_LINE_RE.finditer(content):
            name = match.group(1)
            if name in _VALID_INTENTS:
                declared.add(name)
    return declared


class IntentClassificationGuardrailProvider:
    """Block non-read_file tool calls when no valid [intent] line has been declared.

    Agent sees the error reason and is expected to output `[intent] <INTENT>`
    before retrying. read_file is always allowed (skill reads must not be blocked).
    Empty / None messages fail-open so bootstrap and test contexts are unaffected.
    """

    name = "intent_classification"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name == "read_file":
            return GuardrailDecision(allow=True)

        messages = _lead_messages.get()
        if not messages:
            return GuardrailDecision(allow=True)

        declared = _extract_declared_intents(messages)
        if declared:
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="ethoinsight.intent_not_declared",
                message=(
                    "在派遣 subagent / 调 prep_metric_plan / "
                    "set_experiment_paradigm 等工具前,请先在 message "
                    "中输出 `[intent] <INTENT>` 行(INTENT ∈ {"
                    + ", ".join(sorted(_VALID_INTENTS)) + "})。"
                ),
            )],
            policy_id="intent_classification",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class IntentBridgeMiddleware(AgentMiddleware[AgentState]):
    """Sets the _lead_messages contextvar from thread state before GuardrailMiddleware runs.

    Must be placed BEFORE GuardrailMiddleware[IntentClassificationGuardrailProvider]
    in the middleware chain. State is accessed via request.state.
    """

    def __init__(self):
        super().__init__()

    def _extract_and_set_messages(self, request: ToolCallRequest) -> None:
        state = request.state
        if state is not None and isinstance(state, dict):
            msgs = state.get("messages")
            if isinstance(msgs, list):
                logger.debug(
                    "IntentBridgeMiddleware: setting _lead_messages (%d messages); last is AIMessage=%s, has [intent]=%s",
                    len(msgs),
                    isinstance(msgs[-1], AIMessage) if msgs else False,
                    bool(_INTENT_LINE_RE.search(msgs[-1].content if isinstance(msgs[-1], AIMessage) and isinstance(msgs[-1].content, str) else "")),
                )
                _lead_messages.set(msgs)
            else:
                logger.debug("IntentBridgeMiddleware: messages is not a list (type=%s)", type(msgs).__name__)
        else:
            logger.debug("IntentBridgeMiddleware: state is None or not a dict")

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._extract_and_set_messages(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_and_set_messages(request)
        return await handler(request)
