"""IntentPostStepAskGateProvider — 拦截跳过 ask(viz?) 直接派 chart-maker。

Spec §2.3: 当 INTENT=E2E_FULL_ASKVIZ 且 data-analyst 已完成但
gate3_viz_acknowledged 未落盘时，拦截 task(chart-maker) 并注入强引导 deny。

deny 消息必须含「请改用 X 因为 Y 然后做 Z」结构（spec §1 核心原则）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.experiment_context import read_context
from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

logger = __import__("logging").getLogger(__name__)

# Reuse IntentBridgeMiddleware's ContextVar for messages
_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)

# Bridge middleware sets this from thread_data before GuardrailMiddleware runs
_lead_workspace: ContextVar[str | None] = ContextVar("_lead_workspace", default=None)

_INTENT_LINE_RE = re.compile(r"\[intent\]\s+([A-Z0-9_]+)", re.MULTILINE)


def _extract_latest_intent(messages: list | None) -> str | None:
    """Extract the most recent declared intent from AIMessage content."""
    if not messages:
        return None
    for msg in reversed(messages):
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
            return match.group(1)
    return None


class IntentPostStepAskGateProvider:
    """Block task(chart-maker) when ASKVIZ flow hasn't asked viz question yet.

    Conditions for blocking:
    1. tool = task(subagent_type="chart-maker")
    2. latest intent is E2E_FULL_ASKVIZ
    3. handoff_data_analyst.json exists (data-analyst has completed)
    4. gate3_viz_acknowledged is NOT in experiment-context.json gate_completed

    Fails open (allow) when workspace or context is unavailable.
    """

    name = "intent_post_step_ask_gate"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only intercept task(chart-maker)
        if request.tool_name != "task":
            return GuardrailDecision(allow=True)
        if request.tool_input.get("subagent_type") != "chart-maker":
            return GuardrailDecision(allow=True)

        # Check intent from messages
        messages = _lead_messages.get()
        intent = _extract_latest_intent(messages)
        if intent != "E2E_FULL_ASKVIZ":
            return GuardrailDecision(allow=True)

        # Check workspace
        workspace = _lead_workspace.get()
        if not workspace:
            return GuardrailDecision(allow=True)

        # Check gate3 already acknowledged
        ctx = read_context(workspace)
        if ctx is None:
            return GuardrailDecision(allow=True)

        gate_completed = ctx.get("gate_completed", [])
        if not isinstance(gate_completed, list):
            gate_completed = []
        if "gate3_viz_acknowledged" in gate_completed:
            return GuardrailDecision(allow=True)

        # Only block when data-analyst has actually completed
        handoff_path = Path(workspace) / "handoff_data_analyst.json"
        if not handoff_path.exists():
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="ethoinsight.viz_choice_not_acknowledged",
                message=(
                    "请改调 ask_clarification(question='📊 指标和解读已完成。需要我把结果可视化成图吗?', "
                    "options=['A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)', "
                    "'B. 不用,直接给我报告'])，因为 INTENT=E2E_FULL_ASKVIZ 要求 data-analyst 完成后 "
                    "先反问用户是否需要图表；用户回答后再调 set_viz_choice(choice='yes'|'no') "
                    "落盘 gate3，之后才能派 chart-maker（或跳过直接派 report-writer）。"
                ),
            )],
            policy_id="intent_post_step_ask_gate",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class IntentPostStepAskGateBridge(AgentMiddleware[AgentState]):
    """Sets _lead_workspace contextvar from thread state before GuardrailMiddleware.

    Must be placed BEFORE GuardrailMiddleware[IntentPostStepAskGateProvider]
    in the middleware chain.
    """

    def __init__(self):
        super().__init__()

    def _extract_workspace(self, request: ToolCallRequest) -> None:
        state = request.state
        if state is not None and isinstance(state, dict):
            thread_data = state.get("thread_data")
            if isinstance(thread_data, dict):
                wp = thread_data.get("workspace_path")
                if wp is not None:
                    _lead_workspace.set(wp)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._extract_workspace(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_workspace(request)
        return await handler(request)
