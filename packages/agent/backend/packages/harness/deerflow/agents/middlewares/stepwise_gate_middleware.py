"""StepwiseGateMiddleware — pause graph after each subagent completion in
manual / data-flywheel mode.

## Why this exists

When ``workflow_mode == "manual"`` (the "data flywheel" / dogfood mode in
the UI), the user wants every subagent's output to be a checkpoint:

1. Lead dispatches a ``task(...)`` subagent.
2. Subagent runs, writes its handoff, returns a ``ToolMessage`` to lead.
3. Lead writes a narrative reply explaining what just happened.
4. **At this point the graph MUST pause** so the user can:
   - Read the narrative + see the subagent's artifacts
   - Click feedback buttons (correct / needs_fix / wrong) — data flywheel
   - Send the next message to continue, OR change direction

Without a pause, the lead silently chains ``code-executor → data-analyst →
chart-maker → report-writer`` in one shot. The user never sees individual
metric numbers, never has a chance to course-correct, and never produces
the per-step feedback the training-data flywheel depends on.

## How it works

A pre-tool-call middleware (``wrap_tool_call``) that, in manual mode,
intercepts ``task(...)`` calls whenever the most recent message-history
boundary going backward is a previous ``task`` ``ToolMessage`` (rather
than a ``HumanMessage``):

- HumanMessage first → this is a brand-new user turn, the first task in
  the turn is allowed through.
- task ToolMessage first → a subagent JUST returned and lead is trying
  to immediately fan out to the next one. The middleware returns
  ``Command(goto=END)`` to pause the LangGraph run, exactly the way
  ``ClarificationMiddleware`` does for ``ask_clarification``. The user
  in the UI sees the lead's narrative reply and the feedback affordance,
  then sends a fresh message to resume.

Disabled by default — only attached to the agent in lead_agent.py when
``workflow_mode == "manual"``.

## Why this is the right layer

- Uses LangGraph's native ``Command(goto=END)`` mechanism, the same way
  ``ClarificationMiddleware`` and ``Ev19TemplateGuardrailProvider`` already
  use it. Zero new infrastructure.
- Pre-tool-call hook sees the full state, so the rule reasons about
  message history (not just the single AIMessage being produced).
- Off in auto mode → no impact on the existing full-auto pipeline.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.constants import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


_GATE_TOOL_MESSAGE_CONTENT = (
    "[gate] subagent 已完成,在数据飞轮模式下流水线已暂停。"
    "请审阅 lead 的汇报、查看产物文件,通过反馈按钮留评价(correct / needs_fix / wrong);"
    "完成审阅后,请发送下一条消息让流程继续或调整方向。"
)


class StepwiseGateMiddleware(AgentMiddleware[AgentState]):
    """Pause the agent run after every subagent completion in manual mode.

    Attaches only when ``workflow_mode == "manual"`` (see lead_agent.py).
    In any other mode the middleware is not in the chain at all, so there
    is zero runtime cost.
    """

    def __init__(self, *, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def _should_pause(self, request: ToolCallRequest) -> bool:
        """Decide whether to intercept this ``task`` call with a pause.

        Walk backward through ``messages[:-1]`` (anything except the
        AIMessage that is right now invoking task). The first boundary
        we hit decides:

        - ``HumanMessage`` → fresh user turn; the first task() of the
          turn is fine, do NOT pause.
        - ``ToolMessage(name="task")`` → a prior subagent just returned;
          lead is trying to immediately dispatch the next subagent — pause.

        No boundary found → no prior task in this thread at all (first
        ever dispatch); do NOT pause.
        """
        if not self.enabled:
            return False
        if request.tool_call.get("name") != "task":
            return False

        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return False

        for prior in reversed(messages[:-1]):
            if isinstance(prior, HumanMessage):
                return False
            if isinstance(prior, ToolMessage) and prior.name == "task":
                return True
        return False

    def _build_pause_command(self, request: ToolCallRequest) -> Command:
        tool_call_id = request.tool_call.get("id", "")
        gate_msg = ToolMessage(
            content=_GATE_TOOL_MESSAGE_CONTENT,
            tool_call_id=tool_call_id,
            name="task",
            status="success",
        )
        logger.info(
            "StepwiseGate paused graph after subagent completion (subagent_type=%s, tool_call_id=%s)",
            (request.tool_call.get("args") or {}).get("subagent_type", "?"),
            tool_call_id,
        )
        return Command(
            update={"messages": [gate_msg]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if self._should_pause(request):
            return self._build_pause_command(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if self._should_pause(request):
            return self._build_pause_command(request)
        return await handler(request)
