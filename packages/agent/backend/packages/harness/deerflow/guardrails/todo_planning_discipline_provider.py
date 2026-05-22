"""Guardrail provider that rate-limits write_todos after initial planning.

Prevents the lead agent from re-writing its entire todo list every turn,
while allowing legitimate status transitions and content changes.

Uses a contextvar bridge pattern (same as Ev19TemplateGuardrailProvider)
to access thread_id from the middleware chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextvars import ContextVar
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import GuardrailDecision, GuardrailProvider, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)

_todo_discipline_thread_id: ContextVar[str | None] = ContextVar("todo_discipline_thread_id", default=None)

_PLANNING_BUDGET = 2  # first N write_todos calls always allowed


def _compute_signature(args: dict) -> str:
    """Compute a content-based signature from todo items, excluding status.

    This allows status transitions (pending→in_progress→completed) to pass
    through via the status-diff check, while catching pure re-writes where
    nothing changed at all.
    """
    todos = args.get("todos", [])
    if not isinstance(todos, list):
        return ""
    signatures = []
    for item in todos:
        if isinstance(item, dict):
            content = item.get("content", "")
            active_form = item.get("activeForm", "")
            signatures.append(f"{content}|{active_form}")
    return hashlib.md5(json.dumps(signatures, sort_keys=True, default=str).encode()).hexdigest()


def _compute_status_set(args: dict) -> frozenset:
    """Compute a frozenset of (content, status) pairs for status-diff detection."""
    todos = args.get("todos", [])
    if not isinstance(todos, list):
        return frozenset()
    return frozenset(
        (item.get("content", ""), item.get("status", "pending"))
        for item in todos
        if isinstance(item, dict)
    )


class TodoPlanningDisciplineProvider:
    """Rate-limits write_todos after the initial planning phase.

    After the planning budget is exhausted:
    - Content changed (items added/removed/renamed) → allow
    - Content same, status changed → allow (legitimate status transition)
    - Content same, status same → deny (pure re-write, no change)
    - reason parameter present → always allow (explicit intent)
    """

    name = "todo-planning-discipline"

    def __init__(self, planning_budget: int = _PLANNING_BUDGET):
        self.planning_budget = planning_budget
        self._call_counts: dict[str, int] = {}
        self._last_signatures: dict[str, str] = {}
        self._last_status_sets: dict[str, frozenset] = {}

    def _get_thread_id(self) -> str:
        tid = _todo_discipline_thread_id.get()
        return tid or "default"

    def _deny_decision(self) -> GuardrailDecision:
        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="todo.discipline",
                    message=(
                        "todo 列表已反映当前状态，无需重写。继续执行下一个任务即可。"
                        "若确有合法状态变化或新增条目，请在 reason 参数中说明（下次调用会放行）。"
                    ),
                )
            ],
            policy_id="todo-planning-discipline",
        )

    def _evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name != "write_todos":
            return GuardrailDecision(allow=True, reasons=[])

        tid = self._get_thread_id()
        count = self._call_counts.get(tid, 0) + 1
        self._call_counts[tid] = count

        # Always allow during planning budget
        if count <= self.planning_budget:
            # Record signature baseline from the first call
            if count == 1:
                self._last_signatures[tid] = _compute_signature(request.tool_input)
                self._last_status_sets[tid] = _compute_status_set(request.tool_input)
            return GuardrailDecision(allow=True, reasons=[])

        # reason parameter → always allow (agent explicitly declares intent)
        if request.tool_input.get("reason"):
            self._last_signatures[tid] = _compute_signature(request.tool_input)
            self._last_status_sets[tid] = _compute_status_set(request.tool_input)
            return GuardrailDecision(allow=True, reasons=[])

        new_signature = _compute_signature(request.tool_input)
        new_status_set = _compute_status_set(request.tool_input)
        old_signature = self._last_signatures.get(tid, "")
        old_status_set = self._last_status_sets.get(tid, frozenset())

        # Update stored state
        self._last_signatures[tid] = new_signature
        self._last_status_sets[tid] = new_status_set

        # Content changed (items added/removed/renamed) → allow
        if new_signature != old_signature:
            return GuardrailDecision(allow=True, reasons=[])

        # Content same, status changed → allow (legitimate status transition)
        if new_status_set != old_status_set:
            return GuardrailDecision(allow=True, reasons=[])

        # Content same, status same → pure re-write, deny
        return self._deny_decision()

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self._evaluate(request)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self._evaluate(request)


class TodoDisciplineBridgeMiddleware(AgentMiddleware[AgentState]):
    """Sets the _todo_discipline_thread_id contextvar from request.state.

    Must be placed BEFORE GuardrailMiddleware(TodoPlanningDisciplineProvider)
    in the middleware chain. Follows the same pattern as Ev19WorkspaceBridgeMiddleware.
    """

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        state = request.state
        if state is not None and isinstance(state, dict):
            thread_id = state.get("thread_id")
            if thread_id:
                _todo_discipline_thread_id.set(str(thread_id))
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        state = request.state
        if state is not None and isinstance(state, dict):
            thread_id = state.get("thread_id")
            if thread_id:
                _todo_discipline_thread_id.set(str(thread_id))
        return await handler(request)
