"""Middleware that enforces Gate 1 completion before allowing task() calls.

Only active in workflow_mode="manual". Reads experiment-context.json from
host-side workspace path (resolved from state.thread_data.workspace_path).

Design: mirrors ToolErrorHandlingMiddleware — wraps tool calls at the
middleware layer. Only blocks task() — all other tools pass through.
"""

import logging
from collections.abc import Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deerflow.agents.middlewares.experiment_context import (
    context_exists,
    resolve_workspace_from_state,
)

logger = logging.getLogger(__name__)


class GateEnforcementMiddlewareState(AgentState):
    pass


class GateEnforcementMiddleware(AgentMiddleware[GateEnforcementMiddlewareState]):
    """Intercept task() when Gate 1 hasn't completed in manual mode.

    Only active when `enabled` is True (set in agent.py based on workflow_mode).
    Paths are resolved from state.thread_data.workspace_path (host-side).
    Only blocks task() — all other tools pass through unconditionally.
    """

    state_schema = GateEnforcementMiddlewareState

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def _should_block(self, state: dict) -> bool:
        """Check whether task() should be blocked (context.json missing)."""
        workspace_dir = resolve_workspace_from_state(state)
        if workspace_dir is None:
            # Can't resolve workspace — allow (old thread or missing ThreadDataMiddleware)
            return False
        return not context_exists(workspace_dir)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        if not self.enabled:
            return handler(request)

        tool_name = request.tool_call.get("name", "")

        # Only block task() — all other tools pass through
        if tool_name == "task" and self._should_block(request.state):
            logger.info("GateEnforcementMiddleware: blocking task() — experiment-context.json not found")
            return ToolMessage(
                content=(
                    "请先通过 ask_clarification 确认实验类型后再开始分析。\n"
                    "调用 ask_clarification(question=\"请问您做的是哪类实验？\", "
                    "clarification_type=\"approach_choice\", "
                    "options=[...]) 让用户选择实验范式，然后调用 set_experiment_paradigm tool 记录选择。"
                ),
                tool_call_id=request.tool_call.get("id", ""),
                name="gate_enforcement",
            )

        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        return self.wrap_tool_call(request, handler)
