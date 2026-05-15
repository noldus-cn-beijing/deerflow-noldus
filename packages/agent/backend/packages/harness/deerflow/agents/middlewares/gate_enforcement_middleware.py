"""Middleware that enforces Gate 1 and Gate 2 before allowing task() calls.

Only active in workflow_mode="manual". Reads experiment-context.json and
handoff_code_executor.json from host-side workspace path (resolved from
state.thread_data.workspace_path).

Gate 1 (paradigm confirmation): blocks task() when experiment-context.json is
missing or lacks a valid paradigm field.

Gate 2 (data quality): blocks task(data-analyst) when handoff_code_executor.json
has critical data_quality_warnings that haven't been acknowledged.

Design: mirrors ToolErrorHandlingMiddleware — wraps tool calls at the
middleware layer. Only blocks task() — all other tools pass through.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.experiment_context import (
    context_exists,
    get_critical_warnings,
    is_quality_acknowledged,
    resolve_workspace_from_state,
)

logger = logging.getLogger(__name__)


class GateEnforcementMiddlewareState(AgentState):
    pass


class GateEnforcementMiddleware(AgentMiddleware[GateEnforcementMiddlewareState]):
    """Intercept task() when gate conditions aren't met in manual mode.

    Only active when `enabled` is True (set in agent.py based on workflow_mode).
    Paths are resolved from state.thread_data.workspace_path (host-side).
    Only blocks task() — all other tools pass through unconditionally.
    """

    state_schema = GateEnforcementMiddlewareState

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def _check_gate1(self, state: dict) -> bool:
        """Check whether task() should be blocked (context.json missing)."""
        workspace_dir = resolve_workspace_from_state(state)
        if workspace_dir is None:
            return False
        return not context_exists(workspace_dir)

    def _check_gate2(self, state: dict) -> bool:
        """Check whether task(data-analyst) should be blocked (critical warnings unacknowledged)."""
        workspace_dir = resolve_workspace_from_state(state)
        if workspace_dir is None:
            return False
        critical_warnings = get_critical_warnings(workspace_dir)
        if not critical_warnings:
            return False
        return not is_quality_acknowledged(workspace_dir)

    def _build_block_message(self, request: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=(
                "实验范式尚未确认。请执行以下步骤：\n"
                "1. 如果用户已经明确提到了大类名和具体范式名（如\"斑马鱼鱼群行为\"\"斑马鱼\"）：\n"
                "   直接调用 set_experiment_paradigm(...)，然后重新调用 task()。\n"
                "2. 如果用户只提到了大类但未指定细分（如只说\"焦虑迷宫\"没说具体是 EPM 还是零迷宫）：\n"
                "   调用 ask_clarification 只问细分范式那一级。\n"
                "3. 如果用户什么都没提供：\n"
                "   调用 ask_clarification 分两步：先问大类，再问细分。\n\n"
                "范式分类表见 system prompt 中的\"识别实验范式与实验设计类型\"章节。"
            ),
            tool_call_id=request.tool_call.get("id", ""),
            name="gate_enforcement",
        )

    def _build_quality_block_message(self, warnings: list[dict], request: ToolCallRequest) -> ToolMessage:
        warning_lines = "\n".join(
            f"- [{w.get('severity', 'warning')}] {w.get('message', str(w))}"
            for w in warnings
        )
        return ToolMessage(
            content=(
                f"数据质量检查发现以下 critical 问题，必须先获得用户确认才能继续：\n\n"
                f"{warning_lines}\n\n"
                f"请调用 ask_clarification 告知用户这些问题，提供以下选项：\n"
                f"(a) 排除异常个体并重算 (b) 保留并继续 (c) 查看详情\n\n"
                f"用户确认后，调用 write_file 更新 experiment-context.json，"
                f"在 gate_completed 中添加 'gate2_quality_acknowledged'，然后重新调用 task(data-analyst)。"
            ),
            tool_call_id=request.tool_call.get("id", ""),
            name="gate_enforcement",
        )

    def _get_subagent_type(self, request: ToolCallRequest) -> str | None:
        """Extract subagent_type from task() tool call arguments."""
        args = request.tool_call.get("args", {})
        if isinstance(args, dict):
            return args.get("subagent_type")
        return None

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self.enabled:
            return handler(request)

        tool_name = request.tool_call.get("name", "")

        if tool_name != "task":
            return handler(request)

        subagent_type = self._get_subagent_type(request)
        state = request.state
        workspace_dir = resolve_workspace_from_state(state) or ""

        if subagent_type == "data-analyst":
            if self._check_gate2(state):
                critical_warnings = get_critical_warnings(workspace_dir)
                logger.info(
                    "gate_check | gate=gate2_quality | thread=%s | result=blocked | detail=%d critical warning(s), not acknowledged",
                    state.get("thread_id", "unknown"),
                    len(critical_warnings),
                )
                return self._build_quality_block_message(critical_warnings, request)
            logger.info(
                "gate_check | gate=gate2_quality | thread=%s | result=allowed | detail=%s",
                state.get("thread_id", "unknown"),
                "acknowledged" if is_quality_acknowledged(workspace_dir) else "no critical warnings",
            )
        else:
            if self._check_gate1(state):
                logger.info(
                    "gate_check | gate=gate1_paradigm | thread=%s | result=blocked | detail=experiment-context.json not found",
                    state.get("thread_id", "unknown"),
                )
                return self._build_block_message(request)
            logger.info(
                "gate_check | gate=gate1_paradigm | thread=%s | result=allowed | detail=%s",
                state.get("thread_id", "unknown"),
                "context exists" if workspace_dir and context_exists(workspace_dir) else "workspace_path missing (fail-open)",
            )

        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if not self.enabled:
            return await handler(request)

        tool_name = request.tool_call.get("name", "")

        if tool_name != "task":
            return await handler(request)

        subagent_type = self._get_subagent_type(request)
        state = request.state
        workspace_dir = resolve_workspace_from_state(state) or ""

        if subagent_type == "data-analyst":
            if self._check_gate2(state):
                critical_warnings = get_critical_warnings(workspace_dir)
                logger.info(
                    "gate_check | gate=gate2_quality | thread=%s | result=blocked | detail=%d critical warning(s), not acknowledged",
                    state.get("thread_id", "unknown"),
                    len(critical_warnings),
                )
                return self._build_quality_block_message(critical_warnings, request)
            logger.info(
                "gate_check | gate=gate2_quality | thread=%s | result=allowed | detail=%s",
                state.get("thread_id", "unknown"),
                "acknowledged" if is_quality_acknowledged(workspace_dir) else "no critical warnings",
            )
        else:
            if self._check_gate1(state):
                logger.info(
                    "gate_check | gate=gate1_paradigm | thread=%s | result=blocked | detail=experiment-context.json not found",
                    state.get("thread_id", "unknown"),
                )
                return self._build_block_message(request)
            logger.info(
                "gate_check | gate=gate1_paradigm | thread=%s | result=allowed | detail=%s",
                state.get("thread_id", "unknown"),
                "context exists" if workspace_dir and context_exists(workspace_dir) else "workspace_path missing (fail-open)",
            )

        return await handler(request)
