"""Guardrail provider that blocks task(code-executor) when ev19_template is unset.

Works alongside the existing GateEnforcementMiddleware (which checks the `paradigm`
field). The two have orthogonal responsibilities:
  - GateEnforcementMiddleware: paradigm field present and valid
  - Ev19TemplateGuardrailProvider: ev19_template field present in 62-variant whitelist

The provider only blocks task(code-executor). Other tool calls pass through.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)

# ContextVar bridge — set by Ev19WorkspaceBridgeMiddleware before GuardrailMiddleware evaluates
_ev19_workspace: ContextVar[str | None] = ContextVar("ev19_workspace", default=None)


def set_ev19_workspace(workspace: str | None) -> None:
    """Set the workspace path for the current async context."""
    _ev19_workspace.set(workspace)


def _default_workspace_resolver() -> str | None:
    """Default workspace resolver — caller should pass a callable that returns the host workspace path."""
    return None


class Ev19TemplateGuardrailProvider:
    """Block task(code-executor) when experiment-context.json lacks ev19_template.

    Agent sees the error reason and is expected to call set_experiment_paradigm
    (with ev19_template) or ask_clarification before retrying.
    """

    name = "ev19-template-guardrail"

    def __init__(self, workspace_resolver: Callable[[], str | None] | None = None):
        if workspace_resolver is not None:
            self._resolve_workspace = workspace_resolver
        else:
            # Try contextvar first, then fall back to None
            self._resolve_workspace = lambda: _ev19_workspace.get()

    # --- core check (sync) ---

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # ── Lock check: set_experiment_paradigm cannot change an already-set ev19_template ──
        if request.tool_name == "set_experiment_paradigm":
            workspace = self._resolve_workspace()
            if workspace is not None:
                ctx = self._read_context(workspace)
                if ctx and ctx.get("ev19_template"):
                    args = request.tool_input or {}
                    if not args.get("confirm_template_change"):
                        return GuardrailDecision(
                            allow=False,
                            reasons=[
                                GuardrailReason(
                                    code="ethoinsight.template_already_set",
                                    message=(
                                        f"ev19_template 已设置为 '{ctx['ev19_template']}'，"
                                        "不允许中途修改以保持分析一致。"
                                        "如确实需要修改，请向 set_experiment_paradigm 传 confirm_template_change=True；"
                                        "或建议用户开新 thread 重新分析。"
                                    ),
                                )
                            ],
                            policy_id="ev19-template-guardrail",
                        )
            # First-time set — allow
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Only inspect task() calls
        if request.tool_name != "task":
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        # Only inspect task(code-executor) — other subagents are unaffected
        subagent = request.tool_input.get("subagent_type", "") if request.tool_input else ""
        if "code-executor" not in subagent:
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        workspace = self._resolve_workspace()
        if workspace is None:
            # No workspace context available — fail-open (don't block)
            logger.debug("Ev19TemplateGuardrailProvider: workspace unresolvable, allowing task call")
            return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

        ctx = self._read_context(workspace)
        if ctx is None or not ctx.get("ev19_template"):
            return GuardrailDecision(
                allow=False,
                reasons=[
                    GuardrailReason(
                        code="ethoinsight.no_ev19_template",
                        message=(
                            "EV19 模板尚未设置。请先调用 set_experiment_paradigm(..., ev19_template=...) "
                            "确定模板变体（参考 ethovision-paradigm-knowledge skill 中 references/_facts.md "
                            "的 62 变体白名单）。如果信息不足，先 ask_clarification 反问用户。"
                        ),
                    )
                ],
                policy_id="ev19-template-guardrail",
            )

        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    # --- async wrapper ---

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)

    # --- helpers ---

    def _read_context(self, workspace: str) -> dict | None:
        path = Path(workspace) / "experiment-context.json"
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read experiment-context.json: %s", e)
            return None


class Ev19WorkspaceBridgeMiddleware(AgentMiddleware[AgentState]):
    """Sets the _ev19_workspace contextvar from thread state before GuardrailMiddleware runs.

    Must be placed BEFORE GuardrailMiddleware in the middleware chain.
    State is accessed via request.state (set by ThreadDataMiddleware).
    """

    def __init__(self):
        super().__init__()

    def _extract_and_set_workspace(self, request: ToolCallRequest) -> None:
        """Extract workspace from request.state and set the contextvar."""
        state = request.state
        if state is not None:
            thread_data = state.get("thread_data") if isinstance(state, dict) else None
            if isinstance(thread_data, dict):
                workspace = thread_data.get("workspace_path")
                if workspace:
                    _ev19_workspace.set(str(workspace))

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._extract_and_set_workspace(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_and_set_workspace(request)
        return await handler(request)
