"""IntentPostStepAskGateProvider — 拦截跳过 ask 步骤直接派 dispatch。

A2 泛化版：从 path_registry.PATHS 数据驱动，覆盖所有 ask 步骤
(viz / report / four_choice)，而非只硬保护 E2E_FULL_ASKVIZ 的 ask(viz?)。

viz 拦截行为与改前完全一致（迁移样板，回归红线）。

堵诊断「洞 2」：8 个 ask 点原来只保护了 1 个(viz)，现在全部保护。

deny 消息必须含「请改用 X 因为 Y 然后做 Z」结构（spec §1 核心原则）。
"""

from __future__ import annotations

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
from deerflow.guardrails.path_registry import (
    ASK_GATE_MAP,
    PATHS,
    Step,
    ensure_dispatch_targets_validated,
    to_handoff_name,
)
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


# ---------------------------------------------------------------------------
# viz-specific deny message — preserved byte-for-byte for regression (spec §3.2)
# ---------------------------------------------------------------------------
_VIZ_DENY_MESSAGE = (
    "请改调 ask_clarification(question='📊 指标和解读已完成。需要我把结果可视化成图吗?', "
    "options=['A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)', "
    "'B. 不用,直接给我报告'])，因为 INTENT=E2E_FULL_ASKVIZ 要求 data-analyst 完成后 "
    "先反问用户是否需要图表；用户回答后再调 set_viz_choice(choice='yes'|'no') "
    "落盘 gate3，之后才能派 chart-maker（或跳过直接派 report-writer）。"
)


class IntentPostStepAskGateProvider:
    """Block task(X) when the path has an uncompleted ask step before X.

    Data-driven: reads PATHS to find ask steps for any intent.
    Fails open (allow) when workspace, context, or path is unavailable.

    Viz regression: the ask(viz?) gate for E2E_FULL_ASKVIZ produces
    the exact same deny message as the pre-A2 hardcoded version.
    """

    name = "intent_post_step_ask_gate"

    def __init__(self) -> None:
        ensure_dispatch_targets_validated()

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only intercept task() calls
        if request.tool_name != "task":
            return GuardrailDecision(allow=True)

        subagent_type = request.tool_input.get("subagent_type")
        if not subagent_type:
            return GuardrailDecision(allow=True)

        # Check intent from messages
        messages = _lead_messages.get()
        intent = _extract_latest_intent(messages)
        if not intent:
            return GuardrailDecision(allow=True)

        # Look up the path for this intent
        steps = PATHS.get(intent)
        if not steps:
            return GuardrailDecision(allow=True)

        # Find the position of this dispatch target in the path
        target_idx = None
        for i, step in enumerate(steps):
            if step.kind == "dispatch" and step.target == subagent_type:
                target_idx = i
                break

        if target_idx is None:
            return GuardrailDecision(allow=True)

        # Check workspace
        workspace = _lead_workspace.get()
        if not workspace:
            return GuardrailDecision(allow=True)

        # Check context for gate_completed
        ctx = read_context(workspace)
        if ctx is None:
            return GuardrailDecision(allow=True)

        gate_completed = ctx.get("gate_completed", [])
        if not isinstance(gate_completed, list):
            gate_completed = []

        # Check each ask step before the target dispatch
        for i in range(target_idx):
            step = steps[i]
            if step.kind != "ask":
                continue

            ask_key = step.target
            # "clarify" doesn't use gate — skip
            if ask_key == "clarify":
                continue

            gate_name = ASK_GATE_MAP.get(ask_key)
            if not gate_name:
                continue

            # If gate already acknowledged → this ask step is satisfied
            if gate_name in gate_completed:
                continue

            # Check if the immediately preceding dispatch step has completed.
            # Only check the nearest preceding dispatch — if it completed,
            # all earlier dispatches must have also completed (sequential path).
            # This preserves the original viz semantics where only data-analyst's
            # handoff was checked (not code-executor's).
            immediate_dispatch_done = True
            for j in range(i - 1, -1, -1):
                prev = steps[j]
                if prev.kind == "dispatch":
                    handoff_name = to_handoff_name(prev.target)
                    handoff_path = Path(workspace) / f"handoff_{handoff_name}.json"
                    immediate_dispatch_done = handoff_path.exists()
                    break

            if not immediate_dispatch_done:
                # Preceding dispatches haven't all completed — not our concern
                # (PathSequenceProvider handles that)
                continue

            # This ask step's gate is NOT acknowledged but all preceding
            # dispatches are done → the lead is trying to skip the ask step
            return self._deny_ask_step(intent, ask_key, subagent_type)

        return GuardrailDecision(allow=True)

    def _deny_ask_step(self, intent: str, ask_key: str, target: str) -> GuardrailDecision:
        """Generate deny for skipping an ask step. Viz uses the legacy message."""
        # Viz regression: exact same message as pre-A2
        if ask_key == "viz":
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="ethoinsight.viz_choice_not_acknowledged",
                    message=_VIZ_DENY_MESSAGE,
                )],
                policy_id="intent_post_step_ask_gate",
            )

        # Generic deny for other ask steps
        gate_name = ASK_GATE_MAP.get(ask_key, f"gate for {ask_key}")
        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code=f"ethoinsight.ask_gate_{ask_key}_not_acknowledged",
                message=(
                    f"按 {intent} 路径，在派 {target} 之前需要先完成 ask({ask_key}?) 步骤。"
                    f"请先反问用户并确认选择，落盘 {gate_name} 后再继续。"
                ),
            )],
            policy_id="intent_post_step_ask_gate",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class IntentPostStepAskGateBridge(AgentMiddleware[AgentState]):
    """Sets _lead_messages and _lead_workspace contextvars from thread state.

    Must be placed BEFORE GuardrailMiddleware[IntentPostStepAskGateProvider]
    in the middleware chain.
    """

    def __init__(self):
        super().__init__()

    def _extract_and_set(self, request: ToolCallRequest) -> None:
        state = request.state
        if state is None or not isinstance(state, dict):
            return
        # Set messages
        msgs = state.get("messages")
        if isinstance(msgs, list):
            _lead_messages.set(msgs)
        # Set workspace
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
        self._extract_and_set(request)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._extract_and_set(request)
        return await handler(request)
