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
    ASK_GATE_SETTER_TOOL,
    PATHS,
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


def _current_batch_tool_names(messages: list | None) -> set[str]:
    """Tool-call names in the most recent AIMessage (the current dispatch batch).

    When the lead emits parallel tool_calls in one AIMessage (e.g. set_viz_choice
    + task(chart-maker)), that AIMessage is already appended to state['messages']
    by the time each tool_call is evaluated. This returns the names of all
    tool_calls in that latest AIMessage so the gate check can recognise an
    in-flight gate-setter sibling and avoid a false deny (race fix).

    Returns an empty set when there is no AIMessage or it has no tool_calls.
    """
    if not messages:
        return set()
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return set()
        names: set[str] = set()
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name")
                if name:
                    names.add(str(name))
        return names
    return set()


# ---------------------------------------------------------------------------
# viz-specific deny message — preserved byte-for-byte for regression (spec §3.2)
# ---------------------------------------------------------------------------
_VIZ_DENY_MESSAGE = (
    "请改调 ask_clarification(question='📊 指标和解读已完成。需要我把结果可视化成图吗?', "
    "options=['A. 是,把刚才的结论画成图', "
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

        # ETHO-7 子改动 B：ask 顺序/合并强制。复用 PATHS 声明的 ask 顺序：若某
        # 后置 ask gate 已落盘，而它之前某 ask gate 未落盘（且后者前置 dispatch 已完成），
        # 说明 lead 把分开的两个 ask 点合并/乱序了 → deny。这稳定决策点顺序（viz 恒在 report 前）。
        # 注意：必须先于「按 target 跳步检测」，因为乱序本身就是更上游的违规。
        order_violation = self._detect_ask_order_violation(steps, gate_completed, workspace, intent)
        if order_violation is not None:
            return order_violation

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

            # Race fix: the lead may emit the gate-setter (e.g. set_viz_choice)
            # and this task() as parallel tool_calls in the SAME AIMessage. The
            # gate-setter's write may not have landed yet when we evaluate task().
            # If the current batch already contains the gate-setter for this ask
            # step, the gate IS being acknowledged in-flight → don't false-deny.
            setter_tool = ASK_GATE_SETTER_TOOL.get(ask_key)
            if setter_tool and setter_tool in _current_batch_tool_names(messages):
                continue

            # This ask step's gate is NOT acknowledged but all preceding
            # dispatches are done → the lead is trying to skip the ask step
            return self._deny_ask_step(intent, ask_key, subagent_type)

        return GuardrailDecision(allow=True)

    def _detect_ask_order_violation(
        self,
        steps: list,
        gate_completed: list[str],
        workspace: str,
        intent: str,
    ) -> GuardrailDecision | None:
        """ETHO-7 子改动 B：检测 ask 步骤乱序/合并。

        遍历 PATHS 声明的 ask 顺序，若发现「后置 ask gate 已落盘」而「它之前的某个
        ask gate 未落盘、且后者前置 dispatch 已完成」→ 乱序违规，deny。

        守边界（§2.3）：用户一次性回答多个问题（前后 ask gate 都已落盘）不算违规 ——
        只在「后置已落盘、前置未落盘」这种真乱序时触发。

        复用 ASK_GATE_MAP + 同款「最近前置 dispatch handoff 是否存在」检查，不新增机制。
        """
        # 收集本路径所有 ask 步骤的 (idx, ask_key, gate_name)
        ask_steps: list[tuple[int, str, str]] = []
        for i, step in enumerate(steps):
            if step.kind != "ask":
                continue
            if step.target == "clarify":
                continue
            gate_name = ASK_GATE_MAP.get(step.target)
            if gate_name:
                ask_steps.append((i, step.target, gate_name))

        for later_pos in range(len(ask_steps)):
            later_idx, later_key, later_gate = ask_steps[later_pos]
            if later_gate not in gate_completed:
                continue  # 后置 ask 还没落盘，不构成乱序

            # 后置已落盘 → 它之前的所有 ask gate 都应该已落盘
            for earlier_pos in range(later_pos):
                earlier_idx, earlier_key, earlier_gate = ask_steps[earlier_pos]
                if earlier_gate in gate_completed:
                    continue  # 前置也已落盘，正常

                # 前置 ask 未落盘。只有当它的前置 dispatch 已完成时才判定违规
                # （前置 dispatch 没完成时，该 ask 点还没激活，谈不上乱序）。
                if not self._preceding_dispatch_done(steps, earlier_idx, workspace):
                    continue

                # 同批 race：若当前 AIMessage 已含前置 ask 的 gate-setter，
                # 视为 in-flight 落盘，不判乱序（与跳步检测的 race fix 对齐）。
                setter_tool = ASK_GATE_SETTER_TOOL.get(earlier_key)
                if setter_tool and setter_tool in _current_batch_tool_names(_lead_messages.get()):
                    continue

                return GuardrailDecision(
                    allow=False,
                    reasons=[GuardrailReason(
                        code="ethoinsight.ask_order_violation",
                        message=(
                            f"按 {intent} 路径声明的 ask 顺序，ask({later_key}?) 在 ask({earlier_key}?) 之后，"
                            f"但 {later_gate} 已落盘而 {earlier_gate} 未落盘 —— "
                            f"不能跳过前置 ask({earlier_key}?) 先完成后置 ask({later_key}?)。"
                            f"请先把前置 ask({earlier_key}?) 反问用户并落盘 {earlier_gate}，"
                            f"再继续后续步骤（viz 恒在 report 前，不可合并/乱序）。"
                        ),
                    )],
                    policy_id="intent_post_step_ask_gate",
                )
        return None

    @staticmethod
    def _preceding_dispatch_done(steps: list, ask_idx: int, workspace: str) -> bool:
        """ask_idx 这一步的最近前置 dispatch 的 handoff 是否已落盘。"""
        for j in range(ask_idx - 1, -1, -1):
            prev = steps[j]
            if prev.kind == "dispatch":
                handoff_name = to_handoff_name(prev.target)
                return (Path(workspace) / f"handoff_{handoff_name}.json").exists()
        return True  # 没有前置 dispatch（ask 是路径首步）→ 视为已就绪

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
