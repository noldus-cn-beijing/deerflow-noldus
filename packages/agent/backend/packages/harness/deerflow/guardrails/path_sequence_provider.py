"""PathSequenceProvider — 校验 task() 派遣顺序是否符合 path_registry.PATHS。

堵诊断「洞 1」：跳过 data-analyst 直接 task(chart-maker) 无人拦。

逻辑：
1. 只拦 task 工具
2. 从 messages 提取 latest [intent]
3. 查 PATHS[intent]，找出本次 task(subagent_type=X) 中 X 的位置
4. 校验 X 之前的所有 dispatch step 对应的 handoff 都已落盘
5. 若有前序 dispatch step 的 handoff 缺失 → deny，含明确指令
6. Fail-open：workspace/context 不可用时 allow

与 TaskHandoffAuthorizationProvider 的区别（必须说清）：
- 后者校验"prompt 里有没有写 {{handoff://X}} 占位符"（依赖声明）
- PathSequenceProvider 校验"路径中 X 的前序 dispatch 是否真完成了"（顺序事实）
- 两者正交
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

from deerflow.guardrails.path_registry import (
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

# Reuse the same ContextVar names as intent_post_step_ask_gate_provider.
# Bridge middleware sets these before GuardrailMiddleware runs.
_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)
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


class PathSequenceProvider:
    """Block task(X) when preceding dispatch steps in the path haven't completed.

    Reads PATHS from path_registry to determine correct order.
    Fails open (allow) when workspace, intent, or path is unavailable.
    """

    name = "path_sequence"

    def __init__(self) -> None:
        # Validate dispatch targets on first instantiation
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

        # Target not found in this path's dispatch steps — allow
        # (could be a subagent not in this path, or ask step)
        if target_idx is None:
            return GuardrailDecision(allow=True)

        # Check workspace
        workspace = _lead_workspace.get()
        if not workspace:
            return GuardrailDecision(allow=True)

        # Verify all preceding dispatch steps have completed (handoff file exists)
        missing: list[str] = []
        for i in range(target_idx):
            step = steps[i]
            if step.kind != "dispatch":
                continue
            # Check if the handoff file exists for this subagent
            handoff_name = to_handoff_name(step.target)
            handoff_path = Path(workspace) / f"handoff_{handoff_name}.json"
            if not handoff_path.exists():
                missing.append(step.target)

        if not missing:
            return GuardrailDecision(allow=True)

        # Deny with clear instruction
        missing_desc = "、".join(missing)
        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="ethoinsight.path_sequence_violation",
                message=(
                    f"按 {intent} 路径，{subagent_type} 之前必须先完成 {missing_desc}。"
                    f"请先 task({missing[-1]}) 完成该步骤。"
                ),
            )],
            policy_id="path_sequence",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)


class PathSequenceBridge(AgentMiddleware[AgentState]):
    """Sets _lead_messages and _lead_workspace contextvars from thread state.

    Must be placed BEFORE GuardrailMiddleware[PathSequenceProvider]
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
