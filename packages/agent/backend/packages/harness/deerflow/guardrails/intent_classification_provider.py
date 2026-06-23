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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.path_registry import (
    VALID_INTENTS as _VALID_INTENTS,
)
from deerflow.guardrails.path_registry import (
    VIZ_INTENT_KEYWORDS as _VIZ_INTENT_KEYWORDS,
)
from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

logger = logging.getLogger(__name__)


_lead_messages: ContextVar[list | None] = ContextVar("_lead_messages", default=None)

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


def _latest_declared_intent(messages: list | None) -> str | None:
    """Return the most recent valid [intent] name across the conversation.

    ETHO-7 子改动 A 用它判断 lead 当前声明的是哪个 intent（E2E_FULL vs
    E2E_FULL_ASKVIZ）。取「最近一条」是因为 lead 可能在前几轮声明过别的 intent，
    当前生效的是最新那条。
    """
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
            name = match.group(1)
            if name in _VALID_INTENTS:
                return name
    return None


def _user_text_has_viz_intent(messages: list | None) -> bool:
    """检测对话里**用户**（HumanMessage）是否表达了明确出图意向。

    ETHO-7 子改动 A 的 reward-hacking 防护（§六.1）：只看 HumanMessage 实际文本，
    不看 AIMessage —— lead 不能在自述里假声明「用户要图」骗过本检查。
    """
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, HumanMessage):
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
        if any(kw in content for kw in _VIZ_INTENT_KEYWORDS):
            return True
    return False


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
        if not declared:
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

        # ETHO-7 子改动 A：E2E_FULL（跳出图反问）需对话里有用户明确出图意向。
        # 把「该不该出现出图决策点」从 lead LLM 自由裁量变成可检测规则：
        # 声明 E2E_FULL 但用户实际没提图 → deny，要求改声明 E2E_FULL_ASKVIZ 先问。
        # 这消除「是否出图」决策点运行间随机消失（Run2 跳过它）的现象。
        latest_intent = _latest_declared_intent(messages)
        if latest_intent == "E2E_FULL" and not _user_text_has_viz_intent(messages):
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="ethoinsight.e2e_full_requires_explicit_viz_intent",
                    message=(
                        "你声明了 [intent] E2E_FULL（跳过出图反问、直接出图），"
                        "但对话里用户没有明确表达出图意向（未出现「画/图/可视化/表/箱线」等词）。"
                        "请改声明 [intent] E2E_FULL_ASKVIZ：跑完解读后先反问用户是否需要图表，"
                        "再根据用户回答决定是否派 chart-maker。"
                        "判定依据在 path_registry.VIZ_INTENT_KEYWORDS（用户实际文本，非 lead 自述）。"
                    ),
                )],
                policy_id="intent_classification",
            )

        return GuardrailDecision(allow=True)

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
