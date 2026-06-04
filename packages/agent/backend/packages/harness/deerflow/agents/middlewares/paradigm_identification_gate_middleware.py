"""ParadigmIdentificationGateMiddleware — after_model gate to force identify_ev19_template call.

Replicates deerflow upstream commit #2135 (e4f896e9) pattern:
after_model + @hook_config(can_jump_to=["model"]) to detect "agent produced no
identify tool_call but should have" → inject reminder HumanMessage + jump back.

This is the first line of defense (layer 3a). The InspectGateGuardrailProvider
(layer 3b) serves as a safety net when this middleware's cap is reached.

Key behaviors:
  - Only intervenes when there are uploaded files in the current turn
  - Allows through if identify_ev19_template was already called (ToolMessage exists)
  - Allows through if last AIMessage has identify_ev19_template in its tool_calls
  - Injects a positive-instruction reminder and jumps to model when identify is missing
  - Caps reminders at 2 per turn to prevent infinite loops
  - fail-open: state/messages unavailable → return None
"""

from __future__ import annotations

import logging
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_IDENTIFY_TOOL = "identify_ev19_template"
_MAX_REMINDERS = 2

_REMINDER_CONTENT = (
    "<system_reminder>\n"
    "检测到本轮有上传数据文件,但你尚未真实调用 identify_ev19_template 工具。\n"
    "你对 EV19 模板的判断必须来自该工具的真实返回值。请现在调用\n"
    "identify_ev19_template(uploaded_files=..., user_message=...),用它的返回再决定下一步。\n"
    "</system_reminder>"
)


def _has_identify_in_history(messages: list) -> bool:
    """Check if there's a ToolMessage from identify_ev19_template in the conversation."""
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == _IDENTIFY_TOOL:
            return True
    return False


def _has_identify_in_last_ai_tool_calls(messages: list) -> bool:
    """Check if the last AIMessage has identify_ev19_template in its tool_calls."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if tc.get("name") == _IDENTIFY_TOOL:
                    return True
            break
    return False


def _has_uploaded_files_in_state(state: dict) -> bool:
    """Check if there are uploaded files in the current turn's state."""
    uploaded = state.get("uploaded_files")
    if not uploaded:
        return False
    # uploaded_files is a list; non-empty means there are files
    return len(uploaded) > 0


class ParadigmIdentificationGateMiddleware(AgentMiddleware):
    """after_model middleware that forces identify_ev19_template to be called before
    the agent can proceed to ask_clarification or dispatch subagents when uploaded
    files are present.

    Must be placed in the middleware chain before ClarificationMiddleware.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reminder_counts: dict[str, int] = {}

    def _get_reminder_count(self, runtime: Runtime) -> int:
        """Get the reminder count for the current run."""
        key = getattr(runtime, "run_id", None) or id(runtime)
        return self._reminder_counts.get(str(key), 0)

    def _increment_reminder_count(self, runtime: Runtime) -> None:
        """Increment the reminder count for the current run."""
        key = str(getattr(runtime, "run_id", None) or id(runtime))
        self._reminder_counts[key] = self._reminder_counts.get(key, 0) + 1

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Check after model output whether identify should have been called."""
        try:
            return self._check(state, runtime)
        except Exception:
            logger.debug("ParadigmIdentificationGateMiddleware: check failed, fail-open", exc_info=True)
            return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Async version — delegates to sync."""
        return self.after_model(state, runtime)

    def _check(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # 1. No uploaded data → not our concern
        if not _has_uploaded_files_in_state(state):
            return None

        # 2. Last AIMessage already has identify in tool_calls → it's being called, allow
        if _has_identify_in_last_ai_tool_calls(messages):
            return None

        # 3. identify was already called in history → allow
        if _has_identify_in_history(messages):
            return None

        # 4. Cap: too many reminders → allow (guardrail is the safety net)
        if self._get_reminder_count(runtime) >= _MAX_REMINDERS:
            return None

        # 5. Agent should have called identify but didn't → inject reminder + jump
        self._increment_reminder_count(runtime)
        logger.info(
            "ParadigmIdentificationGateMiddleware: injecting identify reminder (count=%d)",
            self._get_reminder_count(runtime),
        )
        return {
            "messages": [
                HumanMessage(
                    content=_REMINDER_CONTENT,
                    name="paradigm_identification_reminder",
                    additional_kwargs={"hide_from_ui": True},
                )
            ],
            "jump_to": "model",
        }
