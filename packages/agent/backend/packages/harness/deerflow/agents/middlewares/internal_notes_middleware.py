"""Middleware that hides lead agent's internal status notes from the UI.

The lead model (Sonnet, GLM, and others) sometimes emits large internal
status blocks as plain AI text between tool calls — e.g.:

    ## 提取的关键上下文

    ### 任务概述
    ...
    ### 当前执行状态（Todo）
    ...

These dumps are useful to the model itself (keeps orchestration state in
context) but they pollute the user-facing chat stream. Instead of fighting
the model in the prompt, we tag these messages with
``additional_kwargs.hide_from_ui = True`` so the frontend (which already
checks this flag via ``isHiddenFromUIMessage``) simply skips rendering them.

The tagged messages remain in LangGraph state — the model continues to see
its own notes on subsequent turns.

Pattern list is deliberately narrow:
- Starts with a section heading whose title matches a known "internal notes"
  phrase (exact-match, no fuzzy detection).
- Must be an AI message with no tool_calls (pure narration, not an action).

Synthesis messages the user *should* see (e.g. "### 分析结果 / 关键指标 /
关键洞察" template from commit 5) start with a different heading and are
untouched.
"""

from __future__ import annotations

import logging
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


# Exact heading prefixes that mark a message as internal orchestration notes.
# Each entry is matched against the *start* of the content (after stripping
# leading whitespace).
_INTERNAL_NOTE_HEADINGS: tuple[str, ...] = (
    "## 提取的关键上下文",
    "## Extracted Context",
    "## Key Context",
)


def _is_internal_notes_ai_message(msg: object) -> bool:
    """Return True when ``msg`` is a lead-agent internal-notes AI message.

    Requires:
    - AI message type
    - No tool_calls (pure narration)
    - content is a string (not a list of blocks)
    - content starts with one of the known internal-note headings
    """
    if getattr(msg, "type", None) != "ai":
        return False
    if getattr(msg, "tool_calls", None):
        return False
    content = getattr(msg, "content", None)
    if not isinstance(content, str):
        return False
    stripped = content.lstrip()
    return any(stripped.startswith(h) for h in _INTERNAL_NOTE_HEADINGS)


def _already_hidden(msg: object) -> bool:
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    return bool(kwargs.get("hide_from_ui"))


def _tag_last_message_hidden(state: AgentState) -> dict | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    if not _is_internal_notes_ai_message(last_msg):
        return None
    if _already_hidden(last_msg):
        return None

    # Merge with any existing additional_kwargs (preserve other flags).
    existing_kwargs = dict(getattr(last_msg, "additional_kwargs", None) or {})
    existing_kwargs["hide_from_ui"] = True

    updated = last_msg.model_copy(update={"additional_kwargs": existing_kwargs})
    logger.debug(
        "Tagged AI message %s as hide_from_ui (internal notes, %d chars)",
        getattr(last_msg, "id", "?"),
        len(getattr(last_msg, "content", "") or ""),
    )
    # Same id triggers in-place replacement in the LangGraph reducer.
    return {"messages": [updated]}


class InternalNotesMiddleware(AgentMiddleware[AgentState]):
    """Tags lead-agent internal status dumps with hide_from_ui.

    Runs in ``after_model`` so the most recent AI message is available.
    Leaves state otherwise untouched — the message stays in context for
    the model on subsequent turns; only the frontend skips rendering.
    """

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return _tag_last_message_hidden(state)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return _tag_last_message_hidden(state)


__all__ = ["InternalNotesMiddleware"]
