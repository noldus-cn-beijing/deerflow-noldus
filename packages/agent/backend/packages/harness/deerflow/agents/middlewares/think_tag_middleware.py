"""Middleware: move `<think>...</think>` content out of AI message body into reasoning.

Many reasoning-capable models (DeepSeek-R1, minimax, Qwen reasoning variants,
and Claude's extended thinking when the harness renders it inline) emit
`<think>...</think>` tags in the assistant text. These tags are meant for
internal reasoning, not direct user display. The frontend already has a
dedicated collapsible "Reasoning" block driven by
`additional_kwargs.reasoning_content` — this middleware routes inline
`<think>` content into that channel so:

- The main message bubble only shows the final answer.
- The thinking is preserved, collapsed, and user-viewable on demand.
- No prompt-level "please use <think>" rule is required — we just handle
  whatever the model produces.

This runs in `after_model` so it also covers inline thinking produced by
Claude's extended thinking when the gateway serializes it as text (rather
than as a structured `thinking` content block).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Return `text` with all `<think>...</think>` blocks removed (trimmed)."""
    return _THINK_TAG_RE.sub("", text).strip()


def extract_think_tags(text: str) -> list[str]:
    """Return list of `<think>...</think>` inner contents (order preserved, trimmed, non-empty only)."""
    parts: list[str] = []
    for match in _THINK_TAG_RE.finditer(text):
        inner = match.group(0)[len("<think>") : -len("</think>")].strip()
        if inner:
            parts.append(inner)
    return parts


def split_think(text: str) -> tuple[str, str | None]:
    """Split `text` into (clean_body, merged_reasoning_or_None).

    `merged_reasoning` joins all think blocks with `\\n\\n` in source order.
    Returns `(text.strip(), None)` if no think blocks are present.
    """
    thinks = extract_think_tags(text)
    if not thinks:
        return text.strip(), None
    cleaned = strip_think_tags(text)
    return cleaned, "\n\n".join(thinks)


def _merge_reasoning(existing: Any, new: str) -> str:
    """Append `new` reasoning to an existing reasoning string (if any)."""
    if not existing:
        return new
    if isinstance(existing, str):
        return existing + "\n\n" + new
    return str(existing) + "\n\n" + new


class ThinkTagMiddleware(AgentMiddleware):
    """Relocate inline `<think>` blocks in AI messages into `reasoning_content`.

    Pure, stateless, idempotent. Only touches AI messages; only changes the
    newest AI message that has think tags (so repeated runs don't rewrite
    history unnecessarily). On a string-content message we rewrite the
    content; on a list-content message we rewrite only text parts. If no
    think tags are present, the message is unchanged.
    """

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._apply(state)

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._apply(state)

    def _apply(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # Scan from end; rewrite the most recent AI message with think tags.
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if not isinstance(msg, AIMessage):
                continue
            rewritten = self._rewrite(msg)
            if rewritten is None:
                continue
            new_messages = list(messages)
            new_messages[index] = rewritten
            return {"messages": new_messages}
        return None

    def _rewrite(self, msg: AIMessage) -> AIMessage | None:
        content = msg.content
        if isinstance(content, str):
            body, reasoning = split_think(content)
            if reasoning is None:
                return None
            kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
            kwargs["reasoning_content"] = _merge_reasoning(kwargs.get("reasoning_content"), reasoning)
            return msg.model_copy(update={"content": body, "additional_kwargs": kwargs})

        if isinstance(content, list):
            new_parts: list[Any] = []
            collected: list[str] = []
            mutated = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_value = part.get("text", "")
                    if isinstance(text_value, str) and "<think>" in text_value.lower():
                        body, reasoning = split_think(text_value)
                        if reasoning is not None:
                            mutated = True
                            collected.append(reasoning)
                            if body:
                                new_parts.append({**part, "text": body})
                            continue
                new_parts.append(part)

            if not mutated:
                return None

            kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
            if collected:
                kwargs["reasoning_content"] = _merge_reasoning(kwargs.get("reasoning_content"), "\n\n".join(collected))
            return msg.model_copy(update={"content": new_parts, "additional_kwargs": kwargs})

        return None
