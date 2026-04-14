"""Middleware for logging LLM token usage."""

import logging
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class TokenUsageMiddleware(AgentMiddleware):
    """Logs token usage from model response usage_metadata."""

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._log_usage(state)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._log_usage(state)

    def _log_usage(self, state: AgentState) -> None:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        usage = getattr(last, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("input_tokens", "?")
            output_tokens = usage.get("output_tokens", "?")
            logger.info(
                "LLM token usage: input=%s output=%s total=%s",
                input_tokens,
                output_tokens,
                usage.get("total_tokens", "?"),
            )
            # Debug: log message details when tokens are zero (abnormal)
            if input_tokens == 0 and output_tokens == 0:
                content = getattr(last, "content", None)
                tool_calls = getattr(last, "tool_calls", None)
                msg_type = getattr(last, "type", None)
                content_type = type(content).__name__
                content_preview = repr(content)[:200] if content else "EMPTY"
                logger.warning(
                    "ZERO-TOKEN DEBUG: msg_type=%s, content_type=%s, content=%s, tool_calls=%s, num_messages=%d",
                    msg_type, content_type, content_preview, tool_calls, len(messages),
                )
                # Log content types of last few messages
                for i, msg in enumerate(messages[-3:]):
                    mc = getattr(msg, "content", None)
                    logger.warning(
                        "ZERO-TOKEN DEBUG: messages[-%d] type=%s content_type=%s content_preview=%s",
                        3 - i, getattr(msg, "type", "?"), type(mc).__name__, repr(mc)[:150],
                    )
        return None
