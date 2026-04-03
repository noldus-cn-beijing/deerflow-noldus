"""Middleware to fix dangling tool calls in message history.

A dangling tool call occurs when an AIMessage contains tool_calls but there are
no corresponding ToolMessages in the history (e.g., due to user interruption or
request cancellation). This causes LLM errors due to incomplete message format.

This middleware also removes orphan ToolMessages whose tool_call_id does not
match any AIMessage tool_call (e.g. after summarization removed the AI message).

Note: Uses wrap_model_call instead of before_model to ensure patches are inserted
at the correct positions (immediately after each dangling AIMessage), not appended
to the end of the message list as before_model + add_messages reducer would do.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """Inserts placeholder ToolMessages for dangling tool calls before model invocation.

    Scans the message history for AIMessages whose tool_calls lack corresponding
    ToolMessages, and injects synthetic error responses immediately after the
    offending AIMessage so the LLM receives a well-formed conversation.

    Also removes orphan ToolMessages that have no matching AI tool_call.
    """

    def _build_patched_messages(self, messages: list) -> list | None:
        """Return a new message list with patches inserted at the correct positions.

        For each AIMessage with dangling tool_calls (no corresponding ToolMessage),
        a synthetic ToolMessage is inserted immediately after that AIMessage.

        Also removes orphan ToolMessages whose tool_call_id does not match any
        AIMessage tool_call (e.g. after summarization removed the AI message).

        Returns None if no patches are needed.
        """
        # Collect IDs of all existing ToolMessages
        existing_tool_msg_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                existing_tool_msg_ids.add(msg.tool_call_id)

        # Collect IDs of all AI tool_calls
        ai_tool_call_ids: set[str] = set()
        for msg in messages:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id:
                    ai_tool_call_ids.add(tc_id)

        # Check if any patching is needed (dangling AI→Tool or orphan Tool→AI)
        needs_patch = False
        for msg in messages:
            if getattr(msg, "type", None) == "ai":
                for tc in getattr(msg, "tool_calls", None) or []:
                    tc_id = tc.get("id")
                    if tc_id and tc_id not in existing_tool_msg_ids:
                        needs_patch = True
                        break
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id and msg.tool_call_id not in ai_tool_call_ids:
                    needs_patch = True
            if needs_patch:
                break

        if not needs_patch:
            return None

        # Build new list with patches and orphan removal
        patched: list = []
        patched_ids: set[str] = set()
        patch_count = 0
        orphan_count = 0
        for msg in messages:
            # Skip orphan ToolMessages (no matching AI tool_call)
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id and msg.tool_call_id not in ai_tool_call_ids:
                    orphan_count += 1
                    continue
            patched.append(msg)
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id and tc_id not in existing_tool_msg_ids and tc_id not in patched_ids:
                    patched.append(
                        ToolMessage(
                            content="[Tool call was interrupted and did not return a result.]",
                            tool_call_id=tc_id,
                            name=tc.get("name", "unknown"),
                            status="error",
                        )
                    )
                    patched_ids.add(tc_id)
                    patch_count += 1

        if patch_count:
            logger.warning(f"Injecting {patch_count} placeholder ToolMessage(s) for dangling tool calls")
        if orphan_count:
            logger.warning(f"Removed {orphan_count} orphan ToolMessage(s) without matching AI tool_call")
        return patched

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return await handler(request)
