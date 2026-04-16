"""Archiving wrapper around LangChain's SummarizationMiddleware.

Before old messages are removed from LangGraph state, this middleware persists
them as JSON files under the thread's data directory so the frontend can recover
the full conversation history after a page refresh.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, override

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import messages_to_dict
from langgraph.runtime import Runtime

from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)

ARCHIVE_DIR_NAME = "archived_messages"


def _get_thread_id(runtime: Runtime) -> str | None:
    """Extract thread_id from runtime context."""
    return runtime.context.get("thread_id") if runtime.context else None


class ArchivingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that archives removed messages to disk.

    Optionally accepts a ``loop_detection`` middleware instance.  When set,
    the loop-detection tracking state for the current thread is cleared after
    every summarization so that the reduced context and the detector stay in
    sync (otherwise the detector could carry stale counts from messages that
    were already removed).
    """

    def __init__(self, *args: Any, loop_detection: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loop_detection = loop_detection

    @override
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # Archive messages before they are removed
        self._archive_messages(messages_to_summarize, runtime)

        # Reset loop detection so stale counts don't survive compaction
        self._reset_loop_detection(runtime)

        # Proceed with normal summarization
        summary = await self._acreate_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        from langchain_core.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # Archive messages before they are removed
        self._archive_messages(messages_to_summarize, runtime)

        # Reset loop detection so stale counts don't survive compaction
        self._reset_loop_detection(runtime)

        # Proceed with normal summarization
        summary = self._create_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        from langchain_core.messages import RemoveMessage
        from langgraph.graph.message import REMOVE_ALL_MESSAGES

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _reset_loop_detection(self, runtime: Runtime) -> None:
        """Clear loop-detection state for this thread after summarization."""
        if self._loop_detection is None:
            return
        thread_id = _get_thread_id(runtime)
        if thread_id:
            try:
                self._loop_detection.reset(thread_id)
                logger.debug("Reset loop detection for thread %s after summarization", thread_id)
            except Exception:
                logger.exception("Failed to reset loop detection for thread %s", thread_id)

    def _archive_messages(self, messages_to_archive: list, runtime: Runtime) -> None:
        """Persist messages to a JSON file in the thread's data directory."""
        if not messages_to_archive:
            return

        thread_id = _get_thread_id(runtime)
        if not thread_id:
            logger.warning("Cannot archive messages: no thread_id in runtime context")
            return

        try:
            paths = get_paths()
            archive_dir = paths.thread_dir(thread_id) / ARCHIVE_DIR_NAME
            archive_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            archive_file = archive_dir / f"{timestamp}.json"

            serialized = messages_to_dict(messages_to_archive)
            payload = {
                "archived_at": datetime.now(UTC).isoformat(),
                "message_count": len(messages_to_archive),
                "messages": serialized,
            }

            # Atomic write: write to temp then rename
            tmp_file = archive_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
            tmp_file.rename(archive_file)

            logger.info(
                "Archived %d messages for thread %s -> %s",
                len(messages_to_archive),
                thread_id,
                archive_file.name,
            )
        except Exception:
            logger.exception("Failed to archive messages for thread %s", thread_id)
