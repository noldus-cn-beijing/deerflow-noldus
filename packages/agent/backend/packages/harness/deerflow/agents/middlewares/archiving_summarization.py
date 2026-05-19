"""Archiving wrapper around LangChain's SummarizationMiddleware.

When context compaction fires this middleware does three things:

1. **Archive** the messages being dropped to `{thread_dir}/archived_messages/`
   as JSON (same behavior as before — used by the frontend to restore history
   after page refresh).
2. **Write a human-readable summary** to
   `{thread_dir}/user-data/workspace/conversation_summary.md` so the lead
   agent can read it back with the sandbox `read_file` tool. Multiple
   compactions append to the same file under timestamped headings, keeping a
   single file as the source of truth for all compressed history.
3. **Inject a tiny pointer HumanMessage** back into the LangGraph state with
   `additional_kwargs.hide_from_ui=True`. The pointer is one short line
   telling the model where the summary lives; it is hidden from the UI so it
   does not appear as a spurious user bubble. The model's own `<think>`
   outputs (via `ThinkTagMiddleware`) and the `compaction-recovery` skill
   tell it when to actually read the file.

The old behavior — injecting the full LangChain summary as a HumanMessage
prefixed with "Here is a summary..." — is replaced. That behavior leaked
structured-dump summary content into the conversation, where it both
rendered as a user bubble in the UI and encouraged the lead to mimic the
same heading-and-bullet format in its own replies. Text > Brain: let the
summary live in a file, let the model fetch it when needed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.runtime import Runtime

from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)

ARCHIVE_DIR_NAME = "archived_messages"
SUMMARY_FILENAME = "conversation_summary.md"
SUMMARY_VIRTUAL_PATH = f"/mnt/user-data/workspace/{SUMMARY_FILENAME}"

_POINTER_TEMPLATE = (
    "[系统] 前序对话已压缩并追加到 `{path}`。"
    "如需历史细节，使用 `read_file` 读取该文件。"
)


def _get_thread_id(runtime: Runtime) -> str | None:
    """Extract thread_id from runtime context."""
    return runtime.context.get("thread_id") if runtime.context else None


class ArchivingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that archives removed messages to disk and
    writes the compressed summary to a sandbox-visible file.

    Optionally accepts a ``loop_detection`` middleware instance.  When set,
    the loop-detection tracking state for the current thread is cleared after
    every summarization so that the reduced context and the detector stay in
    sync (otherwise the detector could carry stale counts from messages that
    were already removed).
    """

    def __init__(self, *args: Any, loop_detection: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loop_detection = loop_detection
        # Tag the summarization model call with "nostream" so its token
        # deltas are not broadcast to the conversation stream
        # (langgraph.constants.TAG_NOSTREAM). Fine-grained delta providers
        # like DeepSeek otherwise leak the summary into the UI as a phantom
        # AI message.
        self.model = self.model.with_config(tags=["nostream"])

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

        self._archive_messages(messages_to_summarize, runtime)
        self._reset_loop_detection(runtime)

        summary = await self._acreate_summary(messages_to_summarize)
        new_messages = self._build_file_backed_messages(summary, runtime)

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

        self._archive_messages(messages_to_summarize, runtime)
        self._reset_loop_detection(runtime)

        summary = self._create_summary(messages_to_summarize)
        new_messages = self._build_file_backed_messages(summary, runtime)

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

    def _build_file_backed_messages(self, summary: str, runtime: Runtime) -> list[HumanMessage]:
        """Write summary to a workspace file and return a hidden pointer message.

        Falls back to the upstream "full summary in a HumanMessage" behaviour
        if we cannot resolve the thread workspace (e.g. no thread_id in
        runtime context) — better to have the compressed content in the
        conversation than to lose it entirely.
        """
        thread_id = _get_thread_id(runtime)
        summary_path = self._write_summary_file(summary, thread_id)

        if summary_path is None:
            # Fallback: upstream behaviour, but still hide from UI so we
            # do not render the dump as a user bubble.
            return self._fallback_inline_messages(summary)

        pointer_text = _POINTER_TEMPLATE.format(path=SUMMARY_VIRTUAL_PATH)
        pointer = HumanMessage(
            content=pointer_text,
            additional_kwargs={
                "hide_from_ui": True,
                "conversation_summary_path": SUMMARY_VIRTUAL_PATH,
            },
        )
        self._ensure_message_ids([pointer])
        return [pointer]

    def _fallback_inline_messages(self, summary: str) -> list[HumanMessage]:
        """When we cannot write to disk, fall back to upstream inline summary
        but still mark it `hide_from_ui` so the frontend does not render it.
        """
        upstream_messages = super()._build_new_messages(summary)
        tagged: list[HumanMessage] = []
        for msg in upstream_messages:
            kwargs = dict(getattr(msg, "additional_kwargs", None) or {})
            kwargs["hide_from_ui"] = True
            tagged.append(msg.model_copy(update={"additional_kwargs": kwargs}))
        return tagged

    def _write_summary_file(self, summary: str, thread_id: str | None) -> Path | None:
        """Append `summary` under a timestamped heading to the workspace file.
        Returns the host path on success, None on failure (caller falls back)."""
        if not thread_id:
            logger.warning("Cannot write conversation summary: no thread_id in runtime context")
            return None

        try:
            paths = get_paths()
            workspace_dir = paths.sandbox_work_dir(thread_id, user_id=get_effective_user_id())
            workspace_dir.mkdir(parents=True, exist_ok=True)
            summary_path = workspace_dir / SUMMARY_FILENAME

            heading = f"## 压缩于 {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            body = summary.strip() + "\n\n"

            # Append mode so multiple compactions accumulate into one file.
            with summary_path.open("a", encoding="utf-8") as fh:
                if summary_path.stat().st_size == 0:
                    fh.write("# Conversation Summary\n\n")
                    fh.write(
                        "_This file accumulates compressed conversation history. "
                        "Each section below is one compaction event._\n\n"
                    )
                fh.write(heading)
                fh.write(body)

            logger.info(
                "Wrote conversation summary for thread %s (%d chars) -> %s",
                thread_id,
                len(body),
                summary_path,
            )
            return summary_path
        except Exception:
            logger.exception("Failed to write conversation summary for thread %s", thread_id)
            return None

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
            archive_dir = paths.thread_dir(thread_id, user_id=get_effective_user_id()) / ARCHIVE_DIR_NAME
            archive_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            archive_file = archive_dir / f"{timestamp}.json"

            serialized = messages_to_dict(messages_to_archive)
            payload = {
                "archived_at": datetime.now(UTC).isoformat(),
                "message_count": len(messages_to_archive),
                "messages": serialized,
            }

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
