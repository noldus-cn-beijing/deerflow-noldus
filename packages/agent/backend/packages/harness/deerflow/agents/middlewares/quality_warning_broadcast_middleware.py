"""Middleware that surfaces data-analyst quality warnings to the frontend.

Sprint 1 §2.5 promised the frontend would render quality warnings as a red/orange/
yellow/blue banner. The component (`quality-warning-banner.tsx`) is wired to read
`message.additional_kwargs.quality_warnings` from an AIMessage in `thread.messages`,
but nothing in the backend writes that field — `seal_data_analyst_handoff` only
persists `handoff_data_analyst.json` to the workspace, which never reaches the
frontend message stream.

This middleware closes the gap. After the lead model produces its broadcast
AIMessage following a `task(subagent_type="data-analyst")` ToolMessage, we read
`handoff_data_analyst.json` and copy its `quality_warnings` array onto that
AIMessage's `additional_kwargs`. LangGraph treats the same-id message return as
an in-place replacement, so the streamed message hits the frontend with the
banner-ready payload attached.

Triggers only when:
  - The last message in state is an AIMessage with no pending tool_calls
    (i.e. lead's content-only broadcast turn, the one the user actually sees).
  - Somewhere earlier in the message list there is a ToolMessage whose
    matching `task` tool_call had `subagent_type == "data-analyst"` (or
    "data_analyst", the underscore alias).
  - `handoff_data_analyst.json` exists and parses to a dict.
  - The AIMessage doesn't already carry `quality_warnings` (idempotent).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.experiment_context import resolve_workspace_from_state

logger = logging.getLogger(__name__)

_HANDOFF_FILENAME = "handoff_data_analyst.json"
_DATA_ANALYST_ALIASES = frozenset({"data-analyst", "data_analyst"})


class QualityWarningBroadcastMiddleware(AgentMiddleware[AgentState]):
    """Inject data-analyst handoff quality_warnings onto lead's broadcast AIMessage."""

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._maybe_inject(state)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._maybe_inject(state)

    def _maybe_inject(self, state: AgentState) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if getattr(last, "type", None) != "ai":
            return None
        if getattr(last, "tool_calls", None):
            # Lead is dispatching another tool; not the broadcast turn yet.
            return None

        existing_kwargs = getattr(last, "additional_kwargs", None) or {}
        if "quality_warnings" in existing_kwargs:
            return None

        if not _has_recent_data_analyst_toolmessage(messages):
            return None

        warnings = _load_quality_warnings(state)
        if warnings is None:
            # Either no workspace or no handoff file; nothing to inject.
            return None

        updated_kwargs = dict(existing_kwargs)
        updated_kwargs["quality_warnings"] = warnings
        updated_msg = last.model_copy(update={"additional_kwargs": updated_kwargs})

        logger.info(
            "quality_warning_broadcast | thread=%s | injected=%d warnings",
            state.get("thread_id", "unknown"),
            len(warnings),
        )
        return {"messages": [updated_msg]}


def _has_recent_data_analyst_toolmessage(messages: list[Any]) -> bool:
    """Return True if any prior ToolMessage corresponds to a data-analyst task call.

    Builds a map of tool_call_id -> subagent_type from every `task` tool call
    in earlier AIMessages, then walks backwards through the message history
    looking for the most recent ToolMessage. If that ToolMessage's
    `tool_call_id` was a data-analyst dispatch, we treat it as the upstream
    handoff we want to surface.

    Note: We don't stop at the first non-data-analyst ToolMessage. The lead
    can interleave chart-maker / report-writer calls between data-analyst
    and the final broadcast, but the handoff file itself remains the
    authoritative artefact regardless of intermediate calls.
    """
    tool_call_id_to_subagent: dict[str, str] = {}
    for msg in messages[:-1]:
        if getattr(msg, "type", None) != "ai":
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "task":
                continue
            args = tc.get("args") or {}
            sub = args.get("subagent_type")
            if isinstance(sub, str) and tc.get("id"):
                tool_call_id_to_subagent[tc["id"]] = sub

    if not tool_call_id_to_subagent:
        return False

    for msg in reversed(messages[:-1]):
        if getattr(msg, "type", None) != "tool":
            continue
        tcid = getattr(msg, "tool_call_id", None)
        if not tcid:
            continue
        sub = tool_call_id_to_subagent.get(tcid)
        if sub and sub in _DATA_ANALYST_ALIASES:
            return True

    return False


def _load_quality_warnings(state: AgentState) -> list[dict] | None:
    """Read handoff_data_analyst.json and return its quality_warnings list.

    Returns None on any failure (no workspace path, file missing, parse error,
    or field missing/wrong type). Returns [] if the file exists but has no
    warnings — the middleware passes that through as "explicitly zero", but
    note `_maybe_inject` will still inject [] which lets the frontend assert
    "no warnings" rather than "data not loaded yet".
    """
    workspace_dir = resolve_workspace_from_state(state)
    if not workspace_dir:
        return None

    path = Path(workspace_dir) / _HANDOFF_FILENAME
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", _HANDOFF_FILENAME, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse %s: %s", _HANDOFF_FILENAME, exc)
        return None

    if not isinstance(data, dict):
        return None

    # status=in_progress = harness 预置的「未封口」data-analyst 模板（spec
    # 2026-06-23-data-analyst-seal-stepwise-fill-template §3.5），**不是交付物**——
    # 不得消费它的字段（quality_warnings 恒空）。返回 None = 当「未交付/未读到」处理。
    if data.get("status") == "in_progress":
        return None

    warnings = data.get("quality_warnings")
    if not isinstance(warnings, list):
        return None

    return [w for w in warnings if isinstance(w, dict)]
