"""Tests for ParadigmIdentificationGateMiddleware (layer 3a).

Covers:
  - No uploaded files → pass through
  - Last AI has identify in tool_calls → pass through
  - History has identify ToolMessage → pass through
  - Has uploads + no identify → inject reminder + jump_to
  - Reminder cap (2) → pass through after cap reached
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.paradigm_identification_gate_middleware import (
    ParadigmIdentificationGateMiddleware,
    _IDENTIFY_TOOL,
)


def _make_runtime(run_id: str = "test-run") -> MagicMock:
    rt = MagicMock()
    rt.run_id = run_id
    return rt


def _make_state(messages: list | None = None, uploaded_files: list | None = None) -> dict:
    return {
        "messages": messages or [],
        "uploaded_files": uploaded_files,
    }


class TestParadigmIdentificationGate:
    """Unit tests for ParadigmIdentificationGateMiddleware."""

    def test_no_uploaded_files_passes_through(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        state = _make_state(messages=[AIMessage(content="hello")], uploaded_files=None)
        result = mw.after_model(state, _make_runtime())
        assert result is None

    def test_empty_uploaded_files_passes_through(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        state = _make_state(messages=[AIMessage(content="hello")], uploaded_files=[])
        result = mw.after_model(state, _make_runtime())
        assert result is None

    def test_identify_in_last_ai_tool_calls_passes_through(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(
            content="",
            tool_calls=[{"name": _IDENTIFY_TOOL, "args": {}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai], uploaded_files=["/mnt/user-data/uploads/fst.txt"])
        result = mw.after_model(state, _make_runtime())
        assert result is None

    def test_identify_in_history_passes_through(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(content="", tool_calls=[{"name": _IDENTIFY_TOOL, "args": {}, "id": "tc1"}])
        tm = ToolMessage(content="ok", tool_call_id="tc1", name=_IDENTIFY_TOOL)
        state = _make_state(
            messages=[ai, tm, AIMessage(content="next step")],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        result = mw.after_model(state, _make_runtime())
        assert result is None

    def test_no_identify_with_uploads_injects_reminder(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(content="I think this is FST...", tool_calls=[])
        state = _make_state(
            messages=[ai],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"
        assert len(result.get("messages", [])) == 1
        msg = result["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert "identify_ev19_template" in msg.content
        assert msg.additional_kwargs.get("hide_from_ui") is True

    def test_no_identify_asks_clarification_blocked(self) -> None:
        """Agent tries ask_clarification without calling identify first."""
        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(
            content="Let me ask the user",
            tool_calls=[{"name": "ask_clarification", "args": {"question": "FST or TST?"}, "id": "tc1"}],
        )
        state = _make_state(
            messages=[ai],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"

    def test_reminder_cap_prevents_infinite_loop(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        runtime = _make_runtime()
        state = _make_state(
            messages=[AIMessage(content="no identify")],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        # First two should inject reminders
        result1 = mw.after_model(state, runtime)
        assert result1 is not None
        result2 = mw.after_model(state, runtime)
        assert result2 is not None
        # Third should be capped
        result3 = mw.after_model(state, runtime)
        assert result3 is None

    def test_fail_open_on_exception(self) -> None:
        mw = ParadigmIdentificationGateMiddleware()
        # state without messages key at all
        state = {"uploaded_files": ["file.txt"]}
        result = mw.after_model(state, _make_runtime())
        assert result is None

    def test_async_delegates_to_sync(self) -> None:
        import asyncio

        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(content="no identify", tool_calls=[])
        state = _make_state(
            messages=[ai],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        result = asyncio.get_event_loop().run_until_complete(mw.aafter_model(state, _make_runtime()))
        assert result is not None
        assert result.get("jump_to") == "model"

    def test_pure_text_no_tool_calls_with_uploads(self) -> None:
        """Agent outputs pure text (no tool_calls at all) with uploads present."""
        mw = ParadigmIdentificationGateMiddleware()
        ai = AIMessage(content="我先来识别一下范式... 这应该是强迫游泳数据")
        state = _make_state(
            messages=[ai],
            uploaded_files=["/mnt/user-data/uploads/fst.txt"],
        )
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"
