"""Tests for SealGateMiddleware (L1 structural seal gate, spec 2026-06-16).

Covers the after_model gating logic that forces seal_<name>_handoff before a
seal-requiring subagent (data-analyst / chart-maker / report-writer) can exit on a
pure-text AIMessage. Mirrors the test shape of test_paradigm_identification_gate.py.

Cases (spec §5.1):
  - red→green core: last AIMessage no tool_call + no seal ToolMessage + data-analyst
    → inject reminder + jump_to='model'
  - allow: last AIMessage has seal tool_call → None
  - allow: history has seal ToolMessage → None
  - allow: last AIMessage has other tool_call (still working) → None
  - allow: subagent = code-executor → None
  - allow: reminder cap (2) reached → None
  - fail-open: state missing messages / raises → None
  - per-run reminder isolation: different run_id independent counts
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.seal_gate_middleware import (
    _REQUIRES_SEAL,
    SealGateMiddleware,
    _seal_tool_name,
)


def _make_runtime(run_id: str = "test-run") -> MagicMock:
    rt = MagicMock()
    rt.run_id = run_id
    return rt


def _make_state(messages: list | None = None) -> dict:
    return {"messages": messages if messages is not None else []}


class TestSealGateMiddleware:
    """Unit tests for SealGateMiddleware."""

    def test_seal_tool_name_derivation(self) -> None:
        assert _seal_tool_name("data-analyst") == "seal_data_analyst_handoff"
        assert _seal_tool_name("chart-maker") == "seal_chart_maker_handoff"
        assert _seal_tool_name("report-writer") == "seal_report_writer_handoff"

    def test_requires_seal_set(self) -> None:
        assert "data-analyst" in _REQUIRES_SEAL
        assert "chart-maker" in _REQUIRES_SEAL
        assert "report-writer" in _REQUIRES_SEAL
        assert "code-executor" not in _REQUIRES_SEAL

    def test_core_red_green_data_analyst_pure_text_injects_reminder(self) -> None:
        """Model ends on pure text, seal not called → must force back to model."""
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[AIMessage(content="分析已完成")])
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"
        assert len(result.get("messages", [])) == 1
        msg = result["messages"][0]
        assert isinstance(msg, HumanMessage)
        assert "seal_data_analyst_handoff" in msg.content
        assert msg.additional_kwargs.get("hide_from_ui") is True

    def test_allow_last_ai_has_seal_tool_call(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "seal_data_analyst_handoff", "args": {}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai])
        assert mw.after_model(state, _make_runtime()) is None

    def test_allow_history_has_seal_tool_message(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "seal_data_analyst_handoff", "args": {}, "id": "tc1"}],
        )
        tm = ToolMessage(content="ok", tool_call_id="tc1", name="seal_data_analyst_handoff")
        # Last AIMessage is pure text (wants to end), but seal already happened.
        state = _make_state(messages=[ai, tm, AIMessage(content="done")])
        assert mw.after_model(state, _make_runtime()) is None

    def test_allow_last_ai_has_other_tool_call_still_working(self) -> None:
        """Last AIMessage carries a non-seal tool_call → don't interrupt the loop."""
        mw = SealGateMiddleware("data-analyst")
        ai = AIMessage(
            content="let me read the paradigm doc",
            tool_calls=[{"name": "read_file", "args": {"path": "/mnt/..."}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai])
        assert mw.after_model(state, _make_runtime()) is None

    def test_allow_non_seal_subagent_code_executor(self) -> None:
        mw = SealGateMiddleware("code-executor")
        state = _make_state(messages=[AIMessage(content="done computing")])
        assert mw.after_model(state, _make_runtime()) is None

    def test_allow_non_seal_subagent_general(self) -> None:
        mw = SealGateMiddleware("general-purpose")
        state = _make_state(messages=[AIMessage(content="done")])
        assert mw.after_model(state, _make_runtime()) is None

    def test_reminder_cap_prevents_infinite_loop(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        runtime = _make_runtime()
        state = _make_state(messages=[AIMessage(content="no seal")])
        # First two inject reminders; third is capped → None
        assert mw.after_model(state, runtime) is not None
        assert mw.after_model(state, runtime) is not None
        assert mw.after_model(state, runtime) is None

    def test_reminder_count_isolated_per_run(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[AIMessage(content="no seal")])
        rt_a = _make_runtime("run-a")
        rt_b = _make_runtime("run-b")
        # run-a fires twice (capped), run-b independent
        assert mw.after_model(state, rt_a) is not None
        assert mw.after_model(state, rt_b) is not None
        assert mw.after_model(state, rt_a) is not None  # run-a 2nd
        assert mw.after_model(state, rt_a) is None  # run-a capped
        assert mw.after_model(state, rt_b) is not None  # run-b still has budget
        assert mw.after_model(state, rt_b) is None  # run-b capped

    def test_fail_open_on_missing_messages_key(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        # state without messages key at all
        result = mw.after_model({}, _make_runtime())
        assert result is None

    def test_fail_open_on_non_dict_state(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        # non-dict state → state.get unavailable → fail-open
        result = mw.after_model(None, _make_runtime())  # type: ignore[arg-type]
        assert result is None

    def test_fail_open_on_empty_messages(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[])
        assert mw.after_model(state, _make_runtime()) is None

    def test_fail_open_when_no_ai_message(self) -> None:
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[HumanMessage(content="hi")])
        assert mw.after_model(state, _make_runtime()) is None

    async def test_async_delegates_to_sync(self) -> None:
        mw = SealGateMiddleware("chart-maker")
        state = _make_state(messages=[AIMessage(content="charts done")])
        result = await mw.aafter_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"
        assert "seal_chart_maker_handoff" in result["messages"][0].content

    def test_report_writer_gated(self) -> None:
        mw = SealGateMiddleware("report-writer")
        state = _make_state(messages=[AIMessage(content="report written")])
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert "seal_report_writer_handoff" in result["messages"][0].content

    def test_reminder_content_positive_no_negation(self) -> None:
        """CLAUDE.md §6: reminder must use positive framing, no '不要/禁止'."""
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[AIMessage(content="done")])
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        content = result["messages"][0].content
        assert "禁止" not in content
        assert "不要" not in content
