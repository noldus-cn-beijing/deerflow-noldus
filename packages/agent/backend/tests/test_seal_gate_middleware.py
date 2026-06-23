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
        # data-analyst 的封口入口是 finalize（spec 2026-06-23-data-analyst-seal-stepwise-fill-template）
        assert "finalize_data_analyst_handoff" in msg.content
        assert msg.additional_kwargs.get("hide_from_ui") is True

    def test_allow_last_ai_has_seal_tool_call(self) -> None:
        # data-analyst 封口入口 = finalize（spec 2026-06-23-...-fill-template）
        mw = SealGateMiddleware("data-analyst")
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "finalize_data_analyst_handoff", "args": {}, "id": "tc1"}],
        )
        state = _make_state(messages=[ai])
        assert mw.after_model(state, _make_runtime()) is None

    def test_allow_history_has_seal_tool_message(self) -> None:
        # data-analyst 封口入口 = finalize（spec 2026-06-23-...-fill-template）
        mw = SealGateMiddleware("data-analyst")
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "finalize_data_analyst_handoff", "args": {}, "id": "tc1"}],
        )
        tm = ToolMessage(content="ok", tool_call_id="tc1", name="finalize_data_analyst_handoff")
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

    def test_reminder_count_accumulates_across_drifting_runtimes(self) -> None:
        """Regression for the 2026-06-18 production bug (4th EPM dogfood).

        The OLD code keyed the reminder count on
        ``getattr(runtime, "run_id", None) or id(runtime)``. In production the
        ``runtime`` object carries NO ``run_id`` ATTRIBUTE (run_id lives in the
        FLAT ``runtime.context`` dict), so the key fell back to ``id(runtime)``,
        which DIFFERS on each after_model invocation within one run. The count
        never accumulated → cap never tripped → the gate bounced the model
        every turn (capped only by the outer max_turns), burning ~9 wasted
        re-judgement turns before seal-resume caught it.

        Here we feed a DIFFERENT runtime object each call (no run_id attr) —
        exactly what production does — and assert the per-instance counter still
        reaches the cap. With the old keyed-dict code this looped forever (every
        call returned a reminder); with the per-instance int it caps at
        _MAX_REMINDERS.
        """
        mw = SealGateMiddleware("data-analyst")
        state = _make_state(messages=[AIMessage(content="no seal")])
        # Fresh runtime objects WITHOUT a run_id attribute — mirrors prod, where
        # id(runtime) drifts per turn and run_id is absent as an attribute.
        results = [mw.after_model(state, object()) for _ in range(5)]
        # Exactly _MAX_REMINDERS reminders fire, then the gate allows exit.
        fired = [r for r in results if r is not None]
        allowed = [r for r in results if r is None]
        assert len(fired) == 2, f"expected exactly 2 reminders, got {len(fired)}"
        assert len(allowed) == 3, "after cap, every further call must allow exit"
        # The cap is hit on the 3rd call and stays allowed thereafter (no relapse).
        assert results[2] is None and results[3] is None and results[4] is None

    def test_reminder_count_isolated_per_instance(self) -> None:
        """Per-run isolation is achieved by FRESH instances, not by run-id keying.

        The subagent executor builds a fresh SealGateMiddleware per run
        (executor.py:_build_middlewares, called per run — same "fresh instance
        each call" pattern as LoopDetectionMiddleware right above it). So two
        independent runs get two independent counters automatically; a new
        instance starts with a full reminder budget regardless of how many
        reminders a previous run's instance fired.
        """
        state = _make_state(messages=[AIMessage(content="no seal")])
        mw_run_a = SealGateMiddleware("data-analyst")
        assert mw_run_a.after_model(state, object()) is not None
        assert mw_run_a.after_model(state, object()) is not None
        assert mw_run_a.after_model(state, object()) is None  # run-a capped
        # A fresh instance (= a new run) starts over with a full budget.
        mw_run_b = SealGateMiddleware("data-analyst")
        assert mw_run_b.after_model(state, object()) is not None
        assert mw_run_b.after_model(state, object()) is not None
        assert mw_run_b.after_model(state, object()) is None  # run-b capped

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
