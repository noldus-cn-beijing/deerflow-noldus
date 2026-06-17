"""Tests for loop-detection tool-semantics fix (Spec P6, 2026-06-17).

Reproduces the 2026-06-17 EPM dogfood failure: ``write_todos`` (a bookkeeping
tool) called 5 times across a legitimate long E2E (code → data → chart → report)
tripped the per-tool frequency hard limit, and the resulting FORCED STOP stripped
*every* tool_call in the message — including the ``task(report-writer)`` call that
was about to dispatch the report. The report could never be dispatched.

Three fixes are exercised here (照红线四正模式 1/2/3):

1. Bookkeeping / orchestration tools (``write_todos``, ``task``) carry generous
   frequency overrides so a legitimate long E2E does not trip them.
2. When a frequency hard limit *does* fire, only the offending tool's call is
   stripped — sibling calls (``task``/``seal``/``ask_clarification``) survive so
   the flow keeps advancing.
3. The FORCED STOP message is actionable (names the surviving calls).

Real-loop protection (identical-call hash detection, and bash frequency) is
asserted unchanged so the fix cannot weaken the safety net.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from deerflow.agents.middlewares.loop_detection_middleware import (
    LoopDetectionMiddleware,
)


def _make_runtime(thread_id="test-thread", run_id="test-run"):
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id, "run_id": run_id}
    return runtime


def _ai_state(tool_calls, content=""):
    return {"messages": [AIMessage(content=content, tool_calls=tool_calls)]}


def _task_call(subagent_type, idx=0):
    return {
        "name": "task",
        "id": f"task_{idx}",
        "args": {"subagent_type": subagent_type, "description": f"do {subagent_type} #{idx}"},
    }


def _write_todos_call(idx=0):
    return {
        "name": "write_todos",
        "id": f"todos_{idx}",
        "args": {"todos": [{"id": idx, "content": f"step {idx}", "status": "in_progress"}]},
    }


def _bash_call(command, idx=0):
    return {"name": "bash", "id": f"bash_{idx}", "args": {"command": command}}


# ---------------------------------------------------------------------------
# Spec 2.1 — bookkeeping/orchestration tools have generous overrides by default
# ---------------------------------------------------------------------------

class TestBookkeepingOverrideDefaults:
    """write_todos / task must carry generous frequency thresholds out of the box
    so a long E2E does not trip FORCED STOP (正模式 1)."""

    def test_write_todos_six_legitimate_calls_not_tripped(self):
        """Six distinct write_todos updates across a long E2E — each with different
        args, never identical — must NOT trip frequency hard stop with the default
        write_todos override.

        Before the fix: the bare LoopDetectionMiddleware (no override) tripped at
        call 5 because write_todos had no exemption. This is the 2026-06-17 dogfood
        failure.
        """
        mw = LoopDetectionMiddleware.with_semantic_defaults()
        runtime = _make_runtime()

        for i in range(6):
            warning, hard_stop, _ = mw._track_and_check(_ai_state([_write_todos_call(i)]), runtime)
            assert not hard_stop, (
                f"write_todos call {i + 1}: legitimate bookkeeping update must not trip FORCED STOP"
            )

    def test_task_four_legitimate_dispatches_not_tripped(self):
        """One E2E legitimately dispatches code-executor → data-analyst →
        chart-maker → report-writer (4 distinct subagents). That must NOT trip
        the frequency limit on `task`."""
        mw = LoopDetectionMiddleware.with_semantic_defaults()
        runtime = _make_runtime()

        for subagent in ["code-executor", "data-analyst", "chart-maker", "report-writer"]:
            warning, hard_stop, _ = mw._track_and_check(_ai_state([_task_call(subagent)]), runtime)
            assert not hard_stop, (
                f"task({subagent}): legitimate E2E dispatch must not trip FORCED STOP"
            )


# ---------------------------------------------------------------------------
# Spec 2.2 — hard stop strips only the offending tool, preserves siblings
# ---------------------------------------------------------------------------

class TestStripOnlyOffendingTool:
    """When write_todos trips the frequency hard limit, sibling tool_calls in the
    same message (e.g. task(report-writer)) must survive (正模式 2)."""

    def test_write_todos_over_limit_preserves_sibling_task(self):
        """Message = [write_todos (over limit), task(report-writer)].
        After _apply: write_todos call stripped, task call preserved."""
        # Construct a middleware where write_todos trips at hard_limit=2 so we
        # can reach the strip path with a sibling in the same message.
        mw = LoopDetectionMiddleware(
            tool_freq_warn=1,
            tool_freq_hard_limit=2,
            tool_freq_overrides={"write_todos": (1, 2)},
        )
        runtime = _make_runtime()

        # Two prior write_todos calls to bring the counter to the limit.
        mw._apply(_ai_state([_write_todos_call(0)]), runtime)
        mw._apply(_ai_state([_write_todos_call(1)]), runtime)

        # Now a message that contains BOTH write_todos (3rd → over limit) AND
        # a task(report-writer) call that should advance the flow.
        combined = [_write_todos_call(2), _task_call("report-writer", idx=10)]
        result = mw._apply(_ai_state(combined), runtime)

        assert result is not None, "Frequency hard stop should fire for write_todos"
        msgs = result["messages"]
        assert len(msgs) == 1
        stripped = msgs[0]
        surviving_names = [tc["name"] for tc in stripped.tool_calls]
        assert "task" in surviving_names, (
            "task(report-writer) must survive the partial strip — flow must keep advancing"
        )
        assert "write_todos" not in surviving_names, (
            "write_todos (the offending tool) must be stripped"
        )
        # task call identity preserved (subagent_type intact)
        task_calls = [tc for tc in stripped.tool_calls if tc["name"] == "task"]
        assert len(task_calls) == 1
        assert task_calls[0]["args"]["subagent_type"] == "report-writer"

    def test_strip_preserves_multiple_siblings(self):
        """Message = [write_todos (over), task(data-analyst), seal].
        Only write_todos is stripped; task + seal survive."""
        mw = LoopDetectionMiddleware(
            tool_freq_warn=1,
            tool_freq_hard_limit=2,
            tool_freq_overrides={"write_todos": (1, 2)},
        )
        runtime = _make_runtime()
        mw._apply(_ai_state([_write_todos_call(0)]), runtime)
        mw._apply(_ai_state([_write_todos_call(1)]), runtime)

        combined = [
            _write_todos_call(2),
            _task_call("data-analyst", idx=20),
            {"name": "seal", "id": "seal_1", "args": {}},
        ]
        result = mw._apply(_ai_state(combined), runtime)
        assert result is not None
        surviving_names = [tc["name"] for tc in result["messages"][0].tool_calls]
        assert "write_todos" not in surviving_names
        assert "task" in surviving_names
        assert "seal" in surviving_names

    def test_single_offending_tool_strips_to_empty(self):
        """When the message contains only the offending tool, partial strip
        yields an empty tool_calls list (same as the old behaviour for that case)
        — finish_reason flipped to stop, no dangling tool call created."""
        mw = LoopDetectionMiddleware(
            warn_threshold=99,  # avoid hash-based path
            hard_limit=99,
            tool_freq_warn=1,
            tool_freq_hard_limit=2,
        )
        runtime = _make_runtime()
        mw._apply(_ai_state([_bash_call("ls")]), runtime)

        result = mw._apply(_ai_state([_bash_call("ls")]), runtime)
        assert result is not None
        stripped = result["messages"][0]
        assert stripped.tool_calls == []
        # No dangling tool_calls in additional_kwargs either
        assert not (stripped.additional_kwargs or {}).get("tool_calls")


# ---------------------------------------------------------------------------
# Spec 3 — protection on real loops is unchanged
# ---------------------------------------------------------------------------

class TestRealLoopStillCaught:
    """The fix must not weaken real-loop protection."""

    def test_identical_bash_calls_still_hard_stop(self):
        """The same bash command, repeated past the hash-based hard_limit, still
        trips FORCED STOP (hash detection unchanged)."""
        mw = LoopDetectionMiddleware(warn_threshold=2, hard_limit=4)
        runtime = _make_runtime()
        call = [_bash_call("ls -la")]

        for _ in range(3):
            mw._apply(_ai_state(call), runtime)

        result = mw._apply(_ai_state(call), runtime)
        assert result is not None
        assert result["messages"][0].tool_calls == []

    def test_bash_frequency_over_global_limit_still_stops(self):
        """Distinct bash commands past the global frequency hard_limit still trip
        FORCED STOP — only bookkeeping tools get the lenient override."""
        mw = LoopDetectionMiddleware(
            tool_freq_warn=2,
            tool_freq_hard_limit=4,
        )
        runtime = _make_runtime()

        for i in range(3):
            mw._apply(_ai_state([_bash_call(f"cmd_{i}")]), runtime)

        result = mw._apply(_ai_state([_bash_call("cmd_4")]), runtime)
        assert result is not None, "bash frequency over global limit must still hard stop"


# ---------------------------------------------------------------------------
# Spec 3 — full legitimate long E2E sequence completes
# ---------------------------------------------------------------------------

class TestLongE2ESequenceCompletes:
    """A full legitimate E2E tool sequence (code → data → chart → report), each
    step updating todos, must not trip FORCED STOP at any point."""

    def test_full_e2e_no_forced_stop(self):
        mw = LoopDetectionMiddleware.with_semantic_defaults()
        runtime = _make_runtime()

        # Simulate: dispatch each subagent + update todos between steps.
        sequence = []
        for sub in ["code-executor", "data-analyst", "chart-maker", "report-writer"]:
            sequence.append((_task_call(sub),))
            sequence.append((_write_todos_call(len(sequence)),))

        forced = False
        for calls in sequence:
            _, hard_stop, _ = mw._track_and_check(_ai_state(list(calls)), runtime)
            if hard_stop:
                forced = True
                break

        assert not forced, (
            "Legitimate long E2E sequence (4 task dispatches + 4 todo updates) must "
            "not trip FORCED STOP with semantic defaults"
        )
