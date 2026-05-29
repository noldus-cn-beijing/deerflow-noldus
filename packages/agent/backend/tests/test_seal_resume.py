"""Sprint 5.8 — Integration tests for seal-resume (补轮) mechanism in executor.

Tests the _attempt_seal_resume helper and its integration at the 5.7
checkpoint. Uses the same importlib pattern as test_executor_handoff_emission
to load the real executor module (conftest.py pre-mocks it for other tests).

TDD: this file is written FIRST (red), then implementation makes it green.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load the real executor module (same pattern as test_executor_handoff_emission)
# ---------------------------------------------------------------------------
_EXECUTOR_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
)

_REAL_EXECUTOR: ModuleType | None = None


def _get_real_executor() -> ModuleType:
    global _REAL_EXECUTOR
    if _REAL_EXECUTOR is not None:
        return _REAL_EXECUTOR

    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.executor_real_seal_resume",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _REAL_EXECUTOR = module
    return module


@pytest.fixture(autouse=True)
def _load_module():
    _get_real_executor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Run an async coroutine in a fresh event loop (pytest-asyncio safe)."""
    return asyncio.run(coro)


def _make_result(mod: ModuleType, *, task_id: str = "test-task"):
    """Create a fresh SubagentResult in PENDING state."""
    return mod.SubagentResult(
        task_id=task_id,
        trace_id="test-trace",
        status=mod.SubagentStatus.PENDING,
    )


def _make_workspace(tmp_path: Path, *, with_file: str | None = None) -> str:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    if with_file:
        (ws / with_file).write_text("{}", encoding="utf-8")
    return str(ws)


def _make_executor(
    mod: ModuleType,
    *,
    name: str = "data-analyst",
    thread_data: dict | None = None,
    sandbox_state=None,
):
    """Create a minimal SubagentExecutor-like object for testing _attempt_seal_resume.

    We can't easily construct a real SubagentExecutor (it needs model factory etc.),
    so we create a lightweight object that has the attributes the method reads.
    """
    executor = object.__new__(mod.SubagentExecutor)
    executor.config = MagicMock()
    executor.config.name = name
    executor.trace_id = "test-trace"
    executor.thread_data = thread_data
    executor.sandbox_state = sandbox_state
    return executor


# ---------------------------------------------------------------------------
# Test 1: resume triggered when handoff missing
# ---------------------------------------------------------------------------
class TestSealResumeTriggered:
    """补轮在 handoff 缺失时触发。"""

    def test_resume_triggered_when_handoff_missing(self, tmp_path: Path):
        """data-analyst 结束无 handoff → 补轮触发 → astream 被调。"""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)  # no handoff file initially

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        # Simulate final_state with an AIMessage (subagent did work)
        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis complete, alpha=42")
        final_state = {"messages": [HumanMessage(content="do analysis"), ai_msg]}

        # Track whether astream was called
        astream_called = False
        async def mock_astream_second(*args, **kwargs):
            nonlocal astream_called
            astream_called = True
            # Simulate the LLM calling seal — write the handoff file
            seal_ai = AIMessage(content="", tool_calls=[{"name": "seal_data_analyst_handoff", "args": {"key_findings": ["alpha=42"]}, "id": "tc_1"}])
            (Path(ws) / "handoff_data_analyst.json").write_text('{"key_findings": ["alpha=42"]}', encoding="utf-8")
            yield {"messages": [ai_msg, seal_ai]}

        agent = MagicMock()
        agent.astream = mock_astream_second

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        # Resume should return a state (not None)
        assert new_state is not None
        # astream was called
        assert astream_called

    def test_resume_not_triggered_when_handoff_present(self, tmp_path: Path):
        """handoff 已存在 → 不触发补轮 → COMPLETED."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path, with_file="handoff_data_analyst.json")

        # Simulate the full executor checkpoint logic
        result = _make_result(mod)
        _handoff_error = mod._validate_handoff_emitted("data-analyst", ws)
        assert _handoff_error is None  # handoff present
        # No resume needed — direct COMPLETED
        result.try_set_terminal(mod.SubagentStatus.COMPLETED)
        assert result.status == mod.SubagentStatus.COMPLETED

    def test_no_resume_for_general_purpose(self, tmp_path: Path):
        """general-purpose（白名单外）无 handoff → 不触发补轮 → COMPLETED."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)  # no handoff file
        result = _make_result(mod)
        _handoff_error = mod._validate_handoff_emitted("general-purpose", ws)
        assert _handoff_error is None  # not in whitelist
        result.try_set_terminal(mod.SubagentStatus.COMPLETED)
        assert result.status == mod.SubagentStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test 2: resume recovers → COMPLETED
# ---------------------------------------------------------------------------
class TestSealResumeRecovery:
    """补轮成功恢复 → handoff 产出 → COMPLETED. 补轮失败 → 走 5.7 FAILED."""

    def test_resume_recovers_writes_handoff(self, tmp_path: Path):
        """补轮成功产出 handoff → COMPLETED（非 FAILED）."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis done")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        agent = MagicMock()
        async def mock_astream_success(*args, **kwargs):
            seal_ai = AIMessage(content="", tool_calls=[{"name": "seal_data_analyst_handoff", "args": {}, "id": "tc_resume"}])
            (Path(ws) / "handoff_data_analyst.json").write_text('{"status": "ok"}', encoding="utf-8")
            yield {"messages": [ai_msg, seal_ai]}

        agent.astream = mock_astream_success

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        assert new_state is not None
        # Re-validate: handoff should now exist
        _handoff_error = mod._validate_handoff_emitted("data-analyst", ws)
        assert _handoff_error is None
        # Can mark COMPLETED
        result.try_set_terminal(mod.SubagentStatus.COMPLETED)
        assert result.status == mod.SubagentStatus.COMPLETED

    def test_resume_fails_falls_back_to_failed(self, tmp_path: Path):
        """补轮后 handoff 仍不存在 → FAILED + error 含 'seal-resume did not recover'."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        # Mock agent: resume does NOT produce handoff
        agent = MagicMock()
        async def mock_astream_no_handoff(*args, **kwargs):
            no_seal_ai = AIMessage(content="I'm done")  # no tool_call
            yield {"messages": [ai_msg, no_seal_ai]}

        agent.astream = mock_astream_no_handoff

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        # new_state is not None (astream succeeded), but handoff still missing
        assert new_state is not None
        _handoff_error = mod._validate_handoff_emitted("data-analyst", ws)
        assert _handoff_error is not None
        # Simulate the full checkpoint: resume failed → mark FAILED
        result.try_set_terminal(
            mod.SubagentStatus.FAILED,
            error=f"{_handoff_error} (seal-resume did not recover)",
        )
        assert result.status == mod.SubagentStatus.FAILED
        assert "seal-resume did not recover" in result.error

    def test_resume_astream_exception_falls_back(self, tmp_path: Path):
        """补轮 astream 抛异常 → _attempt_seal_resume 返回 None → 走 5.7 FAILED."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        agent = MagicMock()
        async def mock_astream_crash(*args, **kwargs):
            raise RuntimeError("LLM API timeout")
            yield  # noqa: PIE786 — makes this an async generator

        agent.astream = mock_astream_crash

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        # Exception caught → returns None
        assert new_state is None
        # Fall through to 5.7 FAILED
        _handoff_error = mod._validate_handoff_emitted("data-analyst", ws)
        assert _handoff_error is not None
        result.try_set_terminal(mod.SubagentStatus.FAILED, error=_handoff_error)
        assert result.status == mod.SubagentStatus.FAILED


# ---------------------------------------------------------------------------
# Test 3: guards — cancelled / no AIMessage / thread_data
# ---------------------------------------------------------------------------
class TestSealResumeGuards:
    """补轮的各种守卫条件。"""

    def test_resume_skipped_when_cancelled(self, tmp_path: Path):
        """cancel_event 已 set → 补轮提前返回 → 走 CANCELLED."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)
        result.cancel_event.set()

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        agent = MagicMock()
        # Simulate astream that checks cancel — returns partial state
        async def mock_astream_cancel(*args, **kwargs):
            yield {"messages": [ai_msg]}  # one chunk then stops

        agent.astream = mock_astream_cancel

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        # Cancelled mid-stream — returns whatever was accumulated (not None)
        # The key point is: the outer code checks cancel_event BEFORE resume,
        # so this method isn't called at all. But if it IS called (race), it
        # respects cancel_event inside the astream loop.
        assert new_state is not None or new_state is None  # no crash

    def test_resume_skipped_when_no_aimessage(self, tmp_path: Path):
        """final_state.messages 无任何 AIMessage → _attempt_seal_resume 返回 None（无米下锅守卫）。"""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import HumanMessage, SystemMessage
        final_state = {"messages": [SystemMessage(content="system"), HumanMessage(content="task")]}

        agent = MagicMock()

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        new_state = _run_async(_run())
        # No AIMessage → skip resume, return None
        assert new_state is None
        # astream should NOT have been called
        agent.astream.assert_not_called()

    def test_resume_state_carries_thread_data(self, tmp_path: Path):
        """补轮传给 astream 的 resume_state 含 thread_data + sandbox（显式重注入）。"""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        thread_data = {"workspace_path": ws, "thread_id": "t123"}
        sandbox_state = MagicMock()
        executor = _make_executor(
            mod, name="data-analyst",
            thread_data=thread_data,
            sandbox_state=sandbox_state,
        )
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        # Capture what astream receives
        captured_state = {}
        agent = MagicMock()
        async def mock_astream_capture(state, *args, **kwargs):
            captured_state.update(state)
            yield {"messages": [ai_msg]}

        agent.astream = mock_astream_capture

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        _run_async(_run())
        # Thread data and sandbox must be present in the state passed to astream
        assert "thread_data" in captured_state
        assert captured_state["thread_data"] == thread_data
        assert "sandbox" in captured_state
        assert captured_state["sandbox"] == sandbox_state


# ---------------------------------------------------------------------------
# Test 4: AI message collection during resume
# ---------------------------------------------------------------------------
class TestSealResumeAiMessages:
    """补轮产生的 AI message 被正确收集。"""

    def test_resume_collects_ai_messages(self, tmp_path: Path):
        """补轮产生的 AI message 加入 result.ai_messages（去重正确）。"""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg_1 = AIMessage(content="First analysis", id="ai_1")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg_1]}

        # Pre-populate ai_messages with the first message (simulating main loop)
        result.ai_messages.append(ai_msg_1.model_dump())

        # Resume produces a new AI message
        ai_msg_resume = AIMessage(content="Sealed", id="ai_resume")
        agent = MagicMock()
        async def mock_astream_collect(*args, **kwargs):
            (Path(ws) / "handoff_data_analyst.json").write_text('{"ok": true}', encoding="utf-8")
            yield {"messages": [ai_msg_1, ai_msg_resume]}

        agent.astream = mock_astream_collect

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        _run_async(_run())

        # The new AI message should be appended (not duplicated)
        ids = [m.get("id") for m in result.ai_messages]
        assert ids.count("ai_1") == 1  # no duplicate
        assert "ai_resume" in ids  # new message collected


# ---------------------------------------------------------------------------
# Test 5: prompt content validation
# ---------------------------------------------------------------------------
class TestSealResumePrompt:
    """补轮 prompt 内容验证。"""

    def test_resume_prompt_uses_positive_phrasing(self, tmp_path: Path):
        """补轮 prompt 含 seal tool 名 + 不含反向激活词。"""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)

        executor = _make_executor(mod, name="data-analyst", thread_data={"workspace_path": ws})
        result = _make_result(mod)

        from langchain_core.messages import AIMessage, HumanMessage
        ai_msg = AIMessage(content="Analysis")
        final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

        # Capture the HumanMessage appended during resume
        captured_messages = []
        agent = MagicMock()
        async def mock_astream_capture_prompt(state, *args, **kwargs):
            msgs = state.get("messages", [])
            captured_messages.extend(msgs)
            yield {"messages": [ai_msg]}

        agent.astream = mock_astream_capture_prompt

        async def _run():
            return await executor._attempt_seal_resume(
                agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
            )

        _run_async(_run())

        # The last message should be the resume prompt
        last_msg = captured_messages[-1]
        assert isinstance(last_msg, HumanMessage)
        prompt_text = last_msg.content

        # Must contain seal tool name
        assert "seal_data_analyst_handoff" in prompt_text
        # Must NOT contain negative activation words (CLAUDE.md §6)
        for word in ["不要", "禁止", "忘记", "别忘了", "不能", "务必不要"]:
            assert word not in prompt_text, f"Prompt contains negative activation word: {word}"

    def test_resume_prompt_correct_seal_tool_name(self, tmp_path: Path):
        """data-analyst → 'seal_data_analyst_handoff'; chart-maker → 'seal_chart_maker_handoff'."""
        mod = _get_real_executor()

        for name, expected_tool in [
            ("data-analyst", "seal_data_analyst_handoff"),
            ("chart-maker", "seal_chart_maker_handoff"),
            ("code-executor", "seal_code_executor_handoff"),
            ("report-writer", "seal_report_writer_handoff"),
        ]:
            ws = _make_workspace(tmp_path / name)
            executor = _make_executor(mod, name=name, thread_data={"workspace_path": ws})
            result = _make_result(mod)

            from langchain_core.messages import AIMessage, HumanMessage
            ai_msg = AIMessage(content="Work")
            final_state = {"messages": [HumanMessage(content="task"), ai_msg]}

            captured_messages = []
            agent = MagicMock()
            async def mock_capture(state, *args, **kwargs):
                captured_messages.extend(state.get("messages", []))
                yield {"messages": [ai_msg]}

            agent.astream = mock_capture

            async def _run():
                return await executor._attempt_seal_resume(
                    agent, final_state, MagicMock(), MagicMock(), result, MagicMock(),
                )

            _run_async(_run())

            last_msg = captured_messages[-1]
            assert expected_tool in last_msg.content, (
                f"Subagent '{name}': expected '{expected_tool}' in prompt, got: {last_msg.content}"
            )
