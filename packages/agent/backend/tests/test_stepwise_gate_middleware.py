"""Tests for ``StepwiseGateMiddleware`` (data-flywheel mode pause-after-subagent).

Spec (see middleware docstring): in manual / data-flywheel mode, a ``task(...)``
tool call must be intercepted with ``Command(goto=END)`` whenever the most
recent message-history boundary going backward is a previous ``task``
``ToolMessage`` rather than a ``HumanMessage``. This pauses the run after
every subagent completion so the user can give feedback before the next
dispatch.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def _request(name: str, args: dict[str, Any] | None = None, messages: list[Any] | None = None, tc_id: str = "tc-1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": tc_id, "type": "tool_call"},
        tool=None,
        state={"messages": messages or []},
        runtime=MagicMock(),
    )


def _task_tool_msg(call_id: str = "tc-prev") -> ToolMessage:
    return ToolMessage(content="Task Succeeded. Result: handoff written.", tool_call_id=call_id, name="task")


def _human(text: str = "go") -> HumanMessage:
    return HumanMessage(content=text)


def _ai_with_task(call_id: str = "tc-new", subagent: str = "data-analyst") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "id": call_id, "args": {"subagent_type": subagent, "prompt": "go", "description": "x"}}],
        id="ai-new",
    )


def _ai_with_subagent_task_call(call_id: str = "tc-prev") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "id": call_id, "args": {"subagent_type": "code-executor", "prompt": "go", "description": "x"}}],
        id="ai-prev",
    )


# ---------------------------------------------------------------------------
# Pass-through cases
# ---------------------------------------------------------------------------


def test_disabled_passes_through():
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="tc-1")

    mw = StepwiseGateMiddleware(enabled=False)
    history = [_human(), _ai_with_subagent_task_call(), _task_tool_msg(), _ai_with_task()]
    mw.wrap_tool_call(_request("task", messages=history), handler)
    assert captured.get("called") is True, "Disabled middleware must not intercept"


def test_non_task_tool_passes_through():
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="rf-1")

    mw = StepwiseGateMiddleware(enabled=True)
    history = [_human(), _ai_with_subagent_task_call(), _task_tool_msg()]
    mw.wrap_tool_call(_request("read_file", args={"path": "/x"}, messages=history, tc_id="rf-1"), handler)
    assert captured.get("called") is True, "Non-task tools (read_file etc.) must never be gated"


def test_first_task_after_human_message_passes_through():
    """User's brand-new turn — the very first task() of the turn is allowed."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="tc-new")

    mw = StepwiseGateMiddleware(enabled=True)
    # Human → AI(task) → that AI is the one invoking task right now
    history = [_human("分析 EPM"), _ai_with_task()]
    mw.wrap_tool_call(_request("task", messages=history), handler)
    assert captured.get("called") is True


def test_first_ever_task_passes_through():
    """Empty / first-dispatch state: no prior task ToolMessage, no Human boundary above it
    other than maybe the initial user turn — still must pass through."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="tc-new")

    mw = StepwiseGateMiddleware(enabled=True)
    mw.wrap_tool_call(_request("task", messages=[_human(), _ai_with_task()]), handler)
    assert captured.get("called") is True


def test_empty_messages_passes_through():
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="tc-1")

    mw = StepwiseGateMiddleware(enabled=True)
    mw.wrap_tool_call(_request("task", messages=[]), handler)
    assert captured.get("called") is True


# ---------------------------------------------------------------------------
# Gate-pause cases
# ---------------------------------------------------------------------------


def test_task_after_prior_task_toolmessage_is_paused():
    """The exact data-flywheel scenario:
    Human → AI(task) → ToolMessage(task done) → AI(task) ← this one must pause."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    handler_called = {"value": False}

    def handler(req):
        handler_called["value"] = True
        return ToolMessage(content="ok", tool_call_id="tc-new")

    mw = StepwiseGateMiddleware(enabled=True)
    history = [_human("分析"), _ai_with_subagent_task_call(), _task_tool_msg(), _ai_with_task()]
    result = mw.wrap_tool_call(_request("task", messages=history), handler)
    assert handler_called["value"] is False, "Handler must NOT be called when paused"
    assert isinstance(result, Command), "Must return Command(goto=END)"
    assert result.goto == END
    new_msgs = result.update["messages"]
    assert len(new_msgs) == 1
    assert isinstance(new_msgs[0], ToolMessage)
    assert new_msgs[0].name == "task"
    assert new_msgs[0].tool_call_id == "tc-1"
    assert "gate" in new_msgs[0].content.lower() or "暂停" in new_msgs[0].content


def test_human_message_after_prior_task_resets_gate():
    """If a HumanMessage sits between the last task ToolMessage and the new
    task call, it's a fresh user turn — do NOT pause."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    captured: dict[str, Any] = {}

    def handler(req):
        captured["called"] = True
        return ToolMessage(content="ok", tool_call_id="tc-new")

    mw = StepwiseGateMiddleware(enabled=True)
    history = [
        _human("分析"),
        _ai_with_subagent_task_call(),
        _task_tool_msg(),
        AIMessage(content="指标算完,出图?", id="ai-mid"),  # lead's narrative reply (paused, then resumed)
        _human("出图"),  # user came back, sent new message
        _ai_with_task(),  # new task in the new turn
    ]
    mw.wrap_tool_call(_request("task", messages=history), handler)
    assert captured.get("called") is True, "Post-Human task must be allowed (user explicitly continued)"


def test_async_path_pauses_same_way():
    """awrap_tool_call mirrors wrap_tool_call."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    async def handler(req):
        return ToolMessage(content="ok", tool_call_id="tc-new")

    mw = StepwiseGateMiddleware(enabled=True)
    history = [_human(), _ai_with_subagent_task_call(), _task_tool_msg(), _ai_with_task()]
    result = asyncio.run(mw.awrap_tool_call(_request("task", messages=history), handler))
    assert isinstance(result, Command)
    assert result.goto == END


def test_multiple_consecutive_tasks_each_pause():
    """After resuming (user sent new msg), if lead chains tasks again,
    the second of the new chain must pause too."""
    from deerflow.agents.middlewares.stepwise_gate_middleware import StepwiseGateMiddleware

    handler_called = {"value": False}

    def handler(req):
        handler_called["value"] = True
        return ToolMessage(content="ok", tool_call_id="tc-newer")

    mw = StepwiseGateMiddleware(enabled=True)
    history = [
        _human("分析"),
        _ai_with_subagent_task_call("tc-1"),
        _task_tool_msg("tc-1"),
        AIMessage(content="OK", id="ai-narrative"),
        _human("继续"),
        _ai_with_subagent_task_call("tc-2"),
        _task_tool_msg("tc-2"),
        _ai_with_task("tc-3"),  # second consecutive task in the second turn — pause
    ]
    result = mw.wrap_tool_call(_request("task", messages=history, tc_id="tc-3"), handler)
    assert handler_called["value"] is False
    assert isinstance(result, Command)
