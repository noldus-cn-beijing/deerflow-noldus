"""Regression test for the W18/W19 ordering bug discovered in dogfood thread
51b00ac8 (2026-05-19):

W18 ``TaskHandoffAuthorizationProvider`` checks ``tool_input.prompt`` for the
required ``{{handoff://X}}`` placeholders BEFORE ``task_tool`` runs W19's
``_auto_inject_handoff_placeholders``. The result: a lead that hand-writes a
``task`` call without the placeholder is *always* denied, even though W19
would have rescued it. Lead then loops on read_file / re-dispatch until the
LoopDetection / ForcedStop circuit breaker kicks in.

This test asserts that the lead's ``task`` tool call is rewritten BEFORE
GuardrailMiddleware evaluates it — i.e. the auto-injection lives at the
middleware layer (HandoffPlaceholderInjectionMiddleware), not inside
task_tool's own body.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest


@pytest.fixture
def middleware():
    """Import lazily so unrelated test runs aren't penalized by deerflow init cost."""
    from deerflow.agents.middlewares.handoff_placeholder_injection_middleware import (
        HandoffPlaceholderInjectionMiddleware,
    )

    return HandoffPlaceholderInjectionMiddleware()


def _make_request(tool_name: str, args: dict[str, Any]) -> ToolCallRequest:
    """Build a minimal ToolCallRequest matching what ToolNode produces."""
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": "tc-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=MagicMock(),
    )


def test_non_task_tool_passes_through_unchanged(middleware):
    """ls/bash/read_file etc. must not be touched."""
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["args"] = req.tool_call.get("args")
        return ToolMessage(content="ok", tool_call_id="tc-1")

    request = _make_request("ls", {"path": "/mnt"})
    middleware.wrap_tool_call(request, handler)
    assert captured["args"] == {"path": "/mnt"}


def test_task_with_no_required_handoffs_unchanged(middleware):
    """code-executor.required_upstream_handoffs == [] → no injection."""
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    original_prompt = "跑 EPM 流水线,算开臂时间比"
    request = _make_request(
        "task",
        {
            "subagent_type": "code-executor",
            "prompt": original_prompt,
            "description": "calc",
        },
    )
    middleware.wrap_tool_call(request, handler)
    assert captured["prompt"] == original_prompt, "no required handoff → prompt unchanged"


def test_task_with_missing_required_handoff_gets_injected(middleware):
    """chart-maker.required_upstream_handoffs == ['code_executor'].

    If lead's prompt does NOT contain {{handoff://code_executor}}, middleware
    must inject it BEFORE handler is called, so the downstream Guardrail
    (TaskHandoffAuthorizationProvider) sees the injected prompt.
    """
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        captured["subagent_type"] = req.tool_call["args"]["subagent_type"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    request = _make_request(
        "task",
        {
            "subagent_type": "chart-maker",
            "prompt": "出 EPM 轨迹图 + 开臂时间柱状图",
            "description": "plot",
        },
    )
    middleware.wrap_tool_call(request, handler)
    # Original lead-written prose preserved
    assert "出 EPM 轨迹图" in captured["prompt"]
    # Handoff placeholder injected (subagent_type unchanged)
    assert "{{handoff://code_executor}}" in captured["prompt"]
    assert captured["subagent_type"] == "chart-maker"


def test_task_with_handoff_already_present_not_double_injected(middleware):
    """Idempotent: if lead already wrote the placeholder, do not duplicate."""
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    prompt_with_placeholder = (
        "请基于 {{handoff://code_executor}} 出图\n\n"
        "需要轨迹图 + 指标柱状图"
    )
    request = _make_request(
        "task",
        {
            "subagent_type": "chart-maker",
            "prompt": prompt_with_placeholder,
            "description": "plot",
        },
    )
    middleware.wrap_tool_call(request, handler)
    # Exactly one occurrence
    assert captured["prompt"].count("{{handoff://code_executor}}") == 1


def test_task_with_multiple_required_handoffs_all_injected(middleware):
    """report-writer.required_upstream_handoffs == ['code_executor', 'data_analyst']."""
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    request = _make_request(
        "task",
        {
            "subagent_type": "report-writer",
            "prompt": "写一份完整的 EPM 实验报告",
            "description": "report",
        },
    )
    middleware.wrap_tool_call(request, handler)
    assert "{{handoff://code_executor}}" in captured["prompt"]
    assert "{{handoff://data_analyst}}" in captured["prompt"]


def test_unknown_subagent_type_passes_through(middleware):
    """If lead writes a typo'd subagent_type, no injection; let downstream layers
    handle (e.g., task_tool will surface a clear error)."""
    captured: dict[str, Any] = {}

    def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    request = _make_request(
        "task",
        {
            "subagent_type": "nonexistent-agent",
            "prompt": "do something",
            "description": "x",
        },
    )
    middleware.wrap_tool_call(request, handler)
    assert captured["prompt"] == "do something"


def test_handler_receives_overridden_request_not_mutated_original(middleware):
    """Per langchain docs: use request.override(), do not mutate. Verify the
    middleware uses the immutable pattern so original is preserved (in case
    upstream middleware retains a reference)."""
    request = _make_request(
        "task",
        {
            "subagent_type": "chart-maker",
            "prompt": "plot",
            "description": "x",
        },
    )
    original_args = request.tool_call["args"]

    def handler(req: ToolCallRequest) -> ToolMessage:
        # The request handed to handler MUST have injected prompt
        assert "{{handoff://code_executor}}" in req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    middleware.wrap_tool_call(request, handler)
    # Original args object is untouched (no surprise side-effects)
    assert original_args["prompt"] == "plot", "original tool_call args mutated — must use override()"


def test_async_path_same_behavior(middleware):
    """awrap_tool_call must mirror wrap_tool_call for async tool execution."""
    captured: dict[str, Any] = {}

    async def handler(req: ToolCallRequest) -> ToolMessage:
        captured["prompt"] = req.tool_call["args"]["prompt"]
        return ToolMessage(content="ok", tool_call_id="tc-1")

    request = _make_request(
        "task",
        {
            "subagent_type": "chart-maker",
            "prompt": "plot",
            "description": "x",
        },
    )
    asyncio.run(middleware.awrap_tool_call(request, handler))
    assert "{{handoff://code_executor}}" in captured["prompt"]
