"""Tests for GateEnforcementMiddleware.

Follows the same pattern as test_tool_error_handling_middleware.py and
test_sandbox_audit_middleware.py — uses SimpleNamespace for requests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deerflow.agents.middlewares.gate_enforcement_middleware import GateEnforcementMiddleware


def _request(name: str = "task", state: dict | None = None):
    """Build a minimal test request matching the middleware's ToolCallRequest usage."""
    if state is None:
        state = {
            "messages": [],
            "thread_data": {"workspace_path": "/tmp/test-workspace"},
        }
    return SimpleNamespace(tool_call={"name": name, "args": {}, "id": "test_id"}, state=state)


class TestGateEnforcementMiddleware:
    """Test GateEnforcementMiddleware tool call interception."""

    @pytest.fixture
    def mw(self):
        return GateEnforcementMiddleware(enabled=True)

    def test_blocks_task_when_context_missing(self, mw):
        """task() without experiment-context.json → intercepted."""
        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.context_exists",
            return_value=False,
        ):
            req = _request("task")
            handler = MagicMock()
            result = mw.wrap_tool_call(req, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "请先" in result.content

    def test_allows_task_when_context_exists(self, mw):
        """task() with experiment-context.json → passes through."""
        expected = ToolMessage(content="task result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.context_exists",
            return_value=True,
        ):
            req = _request("task")
            result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result is expected

    def test_allows_non_task_tools_always(self, mw):
        """ask_clarification, write_file, bash — always pass through."""
        expected = ToolMessage(content="ok", tool_call_id="test_id")

        for tool_name in ["ask_clarification", "write_file", "bash", "ls", "read_file"]:
            handler = MagicMock(return_value=expected)
            req = _request(tool_name)
            result = mw.wrap_tool_call(req, handler)
            handler.assert_called_once_with(req)
            assert result is expected

    def test_allows_task_when_no_workspace_path(self, mw):
        """If state has no thread_data.workspace_path, allow task() (old thread compat)."""
        expected = ToolMessage(content="task result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request("task", state={"messages": [], "thread_data": {}})
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)

    def test_auto_mode_skips_all_enforcement(self):
        """enabled=False → no enforcement at all."""
        mw = GateEnforcementMiddleware(enabled=False)
        expected = ToolMessage(content="task result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request("task")
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result is expected
