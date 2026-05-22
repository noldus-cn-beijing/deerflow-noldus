"""Tests for GateEnforcementMiddleware.

Follows the same pattern as test_tool_error_handling_middleware.py and
test_sandbox_audit_middleware.py — uses SimpleNamespace for requests.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from deerflow.agents.middlewares.gate_enforcement_middleware import GateEnforcementMiddleware


def _request(name: str = "task", state: dict | None = None, args: dict | None = None):
    """Build a minimal test request matching the middleware's ToolCallRequest usage."""
    if state is None:
        state = {
            "messages": [],
            "thread_data": {"workspace_path": "/tmp/test-workspace"},
        }
    if args is None:
        args = {}
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": "test_id"}, state=state)


def _request_data_analyst(state: dict | None = None):
    """Build a task(data-analyst) request."""
    return _request("task", state=state, args={"subagent_type": "data-analyst"})


class TestGate1:
    """Gate 1: paradigm confirmation enforcement."""

    @pytest.fixture
    def mw(self):
        return GateEnforcementMiddleware(enabled=True)

    def test_blocks_when_no_context(self, mw):
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
        assert "实验范式尚未确认" in result.content

    def test_allows_when_context_exists(self, mw):
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

    def test_allows_when_no_workspace_path(self, mw):
        """If state has no thread_data.workspace_path, allow task() (old thread compat)."""
        expected = ToolMessage(content="task result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request("task", state={"messages": [], "thread_data": {}})
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)


class TestGate2:
    """Gate 2: data quality enforcement (task(data-analyst) only)."""

    @pytest.fixture
    def mw(self):
        return GateEnforcementMiddleware(enabled=True)

    def test_blocks_when_critical_unacknowledged(self, mw):
        """task(data-analyst) with critical warnings, not acknowledged → intercepted."""
        with (
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.get_critical_warnings",
                return_value=[{"severity": "critical", "message": "trajectory gap > 10%: Subject 3"}],
            ),
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.is_quality_acknowledged",
                return_value=False,
            ),
        ):
            req = _request_data_analyst()
            handler = MagicMock()
            result = mw.wrap_tool_call(req, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "数据质量检查" in result.content
        assert "critical" in result.content
        assert "trajectory gap" in result.content
        assert "acknowledge_quality=True" in result.content

    def test_allows_when_no_critical(self, mw):
        """task(data-analyst) with no critical warnings → passes through."""
        expected = ToolMessage(content="analyst result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.get_critical_warnings",
            return_value=[],
        ):
            req = _request_data_analyst()
            result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result is expected

    def test_allows_when_acknowledged(self, mw):
        """task(data-analyst) with critical warnings but already acknowledged → passes through."""
        expected = ToolMessage(content="analyst result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        with (
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.get_critical_warnings",
                return_value=[{"severity": "critical", "message": "trajectory gap"}],
            ),
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.is_quality_acknowledged",
                return_value=True,
            ),
        ):
            req = _request_data_analyst()
            result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result is expected

    def test_allows_when_missing_workspace_path(self, mw):
        """task(data-analyst) with no workspace_path → passes through (fail open)."""
        expected = ToolMessage(content="analyst result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request_data_analyst(state={"messages": [], "thread_data": {}})
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)


class TestNonTaskPassthrough:
    """Non-task tools always pass through."""

    @pytest.fixture
    def mw(self):
        return GateEnforcementMiddleware(enabled=True)

    def test_non_task_tools_pass_through(self, mw):
        """ask_clarification, write_file, bash — always pass through."""
        expected = ToolMessage(content="ok", tool_call_id="test_id")

        for tool_name in ["ask_clarification", "write_file", "bash", "ls", "read_file"]:
            handler = MagicMock(return_value=expected)
            req = _request(tool_name)
            result = mw.wrap_tool_call(req, handler)
            handler.assert_called_once_with(req)
            assert result is expected


class TestDisabledMiddleware:
    """When enabled=False, everything passes through."""

    def test_disabled_passes_through(self):
        """enabled=False → no enforcement at all."""
        mw = GateEnforcementMiddleware(enabled=False)
        expected = ToolMessage(content="task result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request("task")
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)
        assert result is expected

    def test_disabled_data_analyst_passes_through(self):
        """enabled=False for data-analyst too."""
        mw = GateEnforcementMiddleware(enabled=False)
        expected = ToolMessage(content="analyst result", tool_call_id="test_id")
        handler = MagicMock(return_value=expected)

        req = _request_data_analyst()
        result = mw.wrap_tool_call(req, handler)

        handler.assert_called_once_with(req)


class TestGateCheckLogging:
    """Verify structured gate_check logging."""

    @pytest.fixture
    def mw(self):
        return GateEnforcementMiddleware(enabled=True)

    def test_gate1_block_logs(self, mw, caplog):
        """Gate 1 blocked → structured log emitted."""
        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.context_exists",
            return_value=False,
        ):
            req = _request("task")
            handler = MagicMock()

            with caplog.at_level(logging.INFO):
                mw.wrap_tool_call(req, handler)

        assert any("gate_check" in r.message and "gate=gate1_paradigm" in r.message and "result=blocked" in r.message for r in caplog.records)

    def test_gate1_allow_logs(self, mw, caplog):
        """Gate 1 allowed → structured log emitted."""
        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.context_exists",
            return_value=True,
        ):
            req = _request("task")
            handler = MagicMock()

            with caplog.at_level(logging.INFO):
                mw.wrap_tool_call(req, handler)

        assert any("gate_check" in r.message and "gate=gate1_paradigm" in r.message and "result=allowed" in r.message for r in caplog.records)

    def test_gate2_block_logs(self, mw, caplog):
        """Gate 2 blocked → structured log emitted."""
        with (
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.get_critical_warnings",
                return_value=[{"severity": "critical", "message": "error"}],
            ),
            patch(
                "deerflow.agents.middlewares.gate_enforcement_middleware.is_quality_acknowledged",
                return_value=False,
            ),
        ):
            req = _request_data_analyst()
            handler = MagicMock()

            with caplog.at_level(logging.INFO):
                mw.wrap_tool_call(req, handler)

        assert any("gate_check" in r.message and "gate=gate2_quality" in r.message and "result=blocked" in r.message for r in caplog.records)

    def test_gate2_allow_logs(self, mw, caplog):
        """Gate 2 allowed → structured log emitted."""
        with patch(
            "deerflow.agents.middlewares.gate_enforcement_middleware.get_critical_warnings",
            return_value=[],
        ):
            req = _request_data_analyst()
            handler = MagicMock()

            with caplog.at_level(logging.INFO):
                mw.wrap_tool_call(req, handler)

        assert any("gate_check" in r.message and "gate=gate2_quality" in r.message and "result=allowed" in r.message for r in caplog.records)
