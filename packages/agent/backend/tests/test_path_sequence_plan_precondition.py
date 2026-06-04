"""Tests for PathSequenceProvider plan precondition gate (PR-2).

Covers:
- code-executor denied when plan_metrics.json is missing/empty
- chart-maker/report-writer denied when handoff_code_executor.json is missing/empty
- Allowed when preconditions are met
- QA/CLARIFY paths not affected (no plan needed)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deerflow.guardrails.path_sequence_provider import (
    PathSequenceProvider,
    _lead_messages,
    _lead_workspace,
    _check_plan_precondition,
)
from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture()
def provider():
    return PathSequenceProvider()


@pytest.fixture()
def reset_contextvars():
    """Reset contextvars before each test."""
    _lead_messages.set(None)
    _lead_workspace.set(None)
    yield
    _lead_messages.set(None)
    _lead_workspace.set(None)


def _make_request(subagent_type: str) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name="task",
        tool_input={"subagent_type": subagent_type},
    )


def _set_workspace(tmp_path: Path) -> None:
    _lead_workspace.set(str(tmp_path))


# ============================================================
# _check_plan_precondition unit tests
# ============================================================

class TestPlanPreconditionHelper:
    """Direct unit tests for _check_plan_precondition."""

    def test_code_executor_missing_plan_denied(self, tmp_path):
        """code-executor without plan_metrics.json → deny."""
        _set_workspace(tmp_path)
        result = _check_plan_precondition("code-executor", str(tmp_path))
        assert result is not None
        assert not result.allow
        assert "plan_metrics.json" in result.reasons[0].message

    def test_code_executor_empty_plan_denied(self, tmp_path):
        """code-executor with empty plan_metrics.json → deny."""
        _set_workspace(tmp_path)
        (tmp_path / "plan_metrics.json").write_text("")
        result = _check_plan_precondition("code-executor", str(tmp_path))
        assert result is not None
        assert not result.allow

    def test_code_executor_with_plan_allowed(self, tmp_path):
        """code-executor with non-empty plan_metrics.json → allow."""
        _set_workspace(tmp_path)
        (tmp_path / "plan_metrics.json").write_text('{"metrics": [1]}')
        result = _check_plan_precondition("code-executor", str(tmp_path))
        assert result is None  # None means precondition satisfied

    def test_chart_maker_missing_handoff_denied(self, tmp_path):
        """chart-maker without handoff_code_executor.json → deny."""
        _set_workspace(tmp_path)
        result = _check_plan_precondition("chart-maker", str(tmp_path))
        assert result is not None
        assert not result.allow
        assert "handoff_code_executor.json" in result.reasons[0].message

    def test_chart_maker_with_handoff_allowed(self, tmp_path):
        """chart-maker with non-empty handoff_code_executor.json → allow."""
        _set_workspace(tmp_path)
        (tmp_path / "handoff_code_executor.json").write_text('{"status": "ok"}')
        result = _check_plan_precondition("chart-maker", str(tmp_path))
        assert result is None

    def test_report_writer_missing_handoff_denied(self, tmp_path):
        """report-writer without handoff_code_executor.json → deny."""
        _set_workspace(tmp_path)
        result = _check_plan_precondition("report-writer", str(tmp_path))
        assert result is not None
        assert not result.allow

    def test_report_writer_with_handoff_allowed(self, tmp_path):
        """report-writer with non-empty handoff_code_executor.json → allow."""
        _set_workspace(tmp_path)
        (tmp_path / "handoff_code_executor.json").write_text('{"status": "ok"}')
        result = _check_plan_precondition("report-writer", str(tmp_path))
        assert result is None

    def test_other_subagent_not_affected(self, tmp_path):
        """data-analyst does not need plan precondition check."""
        _set_workspace(tmp_path)
        result = _check_plan_precondition("data-analyst", str(tmp_path))
        assert result is None

    def test_qa_subagent_not_affected(self, tmp_path):
        """knowledge-assistant does not need plan precondition check."""
        _set_workspace(tmp_path)
        result = _check_plan_precondition("knowledge-assistant", str(tmp_path))
        assert result is None


# ============================================================
# Full provider integration tests
# ============================================================

class TestProviderPlanPrecondition:
    """Test the full PathSequenceProvider.evaluate with plan precondition."""

    def test_code_executor_no_workspace_allow_fail_open(self, provider, reset_contextvars):
        """Without workspace context, fail-open → allow."""
        _lead_workspace.set(None)
        _lead_messages.set(None)
        result = provider.evaluate(_make_request("code-executor"))
        assert result.allow

    def test_code_executor_no_plan_denied(self, provider, reset_contextvars, tmp_path):
        """With workspace but no plan → deny even without intent."""
        _set_workspace(tmp_path)
        _lead_messages.set(None)  # No intent → would fail-open for sequence check
        result = provider.evaluate(_make_request("code-executor"))
        assert not result.allow
        assert "plan_metrics.json" in result.reasons[0].message

    def test_chart_maker_no_handoff_denied(self, provider, reset_contextvars, tmp_path):
        """chart-maker without handoff → denied."""
        _set_workspace(tmp_path)
        _lead_messages.set(None)
        result = provider.evaluate(_make_request("chart-maker"))
        assert not result.allow
        assert "handoff_code_executor.json" in result.reasons[0].message

    def test_non_task_tool_not_affected(self, provider, reset_contextvars, tmp_path):
        """read_file is not a task tool → allowed."""
        _set_workspace(tmp_path)
        _lead_messages.set(None)
        req = GuardrailRequest(tool_name="read_file", tool_input={})
        result = provider.evaluate(req)
        assert result.allow

    def test_code_executor_with_plan_and_intent_allowed(self, provider, reset_contextvars, tmp_path):
        """With plan present and valid path → allowed."""
        from langchain_core.messages import AIMessage

        _set_workspace(tmp_path)
        (tmp_path / "plan_metrics.json").write_text('{"metrics": [{"id": "test"}]}')
        # Set intent that has code-executor as first dispatch
        msg = AIMessage(content="[intent] E2E_FULL")
        _lead_messages.set([msg])
        result = provider.evaluate(_make_request("code-executor"))
        # code-executor is first dispatch in E2E_FULL → no preceding steps needed
        assert result.allow
