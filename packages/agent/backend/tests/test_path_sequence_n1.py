"""Tests for PathSequenceProvider — n=1 path awareness (data-analyst optional on single subject).

Red anchor: before fix, n=1 chart-maker dispatch without data-analyst handoff was denied.
After fix: n=1 chart-maker/report-writer allowed without data-analyst handoff;
n>=2 still requires data-analyst.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from deerflow.guardrails.path_sequence_provider import (
    PathSequenceProvider,
    _is_single_subject_run,
    _lead_messages,
    _lead_workspace,
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


def _make_request(tool_name: str, subagent_type: str | None = None) -> GuardrailRequest:
    tool_input = {}
    if subagent_type:
        tool_input["subagent_type"] = subagent_type
    return GuardrailRequest(tool_name=tool_name, tool_input=tool_input)


def _set_intent(intent: str) -> None:
    """Set _lead_messages with a real AIMessage containing [intent]."""
    msg = AIMessage(content=f"[intent] {intent}")
    _lead_messages.set([msg])


def _set_workspace(tmp_path: Path) -> None:
    _lead_workspace.set(str(tmp_path))


def _create_handoff(workspace: Path, subagent_name: str) -> None:
    """Create a handoff file for a subagent in the workspace."""
    handoff_file = workspace / f"handoff_{subagent_name}.json"
    handoff_file.write_text(json.dumps({"status": "completed"}))


def _write_ce_handoff(ws: Path, n: int, subjects: int) -> None:
    """Write a handoff_code_executor.json with given n and per_subject count."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "handoff_code_executor.json").write_text(json.dumps({
        "status": "completed",
        "summary": "test handoff",
        "paradigm": "epm",
        "per_subject": {f"Subject {i + 1}": {} for i in range(subjects)},
        "metrics_summary": {"All": {"open_arm_time": {"mean": 1.0, "n": n}}},
    }), encoding="utf-8")


class TestSingleSubjectDetection:
    """Unit tests for _is_single_subject_run helper."""

    def test_n1_detected(self, tmp_path):
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        assert _is_single_subject_run(str(tmp_path)) is True

    def test_n2_not_single(self, tmp_path):
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_missing_handoff_fails_open(self, tmp_path):
        # No handoff → conservative (data-analyst still required)
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_empty_handoff_fails_open(self, tmp_path):
        (tmp_path / "handoff_code_executor.json").write_text("")
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_n1_metrics_only_no_per_subject(self, tmp_path):
        """Only metrics_summary with n=1, no per_subject → detected as single."""
        (tmp_path / "handoff_code_executor.json").write_text(json.dumps({
            "status": "completed",
            "metrics_summary": {"All": {"distance": {"mean": 100.0, "n": 1}}},
        }), encoding="utf-8")
        assert _is_single_subject_run(str(tmp_path)) is True

    def test_no_n_signal_fails_open(self, tmp_path):
        """No per_subject, no n in metrics_summary → fail open."""
        (tmp_path / "handoff_code_executor.json").write_text(json.dumps({
            "status": "completed",
            "metrics_summary": {"All": {"distance": {"mean": 100.0}}},
        }), encoding="utf-8")
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_n5_with_per_subject_2_not_single(self, tmp_path):
        """per_subject has 2 entries → not single regardless of n."""
        _write_ce_handoff(tmp_path, n=1, subjects=2)
        assert _is_single_subject_run(str(tmp_path)) is False


class TestN1ChartMakerAllowedWithoutDataAnalyst:
    """Red anchor: before fix, n=1 chart-maker without data-analyst handoff was denied."""

    def test_n1_chartmaker_allowed(self, provider, reset_contextvars, tmp_path):
        """n=1 chart-maker dispatch: data-analyst handoff missing → still allowed."""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        # Only code-executor handoff exists (n=1), no data-analyst handoff
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is True, (
            f"Expected allow for n=1 chart-maker, got deny: "
            f"{decision.reasons[0].message if decision.reasons else 'no reason'}"
        )

    def test_n1_report_intent_allows_report_writer(self, provider, reset_contextvars, tmp_path):
        """n=1 REPORT intent: report-writer has no predecessors → always allowed
        (needs handoff_code_executor.json for plan precondition)."""
        _set_intent("REPORT")
        _set_workspace(tmp_path)
        # plan precondition: report-writer needs handoff_code_executor.json
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        req = _make_request("task", "report-writer")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_n2_chartmaker_still_requires_data_analyst(self, provider, reset_contextvars, tmp_path):
        """n>=2: data-analyst is still required — no regression."""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False, "n>=2 chart-maker without data-analyst should be denied"
        msg = decision.reasons[0].message
        assert "data-analyst" in msg

    def test_n2_data_analyst_required_partial_handoffs(self, provider, reset_contextvars, tmp_path):
        """n>=2 with code-executor handoff but no data-analyst → data-analyst is required
        and chart-maker should be denied even with code-executor handoff.
        """
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        _create_handoff(tmp_path, "code_executor")
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert "data-analyst" in decision.reasons[0].message

    def test_n1_no_code_executor_handoff_still_denied(self, provider, reset_contextvars, tmp_path):
        """n=1 without code-executor handoff: chart-maker still denied (plan precondition)."""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        # No handoff_code_executor.json → plan precondition should deny
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert "plan_precondition_failed" in decision.reasons[0].code

    def test_n1_code_executor_still_allowed(self, provider, reset_contextvars, tmp_path):
        """code-executor is still allowed as first dispatch regardless of n."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        # plan precondition: code-executor needs plan_metrics.json
        (tmp_path / "plan_metrics.json").write_text('{"metrics": [1]}')
        req = _make_request("task", "code-executor")
        assert provider.evaluate(req).allow is True
