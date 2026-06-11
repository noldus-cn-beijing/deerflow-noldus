"""Tests for PathSequenceProvider — n=1 path awareness.

chart-maker 的 prerequisite 只有 code-executor（不依赖 data-analyst），
所以 n=1 和 n≥2 对于 chart-maker 的 guardrail 行为是一致的。
_is_single_subject_run 保留供 lead prompt fast-path 判定和其他 guardrail 使用。
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
    """Unit tests for _is_single_subject_run helper — retained for lead prompt / other guardrails."""

    def test_n1_detected(self, tmp_path):
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        assert _is_single_subject_run(str(tmp_path)) is True

    def test_n2_not_single(self, tmp_path):
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_missing_handoff_fails_open(self, tmp_path):
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


class TestChartMakerPrerequisites:
    """chart-maker 的 prerequisite 只有 code-executor，不依赖 data-analyst。
    无论 n=1 还是 n≥2，行为一致。
    """

    def test_chart_maker_allowed_with_only_code_executor(self, provider, reset_contextvars, tmp_path):
        """chart-maker with code-executor handoff → allowed (data-analyst NOT needed)."""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        _create_handoff(tmp_path, "code_executor")
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is True, (
            f"chart-maker should be allowed with only code-executor handoff; "
            f"got deny: {decision.reasons[0].message if decision.reasons else 'no reason'}"
        )

    def test_chart_maker_n1_allowed(self, provider, reset_contextvars, tmp_path):
        """n=1: chart-maker with code-executor handoff → allowed.
        (Same as n≥2 — chart-maker never depends on data-analyst.)"""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is True

    def test_chart_maker_n2_allowed_without_data_analyst(self, provider, reset_contextvars, tmp_path):
        """n≥2: chart-maker with code-executor → still allowed without data-analyst.
        (chart-maker only needs code-executor data, not data-analyst interpretation.)"""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is True, (
            f"chart-maker should be allowed with code-executor handoff even when n≥2; "
            f"data-analyst is NOT a prerequisite. "
            f"Got deny: {decision.reasons[0].message if decision.reasons else 'no reason'}"
        )

    def test_chart_maker_without_code_executor_denied(self, provider, reset_contextvars, tmp_path):
        """chart-maker without code-executor handoff → denied by plan precondition."""
        _set_intent("E2E_FULL_ASKVIZ")
        _set_workspace(tmp_path)
        req = _make_request("task", "chart-maker")
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert "plan_precondition_failed" in decision.reasons[0].code

    def test_data_analyst_still_requires_code_executor(self, provider, reset_contextvars, tmp_path):
        """data-analyst's prerequisite is code-executor — must still be enforced."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        req = _make_request("task", "data-analyst")
        decision = provider.evaluate(req)
        assert decision.allow is False
        assert "code-executor" in decision.reasons[0].message

    def test_data_analyst_with_code_executor_allowed(self, provider, reset_contextvars, tmp_path):
        """data-analyst with code-executor handoff → allowed."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        _create_handoff(tmp_path, "code_executor")
        req = _make_request("task", "data-analyst")
        assert provider.evaluate(req).allow is True

    def test_n1_code_executor_still_allowed(self, provider, reset_contextvars, tmp_path):
        """code-executor is always allowed as first dispatch."""
        _set_intent("E2E_FULL")
        _set_workspace(tmp_path)
        (tmp_path / "plan_metrics.json").write_text('{"metrics": [1]}')
        req = _make_request("task", "code-executor")
        assert provider.evaluate(req).allow is True
