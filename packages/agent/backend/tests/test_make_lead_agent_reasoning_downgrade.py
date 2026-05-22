"""Tests for reasoning_effort phase-based downgrade in make_lead_agent."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _step_down(effort):
    """Replicate the step_down helper from agent.py for testing."""
    if effort == "high":
        return "medium"
    if effort == "medium":
        return "low"
    return effort


class TestStepDown:
    def test_high_to_medium(self):
        assert _step_down("high") == "medium"

    def test_medium_to_low(self):
        assert _step_down("medium") == "low"

    def test_low_stays_low(self):
        assert _step_down("low") == "low"

    def test_none_stays_none(self):
        assert _step_down(None) is None


class TestReasoningDowngradeIntegration:
    """Integration-style tests for the downgrade logic in make_lead_agent.

    These mock the config and filesystem to verify the downgrade decision,
    exercising the same read_context + gate_completed logic.
    """

    def _make_config(self, thread_id, reasoning_effort="high"):
        return {
            "configurable": {
                "thread_id": thread_id,
                "reasoning_effort": reasoning_effort,
                "thinking_enabled": True,
                "model_name": "deepseek-v4-pro",
                "is_plan_mode": True,
                "subagent_enabled": True,
                "max_concurrent_subagents": 3,
                "workflow_mode": "auto",
            }
        }

    def test_no_workspace_keeps_configured(self, tmp_path):
        """When experiment-context.json doesn't exist, reasoning_effort is unchanged."""
        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        assert ctx is None
        # No context → no downgrade
        effort = "high"
        if ctx:
            gate_completed = ctx.get("gate_completed", [])
            if "gate2_quality_acknowledged" in gate_completed:
                effort = _step_down(_step_down(effort))
            elif "gate1_paradigm" in gate_completed:
                effort = _step_down(effort)
        assert effort == "high"

    def test_gate1_done_downgrades_once(self, tmp_path):
        """gate_completed=["gate1_paradigm"] → high→medium."""
        ctx_path = tmp_path / "experiment-context.json"
        ctx_path.write_text(json.dumps({"paradigm": "fst", "gate_completed": ["gate1_paradigm"]}))

        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        effort = "high"
        if ctx:
            gate_completed = ctx.get("gate_completed", [])
            if isinstance(gate_completed, list):
                if "gate2_quality_acknowledged" in gate_completed:
                    effort = _step_down(_step_down(effort))
                elif "gate1_paradigm" in gate_completed:
                    effort = _step_down(effort)
        assert effort == "medium"

    def test_gate2_done_downgrades_twice(self, tmp_path):
        """gate_completed with both gates → high→low."""
        ctx_path = tmp_path / "experiment-context.json"
        ctx_path.write_text(
            json.dumps({"paradigm": "fst", "gate_completed": ["gate1_paradigm", "gate2_quality_acknowledged"]})
        )

        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        effort = "high"
        if ctx:
            gate_completed = ctx.get("gate_completed", [])
            if isinstance(gate_completed, list):
                if "gate2_quality_acknowledged" in gate_completed:
                    effort = _step_down(_step_down(effort))
                elif "gate1_paradigm" in gate_completed:
                    effort = _step_down(effort)
        assert effort == "low"

    def test_configured_none_stays_none(self, tmp_path):
        """When reasoning_effort is None, no downgrade even with gate2 done."""
        ctx_path = tmp_path / "experiment-context.json"
        ctx_path.write_text(
            json.dumps({"paradigm": "fst", "gate_completed": ["gate1_paradigm", "gate2_quality_acknowledged"]})
        )

        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        effort = None
        if effort and ctx:  # short-circuit on None
            gate_completed = ctx.get("gate_completed", [])
            if isinstance(gate_completed, list):
                if "gate2_quality_acknowledged" in gate_completed:
                    effort = _step_down(_step_down(effort))
                elif "gate1_paradigm" in gate_completed:
                    effort = _step_down(effort)
        assert effort is None

    def test_configured_low_stays_low(self, tmp_path):
        """When reasoning_effort is low, no further downgrade."""
        ctx_path = tmp_path / "experiment-context.json"
        ctx_path.write_text(
            json.dumps({"paradigm": "fst", "gate_completed": ["gate1_paradigm", "gate2_quality_acknowledged"]})
        )

        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        effort = "low"
        if ctx:
            gate_completed = ctx.get("gate_completed", [])
            if isinstance(gate_completed, list):
                if "gate2_quality_acknowledged" in gate_completed:
                    effort = _step_down(_step_down(effort))
                elif "gate1_paradigm" in gate_completed:
                    effort = _step_down(effort)
        assert effort == "low"

    def test_corrupted_context_fail_safe(self, tmp_path):
        """Corrupted JSON → fail-safe, keeps configured effort."""
        ctx_path = tmp_path / "experiment-context.json"
        ctx_path.write_text("not valid json {{{")

        from deerflow.agents.middlewares.experiment_context import read_context

        ctx = read_context(str(tmp_path))
        # read_context returns None on JSON decode error
        assert ctx is None
