"""Tests for planning instructions in lead_agent prompt."""

import pytest

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.subagents import get_available_subagent_names


def _has_noldus_agents() -> bool:
    names = set(get_available_subagent_names())
    return bool({"code-executor", "data-analyst", "report-writer", "knowledge-assistant"} & names)


def test_prompt_contains_planning_directive_when_noldus_agents_present():
    """Prompt should mention ethoinsight-planning skill when Noldus subagents are registered."""
    if not _has_noldus_agents():
        pytest.skip("Noldus subagents not registered in this environment")

    prompt = apply_prompt_template(subagent_enabled=True)

    assert "ethoinsight-planning" in prompt, "Prompt must reference the planning skill"
    assert "规划先于派遣" in prompt, "Prompt must enforce planning before delegation"


def test_prompt_lists_mandatory_clarification_cases():
    """Prompt should state only paradigm and group inference failures require clarification."""
    if not _has_noldus_agents():
        pytest.skip("Noldus subagents not registered in this environment")

    prompt = apply_prompt_template(subagent_enabled=True)

    assert "范式推断失败" in prompt
    assert "分组无法推断" in prompt


def test_prompt_without_subagents_has_no_planning_directive():
    """When subagents are disabled, planning directive should not appear."""
    prompt = apply_prompt_template(subagent_enabled=False)
    assert "规划先于派遣" not in prompt
