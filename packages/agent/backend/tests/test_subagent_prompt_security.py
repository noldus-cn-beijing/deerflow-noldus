"""Tests for Noldus subagent availability and prompt exposure."""

from deerflow.agents.lead_agent import prompt as prompt_module
from deerflow.subagents import registry as registry_module


def test_get_available_subagent_names_returns_noldus_agents() -> None:
    """Noldus registry exposes custom subagents instead of upstream defaults."""
    names = registry_module.get_available_subagent_names()

    assert "code-executor" in names
    assert "data-analyst" in names
    assert "report-writer" in names
    assert "knowledge-assistant" in names


def test_get_available_subagent_names_does_not_include_upstream_defaults() -> None:
    """Upstream general-purpose and bash are replaced by Noldus agents."""
    names = registry_module.get_available_subagent_names()

    assert "general-purpose" not in names
    assert "bash" not in names


def test_build_subagent_section_contains_noldus_agents(monkeypatch) -> None:
    """Subagent section should mention Noldus-specific agents."""
    monkeypatch.setattr(
        prompt_module,
        "get_available_subagent_names",
        lambda: ["code-executor", "data-analyst", "report-writer", "knowledge-assistant"],
    )

    section = prompt_module._build_subagent_section(3)

    assert "code-executor" in section
    assert "data-analyst" in section
    assert "report-writer" in section
    assert "knowledge-assistant" in section


def test_build_subagent_section_contains_dispatch_rules(monkeypatch) -> None:
    """Subagent section should include Noldus dispatch/orchestration rules."""
    monkeypatch.setattr(
        prompt_module,
        "get_available_subagent_names",
        lambda: ["code-executor", "data-analyst", "report-writer", "knowledge-assistant"],
    )

    section = prompt_module._build_subagent_section(3)

    # Should contain the orchestration examples, not the upstream AWS/Azure examples
    assert "code-executor" in section
    assert "subagent_system" in section
