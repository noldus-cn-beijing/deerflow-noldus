"""Lead prompt contract: language lock + no structured dump output.

These are prompt-level contract tests; they assert that key instruction
phrases are present in the generated system prompt, not that the model
behaves a certain way (that requires E2E).
"""
from __future__ import annotations

from deerflow.agents.lead_agent.prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    _build_subagent_section,
)


def test_language_lock_rule_present_in_system_prompt():
    assert "用户语言" in SYSTEM_PROMPT_TEMPLATE
    # Positive phrasing only (GLM-5.1 rule): no "禁止/不要 X" directives
    assert "用和用户相同的语言回答" in SYSTEM_PROMPT_TEMPLATE


def test_no_dump_style_rule_present():
    # Positive phrasing: tell lead HOW to reply, not what to avoid.
    assert "自然段落" in SYSTEM_PROMPT_TEMPLATE
    assert "项目符号" in SYSTEM_PROMPT_TEMPLATE


def test_subagent_section_still_intact():
    # Guard: language rule changes should not break existing subagent block.
    section = _build_subagent_section(max_concurrent=3)
    assert "data-analyst" in section
    assert "code-executor" in section
    assert "report-writer" in section
