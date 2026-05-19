"""W19: task_tool 自动注入 required_upstream_handoffs 占位符。"""
from __future__ import annotations

from deerflow.tools.builtins.task_tool import (
    _auto_inject_handoff_placeholders,
    _HANDOFF_PLACEHOLDER_RE,
)


def test_auto_inject_for_data_analyst_when_missing():
    new_prompt = _auto_inject_handoff_placeholders("请解读结果", "data-analyst")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found


def test_no_inject_when_no_required_handoffs():
    new_prompt = _auto_inject_handoff_placeholders("请分析", "code-executor")
    assert _HANDOFF_PLACEHOLDER_RE.findall(new_prompt) == []
    assert new_prompt.strip() == "请分析"


def test_no_double_inject():
    original = "请解读 {{handoff://code_executor}} 的结果"
    new_prompt = _auto_inject_handoff_placeholders(original, "data-analyst")
    matches = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert matches.count("code_executor") == 1


def test_multi_required_all_injected():
    new_prompt = _auto_inject_handoff_placeholders("写报告", "report-writer")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found
    assert "data_analyst" in found


def test_partial_handwritten_injects_only_missing():
    original = "写报告 {{handoff://code_executor}}"
    new_prompt = _auto_inject_handoff_placeholders(original, "report-writer")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert found.count("code_executor") == 1
    assert "data_analyst" in found


def test_unknown_subagent_type_passthrough():
    result = _auto_inject_handoff_placeholders("x", "nonexistent")
    assert result == "x"


def test_chart_maker_gets_code_executor():
    new_prompt = _auto_inject_handoff_placeholders("画图", "chart-maker")
    found = _HANDOFF_PLACEHOLDER_RE.findall(new_prompt)
    assert "code_executor" in found
