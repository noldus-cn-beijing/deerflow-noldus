"""W18: TaskHandoffAuthorizationProvider 验收。"""
from __future__ import annotations

from deerflow.guardrails.task_handoff_authorization_provider import (
    TaskHandoffAuthorizationProvider,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_task_request(subagent_type: str, prompt: str) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name="task",
        tool_input={"subagent_type": subagent_type, "prompt": prompt, "description": "x"},
    )


def test_allow_when_no_required_handoffs():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("code-executor", "请分析数据"))
    assert decision.allow


def test_allow_when_required_handoff_present():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(
        _make_task_request("data-analyst", "请解读 {{handoff://code_executor}}")
    )
    assert decision.allow


def test_deny_when_required_handoff_missing():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("data-analyst", "请解读结果"))
    assert not decision.allow
    assert decision.reasons[0].code == "ethoinsight.required_handoff_missing"
    assert "code_executor" in decision.reasons[0].message


def test_deny_when_one_of_multiple_required_missing():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(
        _make_task_request("report-writer", "写报告 {{handoff://code_executor}}")
    )
    assert not decision.allow
    assert "data_analyst" in decision.reasons[0].message


def test_allow_when_all_required_present():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request(
        "report-writer",
        "写报告 {{handoff://code_executor}} {{handoff://data_analyst}}",
    ))
    assert decision.allow


def test_unknown_subagent_type_passes_through():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(_make_task_request("nonexistent", "x"))
    assert decision.allow


def test_non_task_tools_pass_through():
    provider = TaskHandoffAuthorizationProvider()
    decision = provider.evaluate(GuardrailRequest(tool_name="read_file", tool_input={}))
    assert decision.allow
