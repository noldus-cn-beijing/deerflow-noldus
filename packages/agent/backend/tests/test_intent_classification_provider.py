"""W17: IntentClassificationGuardrailProvider 验收。"""
from __future__ import annotations

import pytest
from contextvars import copy_context
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.guardrails.intent_classification_provider import (
    IntentClassificationGuardrailProvider,
    _lead_messages,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_request(tool_name: str, args: dict | None = None) -> GuardrailRequest:
    return GuardrailRequest(tool_name=tool_name, tool_input=args or {})


@pytest.fixture
def provider():
    return IntentClassificationGuardrailProvider()


def test_allow_read_file_always(provider):
    _lead_messages.set([HumanMessage(content="hi")])
    decision = provider.evaluate(_make_request("read_file", {"file_path": "/mnt/skills/x/SKILL.md"}))
    assert decision.allow


def test_allow_when_intent_declared(provider):
    _lead_messages.set([
        HumanMessage(content="分析这个数据"),
        AIMessage(content="[intent] E2E_MIN\n我开始分析..."),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow


def test_deny_when_intent_missing_and_non_read_tool(provider):
    _lead_messages.set([
        HumanMessage(content="分析数据"),
        AIMessage(content="我马上派 subagent。"),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert not decision.allow
    assert decision.reasons[0].code == "ethoinsight.intent_not_declared"


def test_allow_when_messages_empty(provider):
    _lead_messages.set(None)
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow


def test_intent_recognized_even_if_not_last_message(provider):
    _lead_messages.set([
        HumanMessage(content="x"),
        AIMessage(content="[intent] E2E_MIN\nstart"),
        AIMessage(content="now dispatching..."),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert decision.allow


def test_intent_invalid_name_does_not_count(provider):
    _lead_messages.set([
        HumanMessage(content="x"),
        AIMessage(content="[intent] FOO\nrouting"),
    ])
    decision = provider.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
    assert not decision.allow
