"""Tests for IntentPostStepAskGateProvider — gate3 viz ask enforcement."""

import json
from pathlib import Path

from deerflow.guardrails.intent_post_step_ask_gate_provider import (
    IntentPostStepAskGateProvider,
    _extract_latest_intent,
    _lead_messages,
    _lead_workspace,
)
from deerflow.guardrails.provider import GuardrailRequest


def _make_request(tool_name="task", subagent_type="chart-maker"):
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input={"subagent_type": subagent_type, "description": "test", "prompt": "test"},
    )


# ── intent extraction ──────────────────────────────────────────────────────


def test_extract_latest_intent_from_last_ai_message():
    from langchain_core.messages import AIMessage

    msgs = [
        AIMessage(content="[intent] E2E_FULL_ASKVIZ\n用户上传了数据，我来分析"),
    ]
    assert _extract_latest_intent(msgs) == "E2E_FULL_ASKVIZ"


def test_extract_latest_intent_returns_most_recent():
    from langchain_core.messages import AIMessage

    msgs = [
        AIMessage(content="[intent] CLARIFY"),
        AIMessage(content="[intent] E2E_FULL_ASKVIZ\n现在开始分析"),
    ]
    assert _extract_latest_intent(msgs) == "E2E_FULL_ASKVIZ"


def test_extract_latest_intent_none_when_no_intent():
    from langchain_core.messages import AIMessage

    msgs = [AIMessage(content="我来分析数据")]
    assert _extract_latest_intent(msgs) is None


def test_extract_latest_intent_empty_messages():
    assert _extract_latest_intent([]) is None
    assert _extract_latest_intent(None) is None


# ── allow cases ─────────────────────────────────────────────────────────────


def test_allows_non_task_tool():
    provider = IntentPostStepAskGateProvider()
    req = GuardrailRequest(tool_name="read_file", tool_input={})
    decision = provider.evaluate(req)
    assert decision.allow is True

def test_allows_task_with_non_chart_maker():
    provider = IntentPostStepAskGateProvider()
    req = _make_request(subagent_type="code-executor")
    decision = provider.evaluate(req)
    assert decision.allow is True


def test_allows_when_no_workspace():
    """Fail-open: no workspace → don't block."""
    provider = IntentPostStepAskGateProvider()
    _lead_workspace.set(None)
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


def test_allows_when_no_context_file(tmp_path):
    """Fail-open: no experiment-context.json → don't block."""
    provider = IntentPostStepAskGateProvider()
    _lead_workspace.set(str(tmp_path))
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


def test_allows_when_gate3_already_acknowledged(tmp_path):
    """Gate3 already done → allow."""
    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm", "gate3_viz_acknowledged"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))

    provider = IntentPostStepAskGateProvider()
    _lead_workspace.set(str(tmp_path))
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


def test_allows_when_no_data_analyst_handoff(tmp_path):
    """data-analyst not done → don't block yet."""
    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))

    provider = IntentPostStepAskGateProvider()
    _lead_workspace.set(str(tmp_path))
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


def test_allows_when_intent_is_not_askviz(tmp_path):
    """Only E2E_FULL_ASKVIZ is intercepted."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    (tmp_path / "handoff_data_analyst.json").write_text('{"status":"completed"}')

    _lead_messages.set([AIMessage(content="[intent] E2E_FULL")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


# ── deny case ───────────────────────────────────────────────────────────────


def test_blocks_chart_maker_when_askviz_skip(tmp_path):
    """The core scenario: ASKVIZ intent, data-analyst done, gate3 NOT acknowledged."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    (tmp_path / "handoff_data_analyst.json").write_text('{"status":"completed"}')

    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n分析完成")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    decision = provider.evaluate(_make_request())

    assert decision.allow is False
    assert len(decision.reasons) == 1
    reason = decision.reasons[0]
    assert reason.code == "ethoinsight.viz_choice_not_acknowledged"
    # Deny message must contain "请改用" + "因为" + "然后" pattern (spec §1)
    assert "请改调" in reason.message
    assert "因为" in reason.message
    assert "之后" in reason.message
    assert "ask_clarification" in reason.message
    assert "set_viz_choice" in reason.message
    assert "chart-maker" in reason.message
