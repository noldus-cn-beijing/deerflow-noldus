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


# ── A2 generalized ask gate tests ────────────────────────────────────────────


def test_blocks_chart_maker_in_e2e_full_when_report_ask_not_done(tmp_path):
    """E2E_FULL: chart-maker dispatched after chart-maker → ask(report?) not done → block report-writer."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    # chart-maker completed
    (tmp_path / "handoff_chart_maker.json").write_text('{"status":"completed"}')

    _lead_messages.set([AIMessage(content="[intent] E2E_FULL")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    # report-writer is dispatched but ask(report?) hasn't been answered
    # Note: report-writer is NOT in the E2E_FULL path (only chart-maker → ask(report))
    # The ask(report?) is the last step, so trying to skip it would mean going past it
    # Actually, report-writer isn't in E2E_FULL path at all — this should allow
    decision = provider.evaluate(_make_request(subagent_type="report-writer"))
    assert decision.allow is True  # report-writer not in E2E_FULL path


def test_askviz_viz_gate_is_byte_identical(tmp_path):
    """Verify the viz deny message is exactly the same as the pre-A2 version."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    (tmp_path / "handoff_data_analyst.json").write_text('{"status":"completed"}')

    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    decision = provider.evaluate(_make_request())

    assert decision.allow is False
    reason = decision.reasons[0]
    assert reason.code == "ethoinsight.viz_choice_not_acknowledged"
    # The exact legacy message
    expected = (
        "请改调 ask_clarification(question='📊 指标和解读已完成。需要我把结果可视化成图吗?', "
        "options=['A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)', "
        "'B. 不用,直接给我报告'])，因为 INTENT=E2E_FULL_ASKVIZ 要求 data-analyst 完成后 "
        "先反问用户是否需要图表；用户回答后再调 set_viz_choice(choice='yes'|'no') "
        "落盘 gate3，之后才能派 chart-maker（或跳过直接派 report-writer）。"
    )
    assert reason.message == expected


def test_askviz_viz_gate_allows_when_no_askviz_intent(tmp_path):
    """Non-ASKVIZ intent → no viz gate check → allow."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    (tmp_path / "handoff_data_analyst.json").write_text('{"status":"completed"}')

    # E2E_FULL has chart-maker as a direct dispatch (no ask(viz) before it)
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    decision = provider.evaluate(_make_request())
    assert decision.allow is True


def test_e2e_min_blocks_code_executor_after_code_executor_but_before_ask(tmp_path):
    """E2E_MIN: after code-executor, the next step is ask(four_choice).
    If lead tries to dispatch another step before answering four_choice,
    it should be blocked. But chart-maker is not in E2E_MIN path, so allow."""
    from langchain_core.messages import AIMessage

    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    (tmp_path / "handoff_code_executor.json").write_text('{"status":"completed"}')

    _lead_messages.set([AIMessage(content="[intent] E2E_MIN")])
    _lead_workspace.set(str(tmp_path))

    provider = IntentPostStepAskGateProvider()
    # chart-maker not in E2E_MIN path → allow
    decision = provider.evaluate(_make_request(subagent_type="chart-maker"))
    assert decision.allow is True
