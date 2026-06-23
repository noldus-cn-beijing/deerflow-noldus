"""Tests for AskClarificationOptionsProvider — 强制结构化反问点带快捷选项。

ETHO-9 确定性数据修复：path_registry 声明哪些 ask step 必须带 options，
guardrail 拦 ask_clarification 强制之。开放澄清（CLARIFY/clarify）不强制。

这些测试复用 IntentPostStepAskGateBridge 设置的 _lead_messages /
_lead_workspace contextvar（同一桥接中间件在所有 guardrail 之前运行）。
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from deerflow.guardrails.ask_clarification_options_provider import (
    AskClarificationOptionsProvider,
)
from deerflow.guardrails.intent_post_step_ask_gate_provider import (
    _lead_messages,
    _lead_workspace,
)
from deerflow.guardrails.path_registry import (
    PATHS,
    Step,
    next_pending_ask_step,
)
from deerflow.guardrails.provider import GuardrailRequest


# ── helpers ──────────────────────────────────────────────────────────────────


def _ask_request(options: list[str] | None = None, **extra) -> GuardrailRequest:
    tool_input = {
        "question": "需要出图吗?",
        "clarification_type": "approach_choice",
    }
    if options is not None:
        tool_input["options"] = options
    tool_input.update(extra)
    return GuardrailRequest(tool_name="ask_clarification", tool_input=tool_input)


def _seed_askviz_workspace(tmp_path, *, gate_completed: list[str]) -> None:
    """Set up a workspace where data-analyst is done (so ask(viz?) is the pending ask)."""
    ctx = {"paradigm": "epm", "gate_completed": gate_completed}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    # data-analyst handoff done → its preceding dispatch completed
    (tmp_path / "handoff_data_analyst.json").write_text('{"status":"completed"}')


# ── next_pending_ask_step helper ──────────────────────────────────────────────


def test_next_pending_ask_step_returns_requires_options_ask():
    """E2E_FULL_ASKVIZ: data-analyst done, gate3 not acknowledged → pending = ask(viz?)."""
    steps = PATHS["E2E_FULL_ASKVIZ"]

    def handoff_exists(target: str) -> bool:
        return target in {"code-executor", "data-analyst"}

    pending = next_pending_ask_step(
        steps,
        gate_completed=["gate1_paradigm"],
        handoff_exists=handoff_exists,
    )
    assert pending is not None
    assert pending.target == "viz"
    assert pending.requires_options is True


def test_next_pending_ask_step_skips_acknowledged_gate():
    """gate3 acknowledged → ask(viz?) no longer pending; ask(report?) pending after chart-maker."""
    steps = PATHS["E2E_FULL_ASKVIZ"]

    def handoff_exists(target: str) -> bool:
        return target in {"code-executor", "data-analyst", "chart-maker"}

    pending = next_pending_ask_step(
        steps,
        gate_completed=["gate1_paradigm", "gate3_viz_acknowledged"],
        handoff_exists=handoff_exists,
    )
    assert pending is not None
    assert pending.target == "report"
    assert pending.requires_options is True


def test_next_pending_ask_step_none_when_all_done():
    steps = PATHS["E2E_FULL_ASKVIZ"]

    def handoff_exists(target: str) -> bool:
        return True

    pending = next_pending_ask_step(
        steps,
        gate_completed=["gate1_paradigm", "gate3_viz_acknowledged", "gate4_report_acknowledged"],
        handoff_exists=handoff_exists,
    )
    assert pending is None


def test_next_pending_ask_step_open_clarify_not_requires_options():
    """CLARIFY path's ask(clarify) has requires_options=False."""
    steps = PATHS["CLARIFY"]

    def handoff_exists(target: str) -> bool:
        return True

    pending = next_pending_ask_step(
        steps,
        gate_completed=[],
        handoff_exists=handoff_exists,
    )
    assert pending is not None
    assert pending.target == "clarify"
    assert pending.requires_options is False


# ── provider: deny when required options missing ─────────────────────────────


def test_deny_when_required_options_missing(tmp_path):
    """核心：E2E_FULL_ASKVIZ 路径、ask(viz?) requires_options=True，不带 options → deny。"""
    _seed_askviz_workspace(tmp_path, gate_completed=["gate1_paradigm"])
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n分析完成")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(_ask_request(options=None))

    assert decision.allow is False
    assert len(decision.reasons) == 1
    reason = decision.reasons[0]
    # code identifies the ask step
    assert "viz" in reason.code
    # deny message directs the lead: ask_clarification + options
    assert "ask_clarification" in reason.message
    assert "options" in reason.message


def test_allow_when_options_present(tmp_path):
    """同上但带 options(≥2) → allow。"""
    _seed_askviz_workspace(tmp_path, gate_completed=["gate1_paradigm"])
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n分析完成")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(
        _ask_request(options=["A. 是，画图", "B. 不用，直接报告"])
    )
    assert decision.allow is True


def test_allow_open_clarify_without_options(tmp_path):
    """守边界：CLARIFY 路径 clarify step（requires_options=False）不带 options → allow。"""
    ctx = {"paradigm": "epm", "gate_completed": []}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx))
    _lead_messages.set([AIMessage(content="[intent] CLARIFY")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(_ask_request(options=None))
    assert decision.allow is True


def test_empty_options_denied(tmp_path):
    """reward hacking 防护：options=[] 空列表 → deny（不能用空糊弄）。"""
    _seed_askviz_workspace(tmp_path, gate_completed=["gate1_paradigm"])
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n分析完成")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(_ask_request(options=[]))
    assert decision.allow is False


# ── backward compat + fail-open ───────────────────────────────────────────────


def test_step_requires_options_backward_compat():
    """现有 dispatch step / 未标注 ask step → requires_options 默认 False。"""
    dispatch = Step("dispatch", "code-executor")
    assert dispatch.requires_options is False
    unmarked_ask = Step("ask", "clarify")
    assert unmarked_ask.requires_options is False
    # marked ask carries True
    marked = Step("ask", "viz", requires_options=True)
    assert marked.requires_options is True


def test_allow_non_ask_clarification_tool(tmp_path):
    """Only intercept ask_clarification — other tools pass through."""
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    req = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "code-executor"})
    assert provider.evaluate(req).allow is True


def test_fail_open_when_no_intent(tmp_path):
    """No declared intent → don't block (fail-open)."""
    _lead_messages.set([AIMessage(content="我来分析")])
    _lead_workspace.set(str(tmp_path))

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(_ask_request(options=None))
    assert decision.allow is True


def test_fail_open_when_no_workspace():
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ")])
    _lead_workspace.set(None)

    provider = AskClarificationOptionsProvider()
    decision = provider.evaluate(_ask_request(options=None))
    assert decision.allow is True


# ── smoke ─────────────────────────────────────────────────────────────────────


def test_provider_instantiates_and_evaluates():
    """provider 实例化 + 喂合成 request 跑通不抛。"""
    provider = AskClarificationOptionsProvider()
    assert provider.name == "ask_clarification_options"
    # synthesise a request that fails open (no intent / no workspace)
    _lead_messages.set(None)
    _lead_workspace.set(None)
    decision = provider.evaluate(_ask_request(options=None))
    assert decision.allow is True
