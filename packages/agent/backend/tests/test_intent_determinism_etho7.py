"""ETHO-7 决策点确定性测试 —— intent 分类稳定化 + PATHS ask 顺序强制。

Spec: docs/superpowers/specs/2026-06-23-etho7-intent-determinism-decision-point-drift-spec.md

子改动 A（intent 分类确定化）：
  E2E_FULL（跳出图反问）需对话里有用户明确出图意向，否则 deny 要求 E2E_FULL_ASKVIZ。
  reward-hacking 防护（§六.1）：检测 HumanMessage 实际文本，不信 lead 自述。

子改动 B（PATHS ask 顺序强制）：
  扩 IntentPostStepAskGate 检测 ask 乱序/合并 —— 后置 ask 已完成但前置 ask 未完成 → deny。
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.guardrails.intent_classification_provider import (
    IntentClassificationGuardrailProvider,
    _lead_messages,
)
from deerflow.guardrails.intent_post_step_ask_gate_provider import (
    IntentPostStepAskGateProvider,
    _lead_workspace,
)
from deerflow.guardrails.intent_post_step_ask_gate_provider import (
    _lead_messages as _gate_lead_messages,
)
from deerflow.guardrails.path_registry import VIZ_INTENT_KEYWORDS
from deerflow.guardrails.provider import GuardrailRequest

# ── 子改动 A：intent 分类确定化 ────────────────────────────────────────────────


@pytest.fixture
def provider():
    return IntentClassificationGuardrailProvider()


def _task_request() -> GuardrailRequest:
    return GuardrailRequest(
        tool_name="task",
        tool_input={"subagent_type": "code-executor", "description": "x", "prompt": "x"},
    )


def test_deny_e2e_full_without_explicit_viz_intent(provider):
    """核心：lead 声明 E2E_FULL 但对话无明确出图意向 → deny 要求 ASKVIZ。"""
    _lead_messages.set(
        [
            HumanMessage(content="帮我分析一下这个 EPM 数据"),
            AIMessage(content="[intent] E2E_FULL\n我开始全流程包括出图。"),
        ]
    )
    decision = provider.evaluate(_task_request())
    assert not decision.allow
    reason = decision.reasons[0]
    assert reason.code == "ethoinsight.e2e_full_requires_explicit_viz_intent"
    # deny 消息必须指明应改用 E2E_FULL_ASKVIZ（directed deny，spec §1）
    assert "E2E_FULL_ASKVIZ" in reason.message


def test_allow_e2e_full_with_explicit_viz_intent(provider):
    """用户明确「也要图」→ E2E_FULL 放行（不强制反问）。"""
    _lead_messages.set(
        [
            HumanMessage(content="帮我分析这个数据，顺便把图画出来"),
            AIMessage(content="[intent] E2E_FULL\n用户要图，直接画。"),
        ]
    )
    decision = provider.evaluate(_task_request())
    assert decision.allow


def test_allow_e2e_full_askviz_default(provider):
    """默认 ASKVIZ → 放行（不触发 E2E_FULL 校验）。"""
    _lead_messages.set(
        [
            HumanMessage(content="帮我看看这个数据"),
            AIMessage(content="[intent] E2E_FULL_ASKVIZ\n跑完解读再问要不要图。"),
        ]
    )
    decision = provider.evaluate(_task_request())
    assert decision.allow


def test_viz_keywords_ssot_nonempty():
    """SSOT 守护：VIZ_INTENT_KEYWORDS 必须存在且非空（prompt 与 provider 共享）。"""
    assert VIZ_INTENT_KEYWORDS
    assert isinstance(VIZ_INTENT_KEYWORDS, (list, tuple, frozenset, set))
    # 至少覆盖 prompt 列举的核心词，避免漂移
    as_list = list(VIZ_INTENT_KEYWORDS)
    assert "图" in as_list or any("图" in str(k) for k in as_list)


def test_e2e_full_viz_intent_detected_from_user_not_lead_claim(provider):
    """reward-hacking（§六.1）：lead 在 AIMessage 里自述「用户要图」不算数，
    必须在 HumanMessage 实际文本里检测到出图意向才放行。"""
    _lead_messages.set(
        [
            HumanMessage(content="分析下数据"),  # 用户没提图
            AIMessage(content="用户说要图，所以我声明 [intent] E2E_FULL"),
        ]
    )
    decision = provider.evaluate(_task_request())
    assert not decision.allow  # 不信 lead 自述


# ── 子改动 B：PATHS ask 顺序强制 ───────────────────────────────────────────────


def _gate_provider() -> IntentPostStepAskGateProvider:
    return IntentPostStepAskGateProvider()


def _write_handoff(tmp_path, name: str):
    (tmp_path / f"handoff_{name}.json").write_text('{"status":"completed"}')


def _write_ctx(tmp_path, gate_completed: list[str]):
    (tmp_path / "experiment-context.json").write_text(json.dumps({"paradigm": "epm", "gate_completed": gate_completed}))


def test_ask_order_enforced(tmp_path):
    """核心：E2E_FULL_ASKVIZ 里 viz 在 report 前。lead 先完成 report(gate4) 却跳过 viz(gate3)
    → 派 chart-maker（viz 之后）时 deny。"""
    p = _gate_provider()
    # data-analyst done（ask(viz) 的前置 dispatch 完成）
    _write_handoff(tmp_path, "data_analyst")
    # report gate4 已完成但 viz gate3 未完成 = 乱序
    _write_ctx(tmp_path, ["gate1_paradigm", "gate4_report_acknowledged"])

    _gate_lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n直接出图")])
    _lead_workspace.set(str(tmp_path))

    decision = p.evaluate(
        GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "chart-maker", "description": "x", "prompt": "x"},
        )
    )
    assert not decision.allow
    reason = decision.reasons[0]
    assert reason.code == "ethoinsight.ask_order_violation"
    # deny 消息指明前置 ask 未完成（directed）
    assert "ask(viz" in reason.message or "viz" in reason.message


def test_batch_answer_not_false_denied(tmp_path):
    """守边界（§2.3）：用户一次性回答多个问题（viz + report 都已落盘）→ 不误拦。
    这是合法的批量应答，gate3 和 gate4 都在 gate_completed 里，顺序检查应放行。"""
    p = _gate_provider()
    _write_handoff(tmp_path, "data_analyst")
    # 用户一次性回答了 viz=yes 和 report=yes，两个 gate 都已落盘 → 合法批量应答
    _write_ctx(
        tmp_path,
        [
            "gate1_paradigm",
            "gate3_viz_acknowledged",
            "gate4_report_acknowledged",
        ],
    )

    _gate_lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ\n用户一次答完")])
    _lead_workspace.set(str(tmp_path))

    # 派 chart-maker（viz=yes 分支，gate3 已落盘）应放行
    decision = p.evaluate(
        GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "chart-maker", "description": "x", "prompt": "x"},
        )
    )
    assert decision.allow


# ── smoke：两 provider 实例化 + 喂合成 request 跑通不抛 ──────────────────────────


def test_smoke_both_providers_instantiate_and_evaluate():
    """smoke（spec §四 test 6）：两 provider 实例化 + 喂合成 request 跑通不抛。"""
    intent_provider = IntentClassificationGuardrailProvider()
    gate_provider = IntentPostStepAskGateProvider()

    req = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "code-executor"})
    # 不论 allow 与否，两 provider 都不应抛异常
    d1 = intent_provider.evaluate(req)
    d2 = gate_provider.evaluate(req)
    assert isinstance(d1.allow, bool)
    assert isinstance(d2.allow, bool)


# ── scope 守护：identify status 四态逻辑不变 ───────────────────────────────────


def test_identify_status_logic_unchanged():
    """守 scope（§三.3）：identify status 四态逻辑不在本 spec 改动面。
    验证 PATHS 仍声明 8 个 intent、ASK_GATE_MAP 不变、子改动未触碰 identify 路径。"""
    from deerflow.guardrails.path_registry import ASK_GATE_MAP, PATHS

    # 8 intent 不变（本 spec 不增删 intent）
    assert len(PATHS) == 8
    assert "E2E_FULL" in PATHS
    assert "E2E_FULL_ASKVIZ" in PATHS

    # ASK_GATE_MAP 三键不变（本 spec 不改 gate 命名）
    assert set(ASK_GATE_MAP.keys()) == {"viz", "report", "four_choice"}
    assert ASK_GATE_MAP["viz"] == "gate3_viz_acknowledged"
    assert ASK_GATE_MAP["report"] == "gate4_report_acknowledged"
