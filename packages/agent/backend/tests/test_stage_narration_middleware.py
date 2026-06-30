"""TDD tests for StageNarrationMiddleware —— A1 stage_plan 发射（承重墙）。

对应 spec 2026-06-30-... 模块 2 + 验收标准 2（stage_plan 一次性、非流水线不发）。

被测：``deerflow.agents.middlewares.stage_narration_middleware.StageNarrationMiddleware``
- after_model 读 messages → 提取 latest [intent] → 首次见 pipeline intent 时发 stage_plan
- 同一 intent 不重复发（幂等）
- 非流水线意图（QA/CHART/REPORT/CLARIFY）不发
- n=1 时 stage_plan.skipped 含「数据解读」（n 由注入的 resolver 决定，可测）
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.stage_narration_middleware import StageNarrationMiddleware


def _ai_with_intent(intent: str | None, extra: str = "") -> AIMessage:
    content = f"[intent] {intent}\n{extra}" if intent else extra
    return AIMessage(content=content)


def _state(messages: list) -> dict:
    return {"messages": messages}


def _make_mw(n_resolver=None, writer=None) -> tuple[StageNarrationMiddleware, list]:
    """Build middleware with injectable n-resolver + captured writer.

    Returns (mw, emitted) where emitted collects every dict written to the custom track.
    """
    emitted: list[dict] = []

    def _writer(payload: dict) -> None:
        emitted.append(payload)

    mw = StageNarrationMiddleware(n_resolver=n_resolver, writer=writer or _writer)
    return mw, emitted


def _run_after_model(mw, messages, runtime=None):
    """after_model is sync; call it directly with a state dict."""
    return mw.after_model(_state(messages), runtime or MagicMock())


class TestStagePlanEmission:
    def test_emits_stage_plan_for_e2e_full(self):
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        _run_after_model(mw, [HumanMessage(content="分析这批数据并出图"), _ai_with_intent("E2E_FULL")])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        assert len(plans) == 1
        assert plans[0]["stages"] == ["识别范式", "计算指标", "数据解读", "生成图表", "撰写报告"]

    def test_emits_at_most_once_per_intent(self):
        """同一 intent 连续多轮 after_model → stage_plan 只发一次（幂等）。"""
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        ai = _ai_with_intent("E2E_FULL")
        _run_after_model(mw, [HumanMessage(content="x"), ai])
        _run_after_model(mw, [HumanMessage(content="x"), ai, ai])
        _run_after_model(mw, [HumanMessage(content="x"), ai, ai, ai])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        assert len(plans) == 1

    @pytest.mark.parametrize("intent", ["CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"])
    def test_non_pipeline_intent_emits_nothing(self, intent):
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent(intent)])
        assert emitted == []

    def test_no_intent_declared_emits_nothing(self):
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent(None, extra="你好")])
        assert emitted == []

    def test_n1_skips_interpretation(self):
        mw, emitted = _make_mw(n_resolver=lambda: 1)
        _run_after_model(mw, [HumanMessage(content="只有一个样本"), _ai_with_intent("E2E_FULL")])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        assert len(plans) == 1
        assert "数据解读" in plans[0]["skipped"]

    def test_e2e_min_stage_plan(self):
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        _run_after_model(mw, [HumanMessage(content="算下指标"), _ai_with_intent("E2E_MIN")])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        assert len(plans) == 1
        assert plans[0]["stages"] == ["识别范式", "计算指标"]


class TestStagePlanGrounding:
    def test_uses_latest_declared_intent(self):
        """lead 改注意图后，按最新 [intent] 发新的 stage_plan（前一份不重复发）。"""
        mw, emitted = _make_mw(n_resolver=lambda: 6)
        # 先声明 E2E_MIN
        _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent("E2E_MIN")])
        # 后改成 E2E_FULL（新一轮）
        _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent("E2E_MIN"), _ai_with_intent("E2E_FULL")])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        # 两份 plan：先 2 阶段，后 5 阶段
        assert [len(p["stages"]) for p in plans] == [2, 5]

    def test_writer_unavailable_does_not_crash(self):
        """get_stream_writer 在非 graph 上下文会抛 → 中间件吞掉、不崩 turn。"""
        def boom(_payload):
            raise RuntimeError("no stream writer context")

        mw, emitted = _make_mw(n_resolver=lambda: 6, writer=boom)
        # 不应抛
        ret = _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent("E2E_FULL")])
        assert ret is None  # after_model 不改 state

    def test_n_resolver_none_defaults_to_no_skip(self):
        """n_resolver 返回 None（n 未知）→ skipped 为空（不臆测 n=1）。"""
        mw, emitted = _make_mw(n_resolver=lambda: None)
        _run_after_model(mw, [HumanMessage(content="x"), _ai_with_intent("E2E_FULL")])
        plans = [e for e in emitted if e.get("kind") == "stage_plan"]
        assert len(plans) == 1
        assert plans[0]["skipped"] == []
