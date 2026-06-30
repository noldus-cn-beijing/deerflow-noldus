"""TDD tests for StageNarrationMiddleware —— A1 stage_plan 发射（承重墙）。

对应 spec 2026-06-30-... 模块 2 + 验收标准 2（stage_plan 一次性、非流水线不发）。

被测：``deerflow.agents.middlewares.stage_narration_middleware.StageNarrationMiddleware``
- after_model 读 messages → 提取 latest [intent] → 首次见 pipeline intent 时发 stage_plan
- 同一 intent 不重复发（幂等）
- 非流水线意图（QA/CHART/REPORT/CLARIFY）不发
- n=1 时 stage_plan.skipped 含「数据解读」（n 由注入的 resolver 决定，可测）
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

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


# ---------------------------------------------------------------------------
# 缺口 1 —— 「识别范式」阶段的 wrap_tool_call 派发观测点
#
# spec 2026-06-30-a1-stage-narration-coverage-gap-fix 缺口 1：「识别范式」由 lead 自调工具
# 完成（identify_ev19_template / inspect_uploaded_file / prep_metric_plan），不派 subagent，
# 故 A1 既有 task 派遣观测点收不到它。StageNarrationMiddleware 覆盖 wrap_tool_call：
#   - identify_ev19_template / inspect_uploaded_file → 识别范式 active（调 handler 前）
#   - prep_metric_plan 返回 status=ok → 识别范式 completed（调 handler 后，grounded）
# ---------------------------------------------------------------------------


def _tool_call_request(name: str, args: dict | None = None, call_id: str = "call_1") -> ToolCallRequest:
    """Build a minimal ToolCallRequest for wrap_tool_call tests (no real tool needed)."""
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state={"messages": []},
        runtime=None,
    )


def _handler_returning(tool_message: ToolMessage):
    """Build a handler callable that ignores input and returns the given ToolMessage."""

    def _handler(_request):
        return tool_message

    return _handler


class TestIdentifyStageDispatch:
    def test_identify_ev19_template_fires_active(self):
        """调 identify_ev19_template → 在 handler 执行前发「识别范式」active。"""
        mw, emitted = _make_mw()
        req = _tool_call_request("identify_ev19_template", {"uploaded_files": ["/mnt/x.txt"], "user_message": "epm"})
        result = mw.wrap_tool_call(req, _handler_returning(ToolMessage(content="ok", tool_call_id="call_1")))
        acts = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "active"]
        assert len(acts) == 1
        # handler 仍正常执行（返回值透传）
        assert result.content == "ok"

    def test_inspect_uploaded_file_fires_active(self):
        """调 inspect_uploaded_file → 发「识别范式」active（识别阶段的探查工具）。"""
        mw, emitted = _make_mw()
        req = _tool_call_request("inspect_uploaded_file", {"paradigm": "epm"})
        mw.wrap_tool_call(req, _handler_returning(ToolMessage(content="ok", tool_call_id="call_1")))
        acts = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "active"]
        assert len(acts) == 1

    def test_prep_metric_plan_ok_fires_completed(self):
        """prep_metric_plan 返回 status=ok → 发「识别范式」completed（识别完成、即将派 code-executor）。"""
        mw, emitted = _make_mw()
        ok_result = {"status": "ok", "plan_path": "/mnt/.../plan_metrics.json", "plan_summary": {"subject_count": 2}}
        req = _tool_call_request("prep_metric_plan", {"uploaded_files": ["/mnt/x.txt"], "paradigm": "epm"})
        mw.wrap_tool_call(req, _handler_returning(ToolMessage(content=json.dumps(ok_result), tool_call_id="call_1")))
        dones = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "completed"]
        assert len(dones) == 1

    def test_prep_metric_plan_error_does_not_fire_completed(self):
        """prep_metric_plan 返回 status=error → **不发** completed（叙事不撒谎，grounded）。

        spec 验收：人为让识别失败 → 断言不发 completed。
        """
        mw, emitted = _make_mw()
        err_result = {"status": "error", "error_code": "columns_missing", "message": "..."}
        req = _tool_call_request("prep_metric_plan", {"uploaded_files": ["/mnt/x.txt"], "paradigm": "epm"})
        mw.wrap_tool_call(req, _handler_returning(ToolMessage(content=json.dumps(err_result), tool_call_id="call_1")))
        dones = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "completed"]
        assert dones == []

    def test_prep_metric_plan_ambiguous_does_not_fire_completed(self):
        """prep_metric_plan 返回非 ok 的任何状态（含解析失败/非 JSON）→ 不发 completed。"""
        mw, emitted = _make_mw()
        req = _tool_call_request("prep_metric_plan")
        # 非 JSON content → 解析失败 → 当作非 ok
        mw.wrap_tool_call(req, _handler_returning(ToolMessage(content="not json at all", tool_call_id="call_1")))
        dones = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "completed"]
        assert dones == []

    def test_unrelated_tool_fires_nothing(self):
        """非识别类工具（如 present_files）→ 不发任何 stage_update。"""
        mw, emitted = _make_mw()
        req = _tool_call_request("present_files", {"files": ["/mnt/.../report.md"]})
        mw.wrap_tool_call(req, _handler_returning(ToolMessage(content="ok", tool_call_id="call_1")))
        assert emitted == []

    def test_active_fires_only_once_until_completed(self):
        """多次 identify 调用 + 最终 prep ok：active 幂等（识别阶段已 active 就不重复发），completed 仅一次。"""
        mw, emitted = _make_mw()
        # 第一次 identify → active
        mw.wrap_tool_call(_tool_call_request("identify_ev19_template", call_id="c1"), _handler_returning(ToolMessage(content="ok", tool_call_id="c1")))
        # 第二次 identify（重试）→ 不重复发 active
        mw.wrap_tool_call(_tool_call_request("identify_ev19_template", call_id="c2"), _handler_returning(ToolMessage(content="ok", tool_call_id="c2")))
        # prep ok → completed
        mw.wrap_tool_call(_tool_call_request("prep_metric_plan", call_id="c3"), _handler_returning(ToolMessage(content=json.dumps({"status": "ok"}), tool_call_id="c3")))
        acts = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "active"]
        dones = [e for e in emitted if e.get("kind") == "stage_update" and e["stage"] == "识别范式" and e["status"] == "completed"]
        assert len(acts) == 1
        assert len(dones) == 1

    def test_narration_contains_no_viscera(self):
        """防 vacuous（spec 验收）：发出的 narration/阶段名不含工具名 / gate 关键字。"""
        mw, emitted = _make_mw()
        mw.wrap_tool_call(_tool_call_request("identify_ev19_template", call_id="c1"), _handler_returning(ToolMessage(content="ok", tool_call_id="c1")))
        mw.wrap_tool_call(_tool_call_request("prep_metric_plan", call_id="c2"), _handler_returning(ToolMessage(content=json.dumps({"status": "ok"}), tool_call_id="c2")))
        forbidden = ["identify_ev19_template", "prep_metric_plan", "inspect_uploaded_file", "ev19_template", "stage_update"]
        for e in emitted:
            for field in ("stage", "narration"):
                val = e.get(field, "")
                for bad in forbidden:
                    assert bad not in str(val), f"{field}={val!r} 含内脏词 {bad!r}"

    def test_handler_exception_propagates(self):
        """wrap_tool_call 不吞 handler 异常（守 langchain 契约：异常由 ToolNode handle_tool_errors 处理）。"""
        mw, _emitted = _make_mw()

        def boom(_request):
            raise RuntimeError("tool exploded")

        with pytest.raises(RuntimeError, match="tool exploded"):
            mw.wrap_tool_call(_tool_call_request("identify_ev19_template"), boom)
