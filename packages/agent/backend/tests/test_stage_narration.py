"""TDD tests for stage_narration — A1 后端事件分轨地基的纯映射模块。

对应 spec 2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md
模块 3（两类 custom 事件 payload 设计）+ 验收标准 2/3。

被测对象：``deerflow.agents.middlewares.stage_narration``
- ``intent_to_stage_plan``：意图 + n → ``stage_plan`` payload（或 None 表示不发）
- ``stage_for_dispatch``：subagent_type → 人话阶段名（用于 stage_update）
- ``stage_update``：构造 stage_update payload
- 泄漏断言：narration/阶段名不含工具名、subagent 名、gate 关键字
"""

from __future__ import annotations

import pytest

from deerflow.agents.middlewares import stage_narration


# ---------------------------------------------------------------------------
# intent_to_stage_plan —— 意图 + n → stage_plan
# ---------------------------------------------------------------------------
class TestIntentToStagePlan:
    def test_e2e_full_yields_5_stages(self):
        """E2E_FULL → 识别/计算/解读/画图/报告（5 阶段，无 skipped）。"""
        plan = stage_narration.intent_to_stage_plan("E2E_FULL", n=6)
        assert plan is not None
        assert plan["kind"] == "stage_plan"
        assert plan["stages"] == ["识别范式", "计算指标", "数据解读", "生成图表", "撰写报告"]
        assert plan["skipped"] == []

    def test_e2e_min_yields_2_stages(self):
        """E2E_MIN → 识别/计算（2 阶段）。"""
        plan = stage_narration.intent_to_stage_plan("E2E_MIN", n=6)
        assert plan is not None
        assert plan["stages"] == ["识别范式", "计算指标"]
        assert plan["skipped"] == []

    def test_e2e_full_askviz_yields_stages_without_report_initially(self):
        """E2E_FULL_ASKVIZ：出图前先反问；stage_plan 仍列出完整阶段集，
        前端据此渲染 stepper（出图/报告是否进行由 stage_update 驱动）。

        spec 模块 3：ASKVIZ → 识别/计算/解读/(询问)，用户答后按 viz_choice 追加 画图/报告。
        stage_plan 是意图确定后发一次的「计划」，列全阶段；动态性由 stage_update 表达。
        """
        plan = stage_narration.intent_to_stage_plan("E2E_FULL_ASKVIZ", n=6)
        assert plan is not None
        # 必含核心阶段
        assert "识别范式" in plan["stages"]
        assert "计算指标" in plan["stages"]
        assert "数据解读" in plan["stages"]

    def test_n1_marks_interpretation_skipped(self):
        """n=1 单样本 → stage_plan.skipped 含「数据解读」（无统计基础）。"""
        plan = stage_narration.intent_to_stage_plan("E2E_FULL", n=1)
        assert plan is not None
        assert "数据解读" in plan["skipped"]

    def test_n1_e2e_min_no_interpretation_to_skip(self):
        """E2E_MIN 本就无数据解读阶段；n=1 时 skipped 不应凭空塞入。"""
        plan = stage_narration.intent_to_stage_plan("E2E_MIN", n=1)
        assert plan is not None
        assert "数据解读" not in plan["stages"]
        # skipped 只标注「该阶段集里存在但被跳过」的阶段
        for s in plan["skipped"]:
            assert s in plan["stages"]

    @pytest.mark.parametrize("intent", ["CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"])
    def test_non_pipeline_intents_emit_no_plan(self, intent):
        """知识问答 / 闲聊 / 单步追问 / 单步出图报告 → 不发 stage_plan（前端无 stepper）。"""
        assert stage_narration.intent_to_stage_plan(intent, n=6) is None

    def test_unknown_intent_emits_no_plan(self):
        assert stage_narration.intent_to_stage_plan("BOGUS", n=6) is None

    def test_no_intent_emits_no_plan(self):
        assert stage_narration.intent_to_stage_plan(None, n=6) is None


# ---------------------------------------------------------------------------
# stage_for_dispatch —— subagent_type → 人话阶段名
# ---------------------------------------------------------------------------
class TestStageForDispatch:
    def test_code_executor_maps_to_compute(self):
        assert stage_narration.stage_for_dispatch("code-executor") == "计算指标"

    def test_data_analyst_maps_to_interpretation(self):
        assert stage_narration.stage_for_dispatch("data-analyst") == "数据解读"

    def test_chart_maker_maps_to_charts(self):
        assert stage_narration.stage_for_dispatch("chart-maker") == "生成图表"

    def test_report_writer_maps_to_report(self):
        assert stage_narration.stage_for_dispatch("report-writer") == "撰写报告"

    def test_unknown_subagent_returns_none(self):
        """未登记的 subagent 不发 stage_update（不猜阶段名）。"""
        assert stage_narration.stage_for_dispatch("mystery-agent") is None


# ---------------------------------------------------------------------------
# stage_update —— 构造 stage_update payload
# ---------------------------------------------------------------------------
class TestStageUpdate:
    def test_active_update(self):
        upd = stage_narration.stage_update("生成图表", "active", narration="正在生成图表…")
        assert upd == {
            "kind": "stage_update",
            "stage": "生成图表",
            "status": "active",
            "narration": "正在生成图表…",
        }

    def test_completed_update(self):
        upd = stage_narration.stage_update("计算指标", "completed")
        assert upd["kind"] == "stage_update"
        assert upd["stage"] == "计算指标"
        assert upd["status"] == "completed"

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            stage_narration.stage_update("计算指标", "bogus")


# ---------------------------------------------------------------------------
# 防泄漏（vacuous 防护）—— narration/阶段名永不携带内脏
# ---------------------------------------------------------------------------
class TestNoVisceraLeakage:
    """spec 验收 2 防 vacuous：工具名 / 子 agent 名 / gate 关键字永不进入面向用户的轨。

    这是被测断言，不是口头约定。任何 stage_plan/stage_update 的「人话字段」
    （stages / skipped / stage / narration）都不得含下列内脏词。
    """

    FORBIDDEN = [
        # 工具名
        "identify_ev19_template",
        "set_experiment_paradigm",
        "run_metric_plan",
        "run_chart_plan",
        "catalog.resolve",
        # subagent 名（机器侧连字符 + handoff 侧下划线两种形态）
        "code-executor",
        "code_executor",
        "data-analyst",
        "data_analyst",
        "chart-maker",
        "chart_maker",
        "report-writer",
        "report_writer",
        "knowledge-assistant",
        # gate / 内部状态机关键字
        "gate_signals",
        "gate3_viz_acknowledged",
        "ev19_template",
        "path_sequence",
        "ask_clarification",
        "handoff_code_executor",
    ]

    def _all_human_strings(self) -> list[str]:
        """收集所有 stage_narration 可能产出的人话字段。"""
        out: list[str] = []
        for intent in ["E2E_FULL", "E2E_FULL_ASKVIZ", "E2E_MIN"]:
            plan = stage_narration.intent_to_stage_plan(intent, n=6)
            assert plan is not None
            out.extend(plan["stages"])
            out.extend(plan["skipped"])
        # dispatch 阶段名
        for sub in ["code-executor", "data-analyst", "chart-maker", "report-writer"]:
            name = stage_narration.stage_for_dispatch(sub)
            if name:
                out.append(name)
        return out

    def test_stage_names_contain_no_viscera(self):
        for s in self._all_human_strings():
            for bad in self.FORBIDDEN:
                assert bad not in s, f"人话字段 {s!r} 含内脏词 {bad!r}"

    def test_default_narration_templates_contain_no_viscera(self):
        """模块自带的默认 narration 模板（如有）不得含内脏词。"""
        for tmpl in stage_narration.DEFAULT_NARRATIONS.values():
            for bad in self.FORBIDDEN:
                assert bad not in tmpl, f"默认 narration {tmpl!r} 含内脏词 {bad!r}"
