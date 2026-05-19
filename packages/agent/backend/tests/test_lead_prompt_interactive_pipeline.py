"""Acceptance tests for lead prompt pipeline design (W16 update).

W16(2026-05-18): Pipeline 细节(默认派遣顺序/report-writer 非默认/失败处理问句等)
已从 lead prompt 迁移到 ethoinsight-lead-interaction skill(W8 已建)。
本文件改为验证:
  (A) _build_subagent_section 渲染 capability-exposure + 5 条硬约束 + 意图状态机
  (B) prompt 引用 ethoinsight-lead-interaction skill
  (C) 旧 inline 细节已不在 prompt 中
"""
from __future__ import annotations

from pathlib import Path

from deerflow.agents.lead_agent.prompt import _build_subagent_section, apply_prompt_template


def _section() -> str:
    return _build_subagent_section(max_concurrent=3)


def _prompt() -> str:
    return apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)


def _orchestration_source() -> str:
    p = Path(__file__).resolve().parent.parent / "packages" / "harness" / "deerflow" / "agents" / "lead_agent" / "prompt.py"
    return p.read_text(encoding="utf-8")


class TestCapabilityExposure:
    """W16: _build_subagent_section 渲染 5 subagent + 5 条硬约束 + 意图状态机。"""

    def test_renders_all_five_noldus_subagents(self):
        section = _section()
        for name in ["code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"]:
            assert name in section, f"subagent '{name}' missing from section"

    def test_renders_five_hard_constraints(self):
        section = _section()
        assert "[intent]" in section, "missing intent classification"
        assert "Gate before guess" in section, "missing EV19 gate"
        assert "set_experiment_paradigm" in section, "missing ev19_template precondition"
        assert any(w in section.lower() for w in ["guardrail", "拦截"]), "missing guardrail ref"
        assert any(w in section for w in ["静默 bypass", "绝不静默", "硬写假结果"]), "missing silent-bypass ban"

    def test_renders_intent_state_machine(self):
        section = _section()
        for intent in ["E2E_FULL", "E2E_MIN", "CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"]:
            assert intent in section, f"intent '{intent}' missing from state machine"

    def test_renders_role_boundary_constraints(self):
        section = _section()
        assert "handoff_data_analyst.json" in section, "missing data-analyst handoff"
        assert any(w in section for w in ["典型值", "常模", "金标准"]), "missing forbidden-term list"
        assert "品系" in section, "missing metadata constraint"

    def test_renders_process_transparency_rule(self):
        section = _section()
        assert "播报" in section, "missing process transparency rule"


class TestNoOldInlineDetails:
    """W16: 旧 inline 细节(Step N / 默认派遣 / 失败处理 / 呈现模板)已不在 section 中。"""

    def test_no_default_pipeline_language(self):
        section = _section()
        assert "默认派遣顺序" not in section
        assert "report-writer 不是默认步骤" not in section

    def test_no_code_executor_failure_detail(self):
        section = _section()
        assert "code-executor 失败" not in section

    def test_no_data_analyst_failure_detail(self):
        section = _section()
        assert "data-analyst 失败" not in section

    def test_no_report_writer_failure_detail(self):
        section = _section()
        assert "report-writer 失败" not in section

    def test_no_analysis_result_template(self):
        section = _section()
        assert "分析结果呈现模板" not in section

    def test_no_old_clarification_options_language(self):
        section = _section()
        assert "需要结构化研究报告" not in section
        assert "不需要，谢谢" not in section

    def test_no_step_numbers_in_orchestration_guide(self):
        src = _orchestration_source()
        assert "### Step 1:" not in src
        assert "### Step 2:" not in src
        assert "### Step 3:" not in src


class TestSkillReferencePresent:
    """W16: lead prompt 引用 ethoinsight-lead-interaction skill。"""

    def test_prompt_references_lead_interaction_skill(self):
        prompt = _prompt()
        assert "ethoinsight-lead-interaction" in prompt

    def test_prompt_references_skill_md_path(self):
        prompt = _prompt()
        assert "/mnt/skills/ethoinsight-lead-interaction/SKILL.md" in prompt

    def test_section_references_lead_interaction_skill(self):
        section = _section()
        assert "ethoinsight-lead-interaction" in section
