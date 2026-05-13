"""Acceptance tests for lead prompt pipeline redesign (commit 5).

Locks in three guarantees from docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md:
  1. Default pipeline is 2 steps (code-executor → data-analyst) with
     ask_clarification deciding whether to proceed to report-writer.
  2. Every subagent has an ask_clarification failure path, not silent bypass.
  3. Lead prompt enforces "process transparency" — user-facing narration
     before each subagent dispatch.
"""

from __future__ import annotations

from deerflow.agents.lead_agent.prompt import _build_subagent_section


def _section() -> str:
    """Render the subagent section with deterministic args for text assertions."""
    return _build_subagent_section(max_concurrent=3)


class TestDefaultPipelineIsTwoStep:
    def test_default_flow_documented_as_two_subagents_plus_clarify(self):
        section = _section()
        # The phrase "默认派遣顺序" must declare code-executor → data-analyst only.
        assert "默认派遣顺序" in section
        # report-writer must be explicitly called out as not-default.
        assert "report-writer 不是默认步骤" in section

    def test_usage_example_shows_ask_clarification_between_analyst_and_writer(self):
        section = _section()
        # The Step 3 flow must include ask_clarification with 3 options
        # before any report-writer dispatch in Example 1.
        assert "需要结构化研究报告" in section
        assert "不需要，谢谢" in section

    def test_orchestration_guide_documents_conditional_report_writer(self):
        section = _section()
        # The flow must call out "present → ask_clarification → optional report-writer"
        # (phrased as 自然语言呈现 + ask_clarification three-choice).
        assert "自然语言呈现" in section
        assert "ask_clarification" in section

    def test_skip_code_executor_branch_preserved(self):
        """Users saying '只帮我重新写个报告' should still trigger direct report-writer."""
        section = _section()
        assert "只帮我重新写个报告" in section


class TestUnifiedFailureHandling:
    def test_code_executor_failure_routes_to_ask_clarification(self):
        section = _section()
        assert "code-executor 失败" in section
        # Must include the paradigm / format / params / timeout classification.
        assert "范式不支持" in section
        assert "文件解析失败" in section

    def test_data_analyst_failure_has_three_option_ask_clarification(self):
        section = _section()
        assert "data-analyst 失败" in section
        # Required option labels from plan §4.
        assert "跳过专家解读" in section
        assert "中止" in section

    def test_report_writer_failure_has_three_option_ask_clarification(self):
        section = _section()
        assert "report-writer 失败" in section
        assert "只要分析洞察就够了" in section or "不要报告" in section

    def test_forbids_silent_bypass_on_analyst_failure(self):
        section = _section()
        # Plan explicitly forbids bypassing data-analyst to reach report-writer.
        assert "bypass" in section.lower() or "静默" in section or "硬写" in section


class TestProcessTransparency:
    def test_process_transparency_section_exists(self):
        section = _section()
        assert "过程透明" in section

    def test_mentions_dont_expose_subagent_names(self):
        """Lead should say '正在请专家解读' not '调用 data-analyst'."""
        section = _section()
        assert "正在请专家解读" in section or "面向研究员用户" in section

    def test_presentation_template_required_before_clarification(self):
        section = _section()
        assert "分析结果呈现模板" in section
        # Must list 关键指标, 关键洞察, 数据质量提示.
        assert "关键指标" in section
        assert "关键洞察" in section


class TestOrchestrationGuideStep3:
    """Orchestration guide (injected via apply_prompt_template at runtime)
    must describe Step 3 as 'present + ask_clarification', not 'dispatch report-writer'.

    We can't run apply_prompt_template easily in unit tests, so we parse the
    orchestration_guide source string directly from prompt.py.
    """

    def _orchestration_source(self) -> str:
        from pathlib import Path

        p = Path(__file__).resolve().parent.parent / "packages" / "harness" / "deerflow" / "agents" / "lead_agent" / "prompt.py"
        return p.read_text(encoding="utf-8")

    def test_step_3_is_present_and_clarify_not_report_writer(self):
        src = self._orchestration_source()
        # Step 3 heading must exist.
        assert "### Step 3: 自然语言呈现 + ask_clarification" in src
        # And should not re-appear as "Step 3: 派遣 report-writer".
        assert "### Step 3: 派遣 report-writer" not in src

    def test_step_4a_is_report_writer_when_user_requests(self):
        src = self._orchestration_source()
        assert "Step 4a" in src
        assert "报告" in src

    def test_skip_pipeline_branch_preserved(self):
        src = self._orchestration_source()
        assert "只帮我重新写个报告" in src
