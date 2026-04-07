"""Integration tests for the analysis template tool within the DeerFlow agent system.

Covers:
- Tool resolves correctly via resolve_variable()
- Tool appears in code-executor's filtered tool list
- Tool does NOT appear in data-analyst/report-writer tool lists
- Code-executor SubagentConfig includes get_analysis_template in tools
"""

import pytest

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS


class TestCodeExecutorConfig:
    """Verify code-executor config includes the template tool."""

    def test_code_executor_has_template_tool(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        assert "get_analysis_template" in config.tools

    def test_code_executor_has_all_expected_tools(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        expected = {"bash", "read_file", "write_file", "ls", "str_replace", "get_analysis_template"}
        assert set(config.tools) == expected

    def test_data_analyst_no_template_tool(self):
        config = BUILTIN_SUBAGENTS["data-analyst"]
        if config.tools is not None:
            assert "get_analysis_template" not in config.tools

    def test_report_writer_no_template_tool(self):
        config = BUILTIN_SUBAGENTS["report-writer"]
        if config.tools is not None:
            assert "get_analysis_template" not in config.tools


class TestCodeExecutorPrompt:
    """Verify code-executor system prompt references the template tool."""

    def test_prompt_mentions_get_analysis_template(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        assert "get_analysis_template" in config.system_prompt

    def test_prompt_has_customizable_rules(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        assert "CUSTOMIZABLE" in config.system_prompt

    def test_prompt_has_correct_workflow_order(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        prompt = config.system_prompt
        # get_analysis_template should come before write_file and bash in workflow
        template_pos = prompt.find("get_analysis_template")
        write_pos = prompt.find("write_file")
        bash_pos = prompt.find("bash(")
        assert template_pos < write_pos < bash_pos

    def test_prompt_has_wrong_example(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        # Check for wrong example section (may be in Chinese or English)
        assert "❌" in config.system_prompt or "<wrong_example>" in config.system_prompt

    def test_prompt_has_fallback_library_section(self):
        config = BUILTIN_SUBAGENTS["code-executor"]
        assert "ethoinsight" in config.system_prompt.lower()
        # Check for fallback indicator (Chinese or English)
        assert "备用" in config.system_prompt or "FALLBACK" in config.system_prompt


class TestTemplateToolResolution:
    """Test that the tool can be resolved by the reflection system."""

    def test_tool_import(self):
        """Verify the tool can be imported from its config.yaml path."""
        from ethoinsight.templates.tool import get_analysis_template_tool
        assert get_analysis_template_tool.name == "get_analysis_template"

    def test_tool_is_langchain_base_tool(self):
        from langchain_core.tools import BaseTool
        from ethoinsight.templates.tool import get_analysis_template_tool
        assert isinstance(get_analysis_template_tool, BaseTool)

    def test_resolve_variable_path(self):
        """Simulate what config.yaml tool loading does."""
        from deerflow.reflection.resolvers import resolve_variable
        from langchain_core.tools import BaseTool

        tool = resolve_variable("ethoinsight.templates.tool:get_analysis_template_tool", BaseTool)
        assert tool.name == "get_analysis_template"
