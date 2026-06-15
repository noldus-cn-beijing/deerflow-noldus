"""Tests for lead agent tool filtering (Task 3: remove bash/write_file/str_replace)."""

from langchain.tools import BaseTool
from langchain_core.tools import tool as tool_decorator

from deerflow.agents.lead_agent.agent import _LEAD_EXCLUDED_TOOLS, _filter_lead_tools


def _make_named_tool(name: str) -> BaseTool:
    """Build a minimal BaseTool with a given .name attribute."""
    @tool_decorator(name, parse_docstring=False)
    def fn(x: str) -> str:
        """noop."""
        return x
    return fn


class TestFilterLeadToolsPureFunction:
    """纯函数测试：不 patch agent 工厂，直接打 _filter_lead_tools。"""

    def test_excludes_bash(self):
        tools = [_make_named_tool("bash"), _make_named_tool("read_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        names = {t.name for t in result}
        assert "bash" not in names
        assert "read_file" in names

    def test_excludes_write_file(self):
        tools = [_make_named_tool("write_file"), _make_named_tool("read_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "write_file" not in {t.name for t in result}

    def test_excludes_str_replace(self):
        tools = [_make_named_tool("str_replace"), _make_named_tool("ls")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "str_replace" not in {t.name for t in result}

    def test_keeps_ls(self):
        """Q4 决策：lead 保留 ls 验证 code-executor 产物。"""
        tools = [_make_named_tool("ls"), _make_named_tool("bash")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "ls" in {t.name for t in result}

    def test_keeps_read_file(self):
        """lead 需要 read_file 看 handoff JSON。"""
        tools = [_make_named_tool("read_file"), _make_named_tool("write_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "read_file" in {t.name for t in result}

    def test_keeps_prep_metric_plan(self):
        """关键回归：prep_metric_plan 是 lead 替代 bash 调 parse/catalog 的唯一通道，绝不能被误加进 _LEAD_EXCLUDED_TOOLS。"""
        tools = [
            _make_named_tool("prep_metric_plan"),
            _make_named_tool("bash"),
            _make_named_tool("write_file"),
        ]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        names = {t.name for t in result}
        assert "prep_metric_plan" in names
        # 同时确认 bash / write_file 被过滤(防止把这条测试退化成空断言)
        assert "bash" not in names
        assert "write_file" not in names

    def test_excluded_set_is_frozen(self):
        """_LEAD_EXCLUDED_TOOLS 必须含三项，不多不少。"""
        assert _LEAD_EXCLUDED_TOOLS == frozenset({"bash", "write_file", "str_replace"})

    def test_empty_tools_returns_empty(self):
        assert _filter_lead_tools([], _LEAD_EXCLUDED_TOOLS) == []

    def test_no_excluded_tools_returns_all(self):
        tools = [_make_named_tool("read_file"), _make_named_tool("task")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert {t.name for t in result} == {"read_file", "task"}


class TestSubagentToolsUnchanged:
    """子代理工具列表与 _filter_lead_tools 独立 —— 通过 SubagentConfig.tools 显式声明。

    Spec S4 (2026-06-12): code-executor 工具粒度从裸 bash 改为 run_metric_plan 确定性工具。
    bash/write_file/str_replace 已收走（§2.1 红线），run_metric_plan 加入。
    """

    def test_code_executor_uses_run_metric_plan_not_bash(self):
        """S4: code-executor 不再用 bash，改用 run_metric_plan 确定性执行。"""
        from deerflow.subagents.registry import get_subagent_config
        config = get_subagent_config("code-executor")
        assert config is not None, "code-executor subagent 必须注册"
        assert config.tools is not None
        # run_metric_plan 是执行 metrics+stats 的唯一路径
        assert "run_metric_plan" in config.tools, f"run_metric_plan 必须在 tools: {config.tools}"
        # bash/write_file/str_replace 已收走（防 LLM 退回手拼命令老路）
        assert "bash" not in config.tools, f"S4: code-executor 不应再有 bash: {config.tools}"
        assert "write_file" not in config.tools
        assert "str_replace" not in config.tools
        # 显式 disallow 双保险（防被任何机制重新挂回）
        assert "bash" in (config.disallowed_tools or [])

