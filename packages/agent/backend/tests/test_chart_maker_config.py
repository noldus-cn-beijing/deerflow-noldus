"""W13: chart-maker SubagentConfig + 注册到 BUILTIN_SUBAGENTS。"""
from __future__ import annotations

from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY


def test_chart_maker_config_basic_fields():
    cfg = CHART_MAKER_CONFIG
    assert cfg.name == "chart-maker"
    assert "可视化" in cfg.description or "chart" in cfg.description.lower()
    assert cfg.model == "inherit"


def test_chart_maker_capability_metadata():
    cfg = CHART_MAKER_CONFIG
    assert cfg.when_to_use and "画图" in cfg.when_to_use
    assert cfg.input_contract and "chart" in cfg.input_contract.lower()
    assert cfg.output_contract and "handoff_chart_maker.json" in cfg.output_contract
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_chart_maker_tools():
    cfg = CHART_MAKER_CONFIG
    assert "bash" in cfg.tools
    assert "read_file" in cfg.tools
    assert "write_file" in cfg.tools
    assert "present_files" in cfg.tools
    assert "task" in (cfg.disallowed_tools or [])
    assert "ask_clarification" in (cfg.disallowed_tools or [])


def test_chart_maker_skills():
    cfg = CHART_MAKER_CONFIG
    assert "ethoinsight" in cfg.skills
    assert "ethoinsight-chart-maker" in cfg.skills


def test_chart_maker_registered_in_builtins():
    assert "chart-maker" in BUILTIN_SUBAGENTS
    assert BUILTIN_SUBAGENTS["chart-maker"] is CHART_MAKER_CONFIG


def test_chart_maker_handoff_registered():
    assert "chart_maker" in HANDOFF_FILE_REGISTRY
    assert HANDOFF_FILE_REGISTRY["chart_maker"] == "handoff_chart_maker.json"


def test_chart_maker_system_prompt_workflow():
    cfg = CHART_MAKER_CONFIG
    assert "execution-conventions" in cfg.system_prompt
    assert "ethoinsight-chart-maker" in cfg.system_prompt
    assert "catalog.resolve" in cfg.system_prompt
    assert "--mode charts" in cfg.system_prompt
    assert "handoff_chart_maker.json" in cfg.system_prompt
    assert "present_files" in cfg.system_prompt
