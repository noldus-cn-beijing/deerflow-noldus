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
    # Spec 3 (P3): plan_charts.json 改由 prep_chart_plan 工具生成（内部自读 context
    # 拿 column_aliases / groups），不再 bash 拼 catalog.resolve。
    assert "prep_chart_plan" in cfg.system_prompt
    assert "handoff_chart_maker.json" in cfg.system_prompt
    assert "present_files" in cfg.system_prompt


def test_chart_maker_has_prep_chart_plan_tool():
    """Spec 3: chart-maker 工具集含 prep_chart_plan（取代 bash 拼 catalog.resolve）。"""
    assert "prep_chart_plan" in CHART_MAKER_CONFIG.tools


def test_chart_maker_system_prompt_has_parallel_plot_guidance():
    """2026-06-18: chart-maker 教会用 code-executor 同款的 bash -c "... & ... & wait" 并行绘图
    （spec 2026-06-18-chart-maker-parallel-plotting §3.1）。N 张互不依赖的图应 1 turn 跑完，
    而非 N 个串行 turn。"""
    p = CHART_MAKER_CONFIG.system_prompt
    assert "bash -c" in p
    assert "&" in p
    assert "wait" in p


def test_chart_maker_system_prompt_no_bash_bundle_dead_guidance():
    """2026-06-18: chart-maker 受 guardrail 限制（bash 仅可 python -m ethoinsight.scripts.*
    + 文件操作），``bash cat ... > bundle`` 含禁字符 ``>`` 会被 guardrail 直接拦——这条指引
    从来跑不通、还浪费 turn。必须删掉，改为逐个 read_file。"""
    p = CHART_MAKER_CONFIG.system_prompt
    assert "cm_context_bundle" not in p
    assert "bash cat /mnt/user-data/workspace/handoff" not in p
