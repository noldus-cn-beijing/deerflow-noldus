"""W11: code-executor SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG


def test_capability_metadata_set():
    cfg = CODE_EXECUTOR_CONFIG
    assert cfg.when_to_use is not None and "分析" in cfg.when_to_use
    assert cfg.input_contract is not None and "plan_metrics" in cfg.input_contract
    assert cfg.output_contract is not None and "handoff_code_executor.json" in cfg.output_contract
    assert "gate_signals" in cfg.output_contract
    assert cfg.required_upstream_handoffs == []


def test_skills_no_longer_include_ethoinsight_charts():
    cfg = CODE_EXECUTOR_CONFIG
    assert cfg.skills is not None
    assert "ethoinsight-charts" not in cfg.skills
    assert "ethoinsight-code" in cfg.skills


def test_system_prompt_reads_plan_metrics_not_metric_plan():
    cfg = CODE_EXECUTOR_CONFIG
    assert "plan_metrics.json" in cfg.system_prompt
    assert "metric_plan.json" not in cfg.system_prompt


def test_system_prompt_no_longer_runs_charts():
    cfg = CODE_EXECUTOR_CONFIG
    assert "for chart in plan.charts" not in cfg.system_prompt
    assert "plan.charts" not in cfg.system_prompt


def test_system_prompt_still_runs_metrics_and_stats():
    cfg = CODE_EXECUTOR_CONFIG
    assert "plan.metrics" in cfg.system_prompt
    assert "plan.statistics" in cfg.system_prompt
    assert "handoff_code_executor.json" in cfg.system_prompt


def test_disallowed_tools_unchanged():
    cfg = CODE_EXECUTOR_CONFIG
    assert "task" in (cfg.disallowed_tools or [])
    assert "ask_clarification" in (cfg.disallowed_tools or [])
