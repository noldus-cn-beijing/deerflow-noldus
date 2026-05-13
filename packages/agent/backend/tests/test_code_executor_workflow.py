"""Verify code-executor workflow prompt references metric_plan.json, not by-paradigm md."""

from __future__ import annotations


def test_code_executor_workflow_reads_plan_json():
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt

    assert "metric_plan.json" in sp, "workflow must reference plan.json"
    assert "plan.metrics" in sp or "metrics 数组" in sp, "must describe iterating metrics"

    assert "by-paradigm" not in sp, "should no longer read by-paradigm md"
    assert "决策树" not in sp or "选脚本" not in sp, "decision tree responsibility moved to lead"


def test_code_executor_skills_list_unchanged():
    """code-executor 不挂 metric-catalog（保持执行纯净）。"""
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    assert "ethoinsight-metric-catalog" not in (CODE_EXECUTOR_CONFIG.skills or [])
    assert "ethoinsight-code" in (CODE_EXECUTOR_CONFIG.skills or [])
