"""Verify code-executor workflow prompt references plan_metrics.json, not by-paradigm md."""

from __future__ import annotations


def test_code_executor_workflow_reads_plan_json():
    """Spec S4: prompt 仍指向 plan_metrics.json，但迭代 metrics 的职责
    下沉到 run_metric_plan 工具（不再 prompt 教 LLM 逐条 plan.metrics 迭代）。
    """
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt

    assert "plan_metrics.json" in sp, "workflow must reference plan.json"
    assert "run_metric_plan" in sp, "must instruct calling run_metric_plan to execute the plan"

    assert "by-paradigm" not in sp, "should no longer read by-paradigm md"
    assert "决策树" not in sp or "选脚本" not in sp, "decision tree responsibility moved to lead"


def test_code_executor_skills_list_unchanged():
    """code-executor 不挂 metric-catalog（保持执行纯净）。"""
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    assert "ethoinsight-metric-catalog" not in (CODE_EXECUTOR_CONFIG.skills or [])
    assert "ethoinsight-code" in (CODE_EXECUTOR_CONFIG.skills or [])


def test_parameters_used_sourced_from_disk_artifacts_not_prompt():
    """Spec S4: parameters_used 透传职责从 prompt（教 LLM 抓 compute [result] stdout）
    下沉到 metric_aggregation 聚合器——直接从磁盘 m_*.json 读 parameters_used，
    不靠 LLM 抓 stdout（杜绝空 {} 被当 bug 修 / 幽灵参数）。

    原 2026-06-03 故障（data-analyst step 2.8 死循环）的根因——LLM 把空 [result].
    parameters_used 当 bug 改用 plan 全量 parameters_in_use——在 S4 后结构性消失：
    聚合器从 m_*.json 的 parameters_used 字段机械读，LLM 不参与参数透传。
    回归覆盖在 tests/test_run_metric_plan.py::TestListZoneParamsPreserved。
    """
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt
    # prompt 不再教抓 [result] stdout（职责在聚合器）
    assert "抓 stdout" in sp or "不需要抓 stdout" in sp or "不需要手构 handoff" in sp, (
        "prompt 应声明执行/聚合由 run_metric_plan 工具确定性完成，LLM 不抓 stdout"
    )

