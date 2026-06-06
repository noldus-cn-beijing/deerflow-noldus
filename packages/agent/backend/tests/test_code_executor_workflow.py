"""Verify code-executor workflow prompt references plan_metrics.json, not by-paradigm md."""

from __future__ import annotations


def test_code_executor_workflow_reads_plan_json():
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt

    assert "plan_metrics.json" in sp, "workflow must reference plan.json"
    assert "plan.metrics" in sp or "metrics 数组" in sp, "must describe iterating metrics"

    assert "by-paradigm" not in sp, "should no longer read by-paradigm md"
    assert "决策树" not in sp or "选脚本" not in sp, "decision tree responsibility moved to lead"


def test_code_executor_skills_list_unchanged():
    """code-executor 不挂 metric-catalog（保持执行纯净）。"""
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    assert "ethoinsight-metric-catalog" not in (CODE_EXECUTOR_CONFIG.skills or [])
    assert "ethoinsight-code" in (CODE_EXECUTOR_CONFIG.skills or [])


def test_parameters_used_passthrough_from_compute_result_not_plan():
    """parameters_used 必须逐字透传 compute [result]，空 {} 是正确结果，不得回退用 plan 的 parameters_in_use。

    2026-06-03 实证：data-analyst step 2.8 死循环的更上游源——code-executor LLM 把空的
    [result].parameters_used 当 bug，改用 plan_metrics.json 的全量 parameters_in_use，
    导致 metrics_summary 报 12 个未参与计算的幽灵参数。skill 必须正面说明空 {} 是正确且
    有意义的，并指明 [result] 是唯一真相源、plan 的 parameters_in_use 不是。
    """
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt

    # [result] 是 parameters_used 的权威/唯一真相源
    assert "parameters_used" in sp
    assert ("权威来源" in sp and "[result]" in sp) or ("唯一真相源" in sp), (
        "必须声明 parameters_used 以 compute [result] 为权威/唯一真相源"
    )
    # 空 {} 被正面解释为正确，而非"可能为空"这种暗示它是 quirk 的措辞
    assert "空" in sp and ("正确" in sp or "有意义" in sp), (
        "必须正面说明空 parameters_used 是正确且有意义的结果（杜绝 LLM 当 bug 修）"
    )
    # 明确点名 plan 的 parameters_in_use 不是真相源
    assert "parameters_in_use" in sp, (
        "必须点名 plan_metrics.json 的 parameters_in_use 只是'打算用'清单、非实际用到"
    )

