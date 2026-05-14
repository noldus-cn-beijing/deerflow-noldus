"""验证 lead prompt 中含 Issue #3 角色边界硬约束段。

不是 e2e 测试 — 是 prompt 内容存在性检查，防 prompt 被改回旧版退化。
真正的 e2e 验证靠 Task 9 的 dogfood。
"""

from deerflow.agents.lead_agent.prompt import apply_prompt_template


def test_lead_prompt_forbids_self_interpretation_before_data_analyst() -> None:
    """lead 不应在 data-analyst 返回前自己写指标判读。"""
    p = apply_prompt_template(subagent_enabled=True)
    # 关键短语必须存在
    assert "handoff_data_analyst.json" in p, "prompt 必须显式引用 data-analyst handoff 文件名"
    assert "调度员" in p or "调度角色" in p, "prompt 必须将 lead 定位为调度员"


def test_lead_prompt_forbids_unsupported_metadata() -> None:
    """lead 不应引用用户未告知的元数据（品系/体重等）。"""
    p = apply_prompt_template(subagent_enabled=True)
    # 必须显式禁止未告知元数据
    assert "品系" in p, "prompt 必须显式提到品系约束"
    assert "raw" in p.lower() or "headers" in p.lower(), "prompt 必须说明合法来源（raw file headers）"


def test_lead_prompt_forbids_absolute_reference_terms() -> None:
    """lead 不应引用'典型值/常模/金标准'等绝对参考词。"""
    p = apply_prompt_template(subagent_enabled=True)
    # 必须出现这些禁词的明确禁令（出现说明 prompt 提到了它们要禁）
    forbidden_in_constraint_section = ["典型值", "常模", "金标准"]
    for word in forbidden_in_constraint_section:
        assert word in p, f"prompt 必须显式提到禁词 '{word}'（哪怕是禁令上下文）"


def test_lead_prompt_provides_positive_example() -> None:
    """prompt 必须给出 lead 应该如何呈现指标的正例。"""
    p = apply_prompt_template(subagent_enabled=True)
    # 期望含"正例"或"正确做法"等关键词
    assert "正例" in p or "正确做法" in p or "正确示例" in p, (
        "prompt 必须给出 lead 在没收到 data-analyst 结果时如何呈现指标的正例"
    )
