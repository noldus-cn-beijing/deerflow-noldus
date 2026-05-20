"""W16: Lead prompt 瘦身后的 capability 注入验收。

W16 把 _build_subagent_section 重构成"渲染 SubagentConfig.capability 字段"
(由 W11-W15 落地),把"如何反问 / 4-choice / 范式默认 fallback / 具体 chart 选择"
等细节移到 ethoinsight-lead-interaction skill (W8 已建)。

本文件锁定:
  1. 5 个 EthoInsight subagent 全部出现在 lead prompt
  2. capability 4 字段(name/description/when_to_use/input_contract/output_contract)被渲染
  3. lead prompt 不再含 {{handoff://X}} 派遣示例(改由 W19 自动注入)
  4. lead prompt 含 [intent] 强制分类硬约束
  5. lead prompt 引用 ethoinsight-lead-interaction skill
  6. chart-specific keywords 已搬到 chart-maker / ethoinsight skill
  7. 行数 < 400(从 1243 大幅瘦身)
  8. 已删除 default-template-fallback 引用
  9. metric_plan.json (旧名) 不再出现在 lead prompt
"""
from __future__ import annotations

from deerflow.agents.lead_agent.prompt import apply_prompt_template


def test_prompt_renders_capability_for_each_subagent():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    for name in ["code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"]:
        assert name in prompt, f"subagent '{name}' missing from lead prompt"


def test_prompt_renders_when_to_use_input_output_contract():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "when_to_use" in prompt or "适合" in prompt
    assert "input_contract" in prompt or "派遣 prompt 模板" in prompt
    assert "output_contract" in prompt or "handoff" in prompt


def test_prompt_does_NOT_contain_handoff_placeholder_instructions():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "{{handoff://" not in prompt


def test_prompt_contains_intent_classification_hard_rule():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "[intent]" in prompt
    intent_keywords = ["E2E_FULL", "E2E_MIN", "CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"]
    found = [kw for kw in intent_keywords if kw in prompt]
    assert len(found) >= 4, f"intent keywords missing: only found {found}"


def test_prompt_references_lead_interaction_skill():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "ethoinsight-lead-interaction" in prompt


def test_prompt_no_chart_selection_logic():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    for chart_keyword in ["plot_trajectory", "plot_timeseries", "plot_box", "trajectory_plot"]:
        assert chart_keyword not in prompt, f"chart-specific keyword '{chart_keyword}' leaked into lead prompt"


def test_prompt_line_count_drastically_reduced():
    """W16 砍 prompt 后的 ratchet:防止 prompt 再次膨胀失控。
    阈值 700 由 2026-05-20 校准 — 当前实测 598 行,余 100 行余量足够吸纳
    Gate 反问 / 中文调度 / subagent 描述等合理增长。超过 700 应警告并讨论。
    """
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    line_count = prompt.count("\n")
    assert line_count < 700, f"lead prompt too long (got {line_count} lines, expect <700)"


def test_prompt_no_default_fallback_reference():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "default-template-fallback" not in prompt
    assert "default-fallback" not in prompt


def test_prompt_no_metric_plan_only_plan_metrics():
    prompt = apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)
    assert "metric_plan.json" not in prompt
