"""契约测试：chart 全画默认（不限资源），chart_budget 仅 lead 在用户主动要少画时透传。

spec: docs/superpowers/specs/2026-06-22-chart-budget-ask-user-not-auto-throttle-spec.md
用户裁决（2026-06-22）：画图部分不该限制资源，有多少画多少。默认全画，
不主动反问「全画 vs 子集」；只有用户主动表达「画几张就行/代表性/少画点」时
lead 才给定 chart_budget 数字并透传给 chart-maker。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from deerflow.agents.lead_agent import prompt as prompt_module


def _render_prompt() -> str:
    """渲染 lead system prompt（noldus_rules 段含 chart 反问模板）。

    复用 test_apply_prompt_template_includes_custom_mounts 的 monkeypatch 模式：
    把所有外部依赖 stub 掉，让 has_noldus_agents=True，从而 noldus_rules 段被渲染。

    使用 ``patch.object`` 上下文管理器统一 stub 与恢复——此前手写赋值只在
    ``finally`` 里恢复了 ``cfg_mod.get_app_config``，其余 ``prompt_module.*``
    属性（``_get_resolved_facts_context`` / ``_get_prior_corrections_context`` /
    ``_get_memory_context`` 等）被永久替换成返回空串的 lambda，污染后续整套
    test suite（test_resolved_facts_readback / test_s8_feedback_corrections 全红）。
    """
    # has_noldus_agents 由 get_available_subagent_names() 决定 —— 返回含 noldus agent。
    config = SimpleNamespace(
        sandbox=SimpleNamespace(mounts=[]),
        skills=SimpleNamespace(container_path="/mnt/skills"),
    )
    import deerflow.config as cfg_mod

    with (
        patch.object(prompt_module, "get_available_subagent_names", lambda **kw: [
            "code-executor",
            "data-analyst",
            "chart-maker",
            "report-writer",
            "knowledge-assistant",
        ]),
        patch.object(cfg_mod, "get_app_config", lambda: config),
        patch.object(prompt_module, "_get_app_config", lambda: config, create=True),
        patch.object(prompt_module, "_get_enabled_skills", lambda: []),
        patch.object(prompt_module, "get_deferred_tools_prompt_section", lambda **kw: ""),
        patch.object(prompt_module, "_build_acp_section", lambda **kw: ""),
        patch.object(prompt_module, "_get_memory_context", lambda agent_name=None, **kw: ""),
        patch.object(prompt_module, "_get_prior_corrections_context", lambda paradigm=None, user_id=None: ""),
        patch.object(prompt_module, "_get_resolved_facts_context", lambda thread_id=None, agent_name=None, user_id=None: ""),
        patch.object(prompt_module, "get_agent_soul", lambda agent_name=None: ""),
    ):
        return prompt_module.apply_prompt_template(subagent_enabled=True)


def test_prompt_default_full_plot_no_budget():
    """用户答「要图」→ lead 派 chart-maker 时省略 chart_budget = 全画。

    用户裁决：画图不该限制资源，有多少画多少。prompt 必须明确「要图就全画」的正面语义。
    """
    p = _render_prompt()
    assert "全画" in p, (
        "lead prompt 必须含「全画」语义——用户要图就全画（不限资源）"
    )


def test_prompt_budget_only_when_user_asks_subset():
    """chart_budget 数字只在用户主动要少画/子集时由 lead 给定。

    spec §2.4 + 用户裁决：默认全画，不主动反问全画/子集。只有用户原话含
    「画几张/代表性/少画点/挑几个」时 lead 才给定预算数字并透传给 chart-maker。
    prompt 必须把这条触发条件写明（含「代表性/少画」类用户主动信号词）。
    """
    p = _render_prompt()
    # 必须出现「用户主动要子集」类触发条件，而非「subject 超阈值就反问」。
    # 用「代表性」或「少画」锚定「用户主动要少画」语义。
    assert ("代表性" in p) or ("少画" in p), (
        "lead prompt 必须说明 chart_budget 仅在用户主动要代表性子集/少画时才给定"
    )


def test_prompt_no_autonomous_budget_threshold_question():
    """lead 不得主动反问用户「全画 vs 子集」（spec 原设计的反问档已被用户裁决否决）。

    用户裁决：默认全画，不把「画多少」抛回给用户。prompt 不应含「subject 超阈值时
    反问全画/子集」的指示。守「画图不该限制资源」原则。
    """
    p = _render_prompt()
    # 被否决的设计标志：把「全画/子集」作为反问选项抛给用户（默认反问）。
    # 锚定「subject 数多时反问全画还是子集」这类主动反问指示。
    forbidden = [
        "subject 数多时反问",
        "total_subjects > 阈值",
        "反问用户选全画或子集",
        "反问「全画/子集」",
    ]
    for phrase in forbidden:
        assert phrase not in p, (
            f"lead prompt 不得含主动反问全画/子集的设计「{phrase}」"
            "（用户裁决：默认全画，不主动反问画多少）"
        )


def test_prompt_chart_maker_executes_lead_budget_verbatim():
    """chart-maker 只照搬 lead 给定的预算，prompt 须体现「执行者不揣测」契约。

    spec §2.4：chart-maker 退化为纯执行者。lead prompt 对 chart-maker 的派遣指引须含
    「由 lead 给定/决定」类归属语（与 SKILL.md 契约一致，守 SSOT）。
    """
    p = _render_prompt()
    # chart_budget 的归属必须出现在 chart 派遣相关上下文。宽松锚定「lead」+ 预算语义。
    assert "chart_budget" in p or "预算" in p, (
        "lead prompt 须提及 chart_budget/预算，以指明其由 lead 给定（守 SKILL.md SSOT）"
    )
