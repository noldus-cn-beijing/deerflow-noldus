"""契约测试：chart-maker subagent 的 system_prompt 默认全画，不自行设预算。

spec: docs/superpowers/specs/2026-06-22-chart-budget-ask-user-not-auto-throttle-spec.md
用户裁决（2026-06-22）：画图部分不该限制资源，有多少画多少。默认全画，lead 不主动
反问全画/子集；chart_budget 仅在用户主动要少画时由 lead 给定并透传。

chart-maker 的 SubagentConfig.system_prompt（chart_maker.py）是 chart-maker **实际执行**
的最高权威指令源（常驻 context，比 read_file 才读的 SKILL.md 权威更高）。本测试守住
system_prompt 不再含「自行给 6-8 预算」类自主限流语义，防止回潮。
"""
from __future__ import annotations

from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG


def _system_prompt() -> str:
    return CHART_MAKER_CONFIG.system_prompt


def test_chart_maker_system_prompt_defaults_to_full_plot():
    """system_prompt 必须明确「默认省略 chart_budget = 全画」。

    用户裁决：画图不限资源，有多少画多少。system_prompt 是 chart-maker 最高权威指令源，
    必须把「默认全画」写在最显眼处（prep_chart_plan 调用说明的 chart_budget 子项）。
    """
    p = _system_prompt()
    assert "默认省略" in p or "默认不传" in p, (
        "chart_maker system_prompt 必须说明 chart_budget 默认省略 = 全画"
    )
    assert "全画" in p, "chart_maker system_prompt 必须含「全画」语义"


def test_chart_maker_system_prompt_no_autonomous_default_budget():
    """system_prompt 不得含 chart-maker 自行给默认预算数字（如 6-8）的指示。

    spec §一.2：历史 chart-maker 擅自塞 chart_budget=6-8 砍图。这是职责错位的诱因，
    在最高权威指令源里更必须清除。预算数字只能由 lead 在派遣 prompt 给定后透传。
    """
    p = _system_prompt()
    forbidden = [
        "给一个绘图预算总数（如 6-8）",  # 旧:自主给默认预算
        "通常 ≤ 8",  # 旧:假设预算恒给且 ≤ 8
        "给一个绘图预算总数",
    ]
    for phrase in forbidden:
        assert phrase not in p, (
            f"chart_maker system_prompt 不得保留 chart-maker 自主设预算语义「{phrase}」"
            "（预算数字仅由 lead 给定后透传）"
        )


def test_chart_maker_system_prompt_budget_from_lead_verbatim():
    """system_prompt 必须说明 chart_budget 的值来自 lead 派遣 prompt（执行者照搬）。

    spec §2.4 / §六.4：chart-maker 退化为纯执行者。预算值由 lead 决定，
    chart-maker 只照搬，绝不自行揣测。
    """
    p = _system_prompt()
    assert "lead" in p, (
        "chart_maker system_prompt 必须指明 chart_budget 的值由 lead 给定（执行者照搬）"
    )
    assert "照搬" in p or "透传" in p, (
        "chart_maker system_prompt 必须含「照搬/透传」表明 chart-maker 只执行 lead 给定的预算"
    )


def test_chart_maker_system_prompt_full_plot_not_exception():
    """system_prompt 不得把「全画」框定成例外（旧:用户明确要全画时才省略）。

    旧 system_prompt 写「用户明确要"所有个体图"时省略 chart_budget（全画）」——把全画当例外。
    新契约：全画是默认，子集才是例外（用户主动要少画时）。
    """
    p = _system_prompt()
    # 旧「例外框定」的标志：把省略 chart_budget 当成「用户明确要 X 时」才做的事。
    forbidden = [
        '用户明确要"所有个体图"时省略',
        "用户明确要“所有个体图”时省略",
    ]
    for phrase in forbidden:
        assert phrase not in p, (
            f"chart_maker system_prompt 不得把全画框定为例外「{phrase}」"
            "（全画是默认，子集才是用户主动要时的例外）"
        )
