"""契约测试：chart 全画 vs 代表性子集由 lead 反问决定，chart-maker 不擅自限流。

spec: docs/superpowers/specs/2026-06-22-chart-budget-ask-user-not-auto-throttle-spec.md

职责错位回归：历史 SKILL.md 第 26 行 + fallback-decision-tree.md 指示 chart-maker
「用户没明确要所有个体图时自行用 6-8 预算只画代表性子集」——subagent 替用户做了
「画多少」的决策。本测试守住「chart_budget 由 lead 在派遣 prompt 中给定，chart-maker
只执行」的新契约，防止自主限流文字被 sync 或编辑回潮。
"""
from __future__ import annotations

from pathlib import Path

import pytest

# skill 目录在 agent 包内（git 中）。测试文件位于
# packages/agent/backend/tests/，上跳到 packages/agent/（parents[2]），再进 skills/custom。
SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "custom"
    / "ethoinsight-chart-maker"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
FALLBACK_TREE = SKILL_DIR / "references" / "fallback-decision-tree.md"


@pytest.fixture(scope="module")
def skill_md_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fallback_tree_text() -> str:
    return FALLBACK_TREE.read_text(encoding="utf-8")


def test_skill_files_exist():
    """守 SSOT：契约测试依赖这两个文件存在。"""
    assert SKILL_MD.exists(), f"missing SKILL.md at {SKILL_MD}"
    assert FALLBACK_TREE.exists(), f"missing fallback-decision-tree.md at {FALLBACK_TREE}"


def _chart_budget_line(text: str) -> str:
    """提取 SKILL.md 里 chart_budget 那一行（含 P5 可选标记），契约断言聚焦在它。"""
    for line in text.splitlines():
        if "chart_budget" in line and ("P5" in line or "可选" in line):
            return line
    return ""


def test_skill_md_chart_budget_attribution_to_lead(skill_md_text: str):
    """chart_budget 行必须声明「由 lead 给定/决定」。

    spec §2.4：SKILL.md chart_budget 说明改为「由 lead 给定，不自行揣测」。
    这是「谁决定预算值」的职责边界——chart-maker 退化为纯执行者。
    断言聚焦 chart_budget 那一行（非 SKILL.md 全文，避免别处偶然出现的词造成假绿）。
    """
    line = _chart_budget_line(skill_md_text)
    assert line, "SKILL.md 必须有 chart_budget 说明行（P5/可选 标记）"
    assert "lead" in line, (
        f"chart_budget 行必须声明预算由 lead 给定，实际: {line!r}"
    )
    assert "给定" in line or "决定" in line, (
        f"chart_budget 行必须含「给定/决定」表明 lead 是预算来源，实际: {line!r}"
    )


def test_skill_md_no_autonomous_budget_throttling(skill_md_text: str):
    """chart_budget 行不得保留 chart-maker 自行揣测预算数字的语义。

    spec §一.2 / §2.4：历史 chart_budget 说明含「用户没明确要所有个体图时
    用预算（如 6-8）只画代表性子集」——chart-maker 自定一个 6-8 的预算数字。
    必须删除该「自行揣测数字」语义（预算数字现由 lead 给定）。聚焦到 chart_budget 行。
    """
    line = _chart_budget_line(skill_md_text)
    assert line, "SKILL.md 必须有 chart_budget 说明行"
    forbidden = [
        "如 6-8",  # 旧:chart-maker 自行揣测预算数字「如 6-8」
        "6-8",  # 同上（无「如」前缀也禁——数字预算自揣的标志）
    ]
    for phrase in forbidden:
        assert phrase not in line, (
            f"chart_budget 行不得保留 chart-maker 自揣预算数字语义「{phrase}」，实际: {line!r}"
        )


def test_skill_md_executes_lead_budget_or_full(skill_md_text: str):
    """chart_budget 行必须用正面指令说明执行逻辑：lead 给预算→用；未给/全画→省略（全画）。

    spec §六.2：deepseek 正面提示——chart-maker skill 写「按 lead 给定的预算执行」。
    """
    line = _chart_budget_line(skill_md_text)
    assert line, "SKILL.md 必须有 chart_budget 说明行"
    assert "省略" in line, (
        f"chart_budget 行必须说明 lead 未给预算/要全画时省略 chart_budget（全画），实际: {line!r}"
    )


def test_fallback_tree_no_autonomous_throttling(fallback_tree_text: str):
    """fallback-decision-tree.md 不得引导 chart-maker 自行设定预算数字。

    spec §三：fallback-decision-tree.md 去掉「用户没明确要所有个体图时自行用 6-8 预算」
    的自主限流指示。预算值要么由 lead 给定后透传，要么由 resolve 工具按 output_mode
    优先级确定性分配——chart-maker 不自定数字。
    """
    forbidden_phrases = [
        "用户没明确要所有个体图时",
        "自行用 6-8",
        "用预算（如 6-8）",
        "自行揣测",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in fallback_tree_text, (
            f"fallback-decision-tree.md 不得保留 chart-maker 自主限流指示「{phrase}」"
            "（spec §三：去掉自主限流指示）"
        )


def test_fallback_tree_budget_source_is_tool_param_not_self(fallback_tree_text: str):
    """fallback-decision-tree.md 的预算来源必须是 lead/工具传参（catalog.resolve 传预算
    或 prep_chart_plan），且明确「留空=全画」。

    spec §2.4 / §六.4：chart_budget 语义 SSOT 在 select_charts_by_priority + 工具，
    预算值由 lead 给定。决策树应保留「省略/留空 = 全画」的正面执行语义。
    """
    # 预算省略=全画 的正面执行语义必须保留（spec §2.4 末尾：派遣 prompt 未给预算→省略→全画）
    assert "省略" in fallback_tree_text and "全画" in fallback_tree_text, (
        "fallback-decision-tree.md 必须保留「预算留空/省略 = 全画」的正面执行语义"
    )
