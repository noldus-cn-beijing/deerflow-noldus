"""Guard: 复合语义判定规则在 lead prompt 中可被 lead 直接读到。

防回归 (2026-05-20 FST E2E 故障):
  用户说 "做描述性分析和可视化" → lead 错判 E2E_MIN → 派 chart-maker 后无下一步
  → lead 自己 read 5 个 workspace JSON 撞 LoopDetection hard_limit=5 → FORCED STOP。
  根因: skill intent-decision-tree.md 的 trigger 只列 "分析并画图" / "分析一下" 两个
  字面 pattern,没有动词类别概念,lead 漏判 "描述性分析+可视化"=复合语义。

防御目标:
  - lead prompt 主表(L257 一带)和简表(orchestration_guide 末尾)都必须出现复合语义判定
  - 4 个动词类别 CALC/ANALYZE/VISUALIZE/REPORT 名字必须在 prompt 中
  - 歧义偏 E2E_FULL 的兜底规则必须存在(否则下次 lead 仍可能保守判 E2E_MIN)
"""
from __future__ import annotations

from deerflow.agents.lead_agent.prompt import _build_subagent_section, apply_prompt_template


def _section() -> str:
    return _build_subagent_section(max_concurrent=3)


def _prompt() -> str:
    return apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)


class TestCompositeSemanticsRule:
    """主表 (intent state machine 紧随其后) 必须给出复合语义判定。"""

    def test_section_mentions_composite_semantics_judgment(self):
        section = _section()
        assert "复合语义" in section, "复合语义概念缺失,lead 可能保守判 E2E_MIN"

    def test_section_lists_four_verb_categories(self):
        section = _section()
        for category in ["CALC", "ANALYZE", "VISUALIZE", "REPORT"]:
            assert category in section, f"动词类别 '{category}' 缺失"

    def test_section_states_ge_two_categories_rule(self):
        """≥2 类 = 复合语义 是 E2E_FULL 与 E2E_MIN 唯一分水岭。"""
        section = _section()
        assert any(token in section for token in ["≥2", ">=2", "至少 2"]), \
            "≥2 个动词类别 = 复合语义 这条规则缺失"

    def test_section_has_ambiguity_fallback(self):
        """歧义偏 E2E_FULL 兜底:避免 lead 在 chart-maker 完成后撞硬限。"""
        section = _section()
        assert "歧义" in section or "偏向" in section, \
            "歧义兜底规则缺失;下次 lead 在边界 case 仍可能保守判 E2E_MIN"


class TestCompositeSemanticsExampleInSkill:
    """具体例子 (description analysis + visualization) 仍在 skill markdown 中。

    我们不在 prompt 里把每个例子都列出来 (会撑爆 prompt) ,而是要求 lead 看 skill 时
    能找到 '描述性分析和可视化' 这个具体 case 的归类示范。
    """

    def test_skill_decision_tree_has_compose_example(self):
        from pathlib import Path
        skill_path = Path("/home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md")
        if not skill_path.exists():
            import pytest
            pytest.skip(f"skill markdown not at {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        assert "描述性分析" in text and "可视化" in text, \
            "skill decision tree 缺少 '描述性分析和可视化' 这个具体 case 示范"
        assert "ANALYZE" in text and "VISUALIZE" in text, \
            "skill decision tree 缺少 4 个动词类别命名"


class TestPipelineSummaryMentionsRule:
    """orchestration_guide 末尾的简表也必须提一句复合语义。"""

    def test_orchestration_summary_mentions_composite(self):
        prompt = _prompt()
        assert "复合语义" in prompt, \
            "orchestration_guide 末尾简表缺少复合语义提示;lead 可能不读 skill 仍漏判"
