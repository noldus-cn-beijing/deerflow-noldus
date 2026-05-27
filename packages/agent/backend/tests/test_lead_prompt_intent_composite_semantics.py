"""Guard: 复合语义判定规则在 lead prompt 中可被 lead 直接读到。

防回归 (2026-05-20 FST E2E 故障):
  用户说 "做描述性分析和可视化" → lead 错判 E2E_MIN → 派 chart-maker 后无下一步
  → lead 自己 read 5 个 workspace JSON 撞 LoopDetection hard_limit=5 → FORCED STOP。
  根因: skill intent-decision-tree.md 的 trigger 只列 "分析并画图" / "分析一下" 两个
  字面 pattern,没有动词类别概念,lead 漏判 "描述性分析+可视化"=复合语义。

防回归 (2026-05-21 FST E2E thread f3fbce44 模糊语义 4 次摇摆):
  用户说 "帮我分析一下大鼠强迫游泳" → lead 在 thinking 反复重读规则,4 次摇摆才定到 E2E_FULL。
  根因: "分析" 单词在 4 类归类法里是单一 ANALYZE → E2E_MIN,但科研语境下"分析"通常等同
  "全流程",归类法和实际用法冲突。

防御目标(已更新到 E2E_FULL_ASKVIZ 体系):
  - lead prompt 主表必须出现 3 个新 intent: E2E_FULL_ASKVIZ / E2E_FULL / E2E_MIN
  - Fast-path 必须存在:模糊总称("分析"/"看看")→ E2E_FULL_ASKVIZ,明确出图词("画/图/可视化")→ E2E_FULL
  - 4 个动词类别 CALC/ANALYZE/VISUALIZE/REPORT 名字仍在 prompt 中(fallback 路径)
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

    def test_section_has_askviz_intent(self):
        """E2E_FULL_ASKVIZ 是模糊总称的归宿,必须在主表里。"""
        section = _section()
        assert "E2E_FULL_ASKVIZ" in section, \
            "E2E_FULL_ASKVIZ intent 缺失,模糊语义('分析一下')将无法走低置信路径"

    def test_section_has_fast_path_for_vague_terms(self):
        """模糊总称('分析/看看/研究下')必须有 fast-path 短路到 ASKVIZ,避免归类摇摆。"""
        section = _section()
        assert "分析" in section and "看看" in section, \
            "fast-path 缺少'分析'/'看看'模糊总称触发词"
        assert "ASKVIZ" in section, "fast-path 没有指向 E2E_FULL_ASKVIZ"

    def test_section_has_explicit_viz_triggers(self):
        """明确出图意愿触发词('画/图/可视化/箱线/轨迹/表')必须在 prompt 里列出。"""
        section = _section()
        assert any(token in section for token in ["画", "可视化"]), \
            "明确出图意愿触发词缺失;lead 无法区分明示 vs 模糊"
        assert any(token in section for token in ["箱线", "轨迹", "趋势", "表"]), \
            "具体图种名/表格触发词缺失;明示画图意愿的细分识别无依据"

    def test_section_has_ambiguity_fallback(self):
        """歧义兜底:避免 lead 在边界 case 摇摆。"""
        section = _section()
        assert "歧义" in section or "偏向" in section, \
            "歧义兜底规则缺失;下次 lead 在边界 case 仍可能反复摇摆"

    def test_section_requires_summary_before_askviz_clarification(self):
        """ASKVIZ 反问前 lead 必须先汇报 data-analyst 发现(防回归:thread 7456611e
        直接 ask 没汇报,用户只看到反问卡片看不到分析结果)。"""
        section = _section()
        # 关键短语之一必须出现,描述"先汇报 + 再 ask"两步流程
        assert any(
            phrase in section
            for phrase in ["先输出一段汇报", "搬运 data-analyst", "搬运 ... key_findings", "搬运 data-analyst 的 key_findings"]
        ), "ASKVIZ 流程缺少'反问前必须汇报 data-analyst 发现'的明示规则"


class TestCompositeSemanticsExampleInSkill:
    """具体例子仍在 skill markdown 中。"""

    def test_skill_decision_tree_has_compose_example(self):
        from pathlib import Path
        skill_path = Path("/home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight-lead-interaction/references/intent-decision-tree.md")
        if not skill_path.exists():
            import pytest
            pytest.skip(f"skill markdown not at {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        assert "可视化" in text, \
            "skill decision tree 缺少 '可视化' 触发词示范"
        assert "ANALYZE" in text and "VISUALIZE" in text, \
            "skill decision tree 缺少 4 个动词类别命名"
        assert "E2E_FULL_ASKVIZ" in text, \
            "skill decision tree 必须解释 E2E_FULL_ASKVIZ 的触发条件"
        assert "分析一下" in text, \
            "skill decision tree 必须给出模糊总称 case(用户原话)"


class TestPipelineSummaryMentionsRule:
    """orchestration_guide 末尾的简表也必须提一句复合语义。"""

    def test_orchestration_summary_mentions_composite(self):
        prompt = _prompt()
        assert "复合语义" in prompt, \
            "orchestration_guide 末尾简表缺少复合语义提示;lead 可能不读 skill 仍漏判"
        assert "ASKVIZ" in prompt or "E2E_FULL_ASKVIZ" in prompt, \
            "orchestration_guide 简表缺少 E2E_FULL_ASKVIZ 提示"
