"""Spec A 路由回归测试: 初次数据判读归 data-analyst, knowledge-assistant 场景A 收窄。

防回归 (2026-06-08 EPM n=1 dogfood thread 7d4d9b8e):
  用户首次发"帮我进行数据洞察" → lead 判 QA_FACT → 派 knowledge-assistant 从零产出完整判读
  (knowledge_response.md, 7548 字节, 14 处输出宪法禁止词)。
  根因: knowledge-assistant 场景 A 允许"基于已有分析结果的追问"但被滥用于从零判读;
  lead 路由没有区分"初次判读"vs"对已有结论追问"。

防御目标:
  - knowledge-assistant 场景 A 必须声明前提(已有 data-analyst handoff) + 边界(完整判读由 data-analyst 负责)
  - lead prompt 必须包含 workspace 状态驱动的路由: 无 handoff→data-analyst / 有 handoff→knowledge-assistant
  - n=1 fast-path 必须澄清"自动跳过仅指自动流水线,用户主动要洞察仍派 data-analyst"
"""
from __future__ import annotations

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.subagents.builtins.knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG


def _prompt() -> str:
    return apply_prompt_template(subagent_enabled=True, max_concurrent_subagents=3)


# ── A1: knowledge-assistant 场景 A 收窄 ──

class TestKnowledgeAssistantSceneANarrowed:
    """knowledge-assistant 场景 A 必须声明前提 + 边界,防从零判读越界。"""

    def test_scene_a_declares_prerequisite_handoff(self):
        """场景 A 必须声明前提: workspace 已有 data-analyst 判读结论。"""
        prompt = KNOWLEDGE_ASSISTANT_CONFIG.system_prompt
        assert ("已完成" in prompt or "已有" in prompt), \
            "场景 A 未声明'已完成/已有'判读前提; lead 可能误判为可从零判读"
        assert "handoff_data_analyst.json" in prompt, \
            "场景 A 未明确提 handoff_data_analyst.json 文件名; lead 无法用 workspace 状态判断"

    def test_scene_a_routes_full_insight_to_data_analyst(self):
        """场景 A 必须在越界时明确: 完整判读由 data-analyst 负责。"""
        prompt = KNOWLEDGE_ASSISTANT_CONFIG.system_prompt
        assert "data-analyst" in prompt, \
            "场景 A 未提及 data-analyst; knowledge-assistant 不知道完整判读该派谁"
        assert "完整判读建议由 data-analyst 完成" in prompt or \
            "完整的数据判读由 data-analyst 负责" in prompt, \
            "场景 A 缺少越界时的明确指引; knowledge-assistant 可能仍从零出完整判读"

    def test_scene_a_explain_not_regenerate(self):
        """场景 A 必须强调【解释】已有结论,而非【重新生成】。"""
        prompt = KNOWLEDGE_ASSISTANT_CONFIG.system_prompt
        assert "解释" in prompt, \
            "场景 A 未强调'解释'职责; 可能被理解为'生成判读'"
        assert "重新生成" in prompt or "而非" in prompt, \
            "场景 A 未区分'解释已有结论'vs'重新生成完整判读'"


# ── A2: lead 路由 —— 初次数据判读 → data-analyst ──

class TestLeadRoutesInitialInsightToDataAnalyst:
    """lead prompt 必须包含 workspace 状态驱动的判读路由。"""

    def test_prompt_has_workspace_state_routing(self):
        """路由必须用 handoff_data_analyst.json 存在与否作为区分信号。"""
        prompt = _prompt()
        assert "handoff_data_analyst" in prompt, \
            "lead prompt 缺少 handoff_data_analyst 文件名; 无法用 workspace 状态路由"

    def test_prompt_routes_initial_insight_to_data_analyst(self):
        """无 handoff + 用户要判读/洞察 → data-analyst(初次判读)。"""
        prompt = _prompt()
        assert "初次判读" in prompt, \
            "lead prompt 缺少'初次判读'概念; 无法区分首次 vs 追问"
        # 关键路由信号: data-analyst 在路由规则里被提及
        assert "data-analyst" in prompt, \
            "lead prompt 路由规则未提及 data-analyst"

    def test_prompt_n1_fastpath_clarifies_proactive_insight(self):
        """n=1 fast-path 必须澄清: 自动跳过仅指自动流水线,用户主动要洞察仍派 data-analyst。"""
        prompt = _prompt()
        assert "fast-path" in prompt.lower() or "快速路径" in prompt, \
            "lead prompt 缺少 n=1 fast-path 说明"
        # 关键澄清: fast-path 跳过 data-analyst 只是自动流水线行为,不是永久的
        assert ("自动流水线" in prompt or "自动跳过" in prompt), \
            "n=1 fast-path 未澄清'仅适用自动流水线'; lead 可能永久不派 data-analyst"

    def test_prompt_skip_planning_distinguishes_workspace_state(self):
        """跳过规划场景必须按 workspace 状态区分三种情况。"""
        prompt = _prompt()
        # 三种情况的区分信号都应存在
        assert "QA_KNOWLEDGE" in prompt, \
            "跳过规划场景缺少 QA_KNOWLEDGE(纯通用知识)路由"
        assert "场景 A" in prompt or "场景A" in prompt, \
            "跳过规划场景缺少场景 A(已有判读追问)路由"
        assert "初次判读" in prompt, \
            "跳过规划场景缺少'初次判读→data-analyst'路由"


# ── A3: knowledge-assistant when_to_use 收窄 ──

class TestKnowledgeAssistantWhenToUse:
    """when_to_use 必须反映收窄后的职责边界。"""

    def test_when_to_use_excludes_full_initial_insight(self):
        """when_to_use 不适合列表必须含'完整初次数据判读→派 data-analyst'。"""
        when = KNOWLEDGE_ASSISTANT_CONFIG.when_to_use
        assert "data-analyst" in when, \
            "when_to_use 未提及 data-analyst; lead 读描述时无法知道完整判读派谁"
        assert "初次" in when or "尚无" in when, \
            "when_to_use 未区分'已有结论追问'vs'初次判读'"

    def test_when_to_use_qa_fact_requires_existing_analysis(self):
        """QA_FACT 适合条件必须明确'已有 data-analyst 判读结论'。"""
        when = KNOWLEDGE_ASSISTANT_CONFIG.when_to_use
        assert "已有" in when, \
            "QA_FACT 适合条件缺少'已有'前提; lead 可能认为任何数据问题都可派"
        assert "QA_FACT" in when, \
            "when_to_use 缺少 QA_FACT 意图标注"
