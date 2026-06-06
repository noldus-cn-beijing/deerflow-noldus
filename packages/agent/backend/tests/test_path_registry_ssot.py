"""CI sentinel: 验证 path_registry 是真正的 SSOT — prompt 渲染、provider、INTENT 定义三处一致。

改 path_registry.PATHS → prompt 渲染 + 两个 provider 自动同步；
任何一处脱离 SSOT 手写 → CI 红。
"""

from __future__ import annotations

import re

from deerflow.agents.lead_agent.prompt import _render_intent_state_machine
from deerflow.guardrails.intent_classification_provider import _VALID_INTENTS
from deerflow.guardrails.path_registry import (
    ASK_GATE_MAP,
    PATHS,
    VALID_INTENTS,
    Step,
    ensure_dispatch_targets_validated,
    to_handoff_name,
)


class TestRenderMatchesLegacyArrowDiagram:
    """_render_intent_state_machine() 输出语义等价于原手写箭头图。

    验证 A1 纯重构：每条 INTENT 的路径链完整、步骤顺序正确、ask/dispatch 语义正确。
    不是逐字比对（因为格式变了：trigger 描述已移到「复合语义判定」段），
    而是验证 SSOT 渲染输出的语义内容。
    """

    def test_all_8_intents_rendered(self):
        rendered = _render_intent_state_machine()
        for intent in PATHS:
            assert intent in rendered, f"INTENT {intent} missing from rendered output"

    def test_e2e_full_askviz_chain(self):
        """E2E_FULL_ASKVIZ: code-executor → data-analyst → ask(viz?) → [viz==yes]chart-maker → ask(report?)"""
        rendered = _render_intent_state_machine()
        askviz_line = [l for l in rendered.split("\n") if l.startswith("E2E_FULL_ASKVIZ")][0]
        # All steps present in correct order
        assert "code-executor" in askviz_line
        assert "data-analyst" in askviz_line
        assert "ask(viz?)" in askviz_line
        assert "[viz==yes]chart-maker" in askviz_line
        assert "ask(report?)" in askviz_line
        # Verify order: code-executor before data-analyst before ask(viz?) before chart-maker
        assert askviz_line.index("code-executor") < askviz_line.index("data-analyst")
        assert askviz_line.index("data-analyst") < askviz_line.index("ask(viz?)")
        assert askviz_line.index("ask(viz?)") < askviz_line.index("chart-maker")

    def test_e2e_full_chain(self):
        """E2E_FULL: code-executor → data-analyst → chart-maker → ask(report?)"""
        rendered = _render_intent_state_machine()
        full_line = [l for l in rendered.split("\n") if l.startswith("E2E_FULL →")][0]
        assert "code-executor" in full_line
        assert "data-analyst" in full_line
        assert "chart-maker" in full_line
        assert "ask(report?)" in full_line

    def test_e2e_min_chain(self):
        """E2E_MIN: code-executor → ask(four_choice?)"""
        rendered = _render_intent_state_machine()
        min_line = [l for l in rendered.split("\n") if l.startswith("E2E_MIN")][0]
        assert "code-executor" in min_line
        assert "ask(four_choice?)" in min_line
        assert "chart-maker" not in min_line

    def test_simple_intents(self):
        """CHART/REPORT/QA_FACT/QA_KNOWLEDGE/CLARIFY: single-step paths."""
        rendered = _render_intent_state_machine()
        assert "CHART → chart-maker" in rendered
        assert "REPORT → report-writer" in rendered
        assert "QA_FACT → knowledge-assistant" in rendered
        assert "QA_KNOWLEDGE → knowledge-assistant" in rendered
        assert "CLARIFY → ask(clarify?)" in rendered

    def test_rendered_output_format(self):
        """Each rendered line is INTENT → chain with → separators."""
        rendered = _render_intent_state_machine()
        lines = [l.strip() for l in rendered.strip().split("\n") if l.strip()]
        assert len(lines) == len(PATHS), f"Expected {len(PATHS)} lines, got {len(lines)}"
        for line in lines:
            # Each line starts with an INTENT name followed by →
            assert re.match(r"^[A-Z0-9_]+ → ", line), f"Bad format: {line}"


class TestValidIntentsSingleSource:
    """intent_classification_provider 的 _VALID_INTENTS 必须来自 path_registry.VALID_INTENTS。"""

    def test_same_object(self):
        """_VALID_INTENTS is exactly the same frozenset object from path_registry."""
        assert _VALID_INTENTS is VALID_INTENTS, (
            "intent_classification_provider._VALID_INTENTS must be imported from "
            "path_registry.VALID_INTENTS, not independently defined"
        )

    def test_contains_all_8_intents(self):
        expected = {"E2E_FULL", "E2E_FULL_ASKVIZ", "E2E_MIN", "CHART", "REPORT", "QA_FACT", "QA_KNOWLEDGE", "CLARIFY"}
        assert set(VALID_INTENTS) == expected


class TestEveryDispatchTargetIsRegisteredSubagent:
    """每个 dispatch Step.target 必须是 BUILTIN_SUBAGENTS 的合法 key。"""

    def test_all_dispatch_targets_registered(self):
        """ensure_dispatch_targets_validated() raises ValueError on mismatch."""
        # This calls the lazy validation — if any target is not registered, it raises
        ensure_dispatch_targets_validated()

    def test_specific_targets(self):
        """Verify the specific dispatch targets we expect."""
        dispatch_targets = set()
        for steps in PATHS.values():
            for step in steps:
                if step.kind == "dispatch":
                    dispatch_targets.add(step.target)
        expected = {"code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"}
        assert dispatch_targets == expected


class TestEveryAskKeyHasGateMapping:
    """每个 ask Step.target（除 clarify）必须在 ASK_GATE_MAP 中有映射。"""

    def test_all_ask_keys_mapped(self):
        ask_keys = set()
        for steps in PATHS.values():
            for step in steps:
                if step.kind == "ask":
                    ask_keys.add(step.target)
        # "clarify" is special — no gate, uses ask_clarification directly
        keys_needing_gate = ask_keys - {"clarify"}
        for key in keys_needing_gate:
            assert key in ASK_GATE_MAP, f"ask key {key!r} missing from ASK_GATE_MAP"

    def test_gate_values_are_valid(self):
        """Gate values follow the gate_*_acknowledged pattern."""
        for key, gate in ASK_GATE_MAP.items():
            assert re.match(r"^gate\w+_acknowledged$", gate), (
                f"ASK_GATE_MAP[{key!r}] = {gate!r} doesn't match gate pattern"
            )


class TestPathSequenceProviderReadsPaths:
    """证明 PathSequenceProvider 的顺序判断来自 PATHS。

    改 PATHS 的一条 → provider 行为随之变。
    这个测试在 A2 阶段 PathSequenceProvider 实现后才真正有意义；
    A1 阶段先验证 PATHS 数据的 structure 足以驱动顺序判断。
    """

    def test_paths_defines_dispatch_order(self):
        """PATHS 中每个 INTENT 的 dispatch steps 是有序的。"""
        for intent, steps in PATHS.items():
            dispatch_steps = [s for s in steps if s.kind == "dispatch"]
            # Verify dispatch steps maintain insertion order (list is ordered)
            for i in range(len(dispatch_steps) - 1):
                current = dispatch_steps[i]
                next_step = dispatch_steps[i + 1]
                assert current.target != next_step.target, (
                    f"PATHS[{intent}] has consecutive dispatch to same target {current.target}"
                )

    def test_to_handoff_name_maps_correctly(self):
        """to_handoff_name converts prompt-side names to handoff-side names."""
        assert to_handoff_name("code-executor") == "code_executor"
        assert to_handoff_name("data-analyst") == "data_analyst"
        assert to_handoff_name("chart-maker") == "chart_maker"
        assert to_handoff_name("knowledge-assistant") == "knowledge_assistant"
        assert to_handoff_name("report-writer") == "report_writer"

    def test_e2e_full_predecessor_handoffs(self):
        """E2E_FULL 中 data-analyst 之前必须有 code-executor 完成。"""
        steps = PATHS["E2E_FULL"]
        code_idx = next(i for i, s in enumerate(steps) if s.kind == "dispatch" and s.target == "code-executor")
        analyst_idx = next(i for i, s in enumerate(steps) if s.kind == "dispatch" and s.target == "data-analyst")
        assert code_idx < analyst_idx, "code-executor must come before data-analyst in E2E_FULL"
