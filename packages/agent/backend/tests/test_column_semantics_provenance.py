"""TDD tests for HITL over-confirmation fix (spec 2026-06-30).

Core invariant: a `confirmed=true` written into experiment-context.json
column_semantics must be PROVEN to come from the CURRENT user turn. Memory
historical prefs may only be prefilled as `confirmed=false` (pending), never
silently stamped confirmed — the agent must not answer on the user's behalf
for items the user did NOT answer this turn.

Covers spec 四:
  1. 结构门单测：用户只回模板、memory 有列语义偏好 → 落盘 confirmed=false +
     confirmed_source=prefilled_from_memory（不是 confirmed=true）。
  2. 下游门联动测：column_semantics 有 confirmed=false 列时，派 code-executor 的
     guard deny 并触发 ask_clarification（断言 deny + 重问指令）。
  3. 正向路径不回归：用户本轮明确回答列语义 → confirmed=true +
     confirmed_source=user_current_turn，分析放行。
  4. 防 vacuous：去掉「prefilled_from_memory 强制降级」那行 → 测 1 应变红。
"""

import json
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gate1_runtime(tmp_path):
    """A MagicMock runtime whose thread_data points at tmp_path workspace."""
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": str(tmp_path)}}
    return runtime


def _gate1_seed(tmp_path, runtime, *, ev19_template="PlusMaze-AllZones"):
    """Run a Gate 1 paradigm call so experiment-context.json exists."""
    from deerflow.agents.middlewares.experiment_context import (
        set_experiment_paradigm_tool,
    )

    set_experiment_paradigm_tool.func(
        paradigm="elevated_plus_maze",
        paradigm_cn="高架十字迷宫",
        category="anxiety",
        subject="rodent",
        ev19_template=ev19_template,
        workspace_dir=str(tmp_path),
        runtime=runtime,
    )


# ---------------------------------------------------------------------------
# Test 1 — memory-prefilled column semantics must NOT be confirmed
# ---------------------------------------------------------------------------


class TestMemoryPrefilledSemanticsDowngraded:
    """Spec 四.1: agent only answered the template; column semantics came from
    memory historical prefs → the persisted entries must be confirmed=false and
    carry confirmed_source=prefilled_from_memory (NOT confirmed=true)."""

    def test_prefilled_from_memory_is_downgraded_to_unconfirmed(self, tmp_path):
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        # The LLM stamped confirmed=true on memory-derived prefs (the bug).
        # It declares the honest source via column_semantics_source.
        cs = {
            "columns": {
                "open": {
                    "raw_name": "open",
                    "resolves_to": "open_arms",
                    "meaning_zh": "开臂",
                    "confirmed": True,  # LLM wants to confirm — gate must downgrade
                },
                "closed": {
                    "raw_name": "closed",
                    "resolves_to": "closed_arms",
                    "meaning_zh": "闭臂",
                    "confirmed": True,
                },
            },
        }
        result_raw = set_experiment_paradigm_tool.func(
            column_semantics=cs,
            column_semantics_source="prefilled_from_memory",
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        assert json.loads(result_raw)["status"] == "ok"

        ctx = read_context(str(tmp_path))
        columns = ctx["column_semantics"]["columns"]
        for col in ("open", "closed"):
            assert columns[col]["confirmed"] is False, (
                f"memory-prefilled column {col!r} must be downgraded to confirmed=false, "
                f"got {columns[col]['confirmed']!r}"
            )
            assert columns[col]["confirmed_source"] == "prefilled_from_memory"

    def test_prefilled_columns_produce_no_aliases(self, tmp_path):
        """Downgraded (unconfirmed) memory prefs must NOT seed column_aliases —
        they are pending, so resolve must not silently consume them either."""
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        cs = {
            "columns": {
                "open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True},
            },
        }
        set_experiment_paradigm_tool.func(
            column_semantics=cs,
            column_semantics_source="prefilled_from_memory",
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        ctx = read_context(str(tmp_path))
        # _derive_column_aliases skips unconfirmed → no aliases written.
        assert ctx.get("column_aliases", {}) == {}


# ---------------------------------------------------------------------------
# Test 2 — downstream gate denies code-executor + instructs ask_clarification
# ---------------------------------------------------------------------------


class TestDownstreamGateDeniesOnUnconfirmed:
    """Spec 四.2: with a confirmed=false column still on disk, dispatching
    code-executor is denied and the reason instructs ask_clarification."""

    def test_guard_denies_and_instructs_ask_clarification(self, tmp_path):
        from deerflow.agents.middlewares.experiment_context import (
            set_experiment_paradigm_tool,
        )
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        # Prefill from memory → downgraded to confirmed=false on disk.
        cs = {
            "columns": {
                "open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True},
            },
        }
        set_experiment_paradigm_tool.func(
            column_semantics=cs,
            column_semantics_source="prefilled_from_memory",
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "run metrics"},
        )
        decision = provider.evaluate(request)
        assert decision.allow is False
        msg = " ".join(r.message for r in decision.reasons)
        assert "ask_clarification" in msg or "对齐" in msg, (
            "deny reason must instruct the agent to re-ask the user, "
            f"got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — positive path: user answered this turn → confirmed + dispatch allowed
# ---------------------------------------------------------------------------


class TestUserCurrentTurnConfirmedAndAllowed:
    """Spec 四.3: user explicitly answered column semantics this turn →
    confirmed=true + confirmed_source=user_current_turn, dispatch proceeds."""

    def test_user_current_turn_confirmed_and_dispatch_allowed(self, tmp_path):
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        cs = {
            "columns": {
                "open": {"raw_name": "open", "resolves_to": "open_arms", "meaning_zh": "开臂", "confirmed": True},
            },
        }
        result_raw = set_experiment_paradigm_tool.func(
            column_semantics=cs,
            column_semantics_source="user_current_turn",
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        assert json.loads(result_raw)["status"] == "ok"

        ctx = read_context(str(tmp_path))
        entry = ctx["column_semantics"]["columns"]["open"]
        assert entry["confirmed"] is True
        assert entry["confirmed_source"] == "user_current_turn"

        # Dispatch is allowed (gate sees confirmed=true).
        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "run metrics"},
        )
        assert provider.evaluate(request).allow is True


# ---------------------------------------------------------------------------
# Test 4 — anti-vacuous: the downgrade line is load-bearing
# ---------------------------------------------------------------------------


class TestAntiVacuousDowngrade:
    """Spec 四.4: if the deterministic downgrade is removed, Test 1's assertion
    must go red. We simulate the neuter by calling the downgrade helper directly
    and asserting it flips confirmed true→false. A vacuous guard (one that never
    fires) would leave confirmed=true and this test fails."""

    def test_downgrade_helper_flips_confirmed_true_to_false(self):
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_column_semantics_provenance,
        )

        cs = {
            "columns": {
                "open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True},
                "closed": {"raw_name": "closed", "resolves_to": "closed_arms", "confirmed": True},
            },
        }
        _stamp_column_semantics_provenance(cs, source="prefilled_from_memory")
        for col in ("open", "closed"):
            assert cs["columns"][col]["confirmed"] is False
            assert cs["columns"][col]["confirmed_source"] == "prefilled_from_memory"

    def test_downgrade_helper_preserves_user_current_turn(self):
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_column_semantics_provenance,
        )

        cs = {"columns": {"open": {"resolves_to": "open_arms", "confirmed": True}}}
        _stamp_column_semantics_provenance(cs, source="user_current_turn")
        assert cs["columns"]["open"]["confirmed"] is True
        assert cs["columns"]["open"]["confirmed_source"] == "user_current_turn"


# ---------------------------------------------------------------------------
# Resolved-facts provenance (spec 三.3) — memory-sourced resolved facts are
# stamped but NOT counted as this-turn-confirmed.
# ---------------------------------------------------------------------------


class TestResolvedFactsProvenance:
    """Spec 三.3: resolved_facts write-through records a source marker so a
    downstream consumer can tell memory-prefilled groups apart from
    this-turn-confirmed groups. Memory source does not flip a confirmed flag."""

    def test_apply_resolved_facts_stamps_source(self):
        from deerflow.agents.middlewares.experiment_context import (
            _apply_resolved_facts,
        )

        data: dict = {"paradigm": "epm"}
        _apply_resolved_facts(
            data,
            [{"key": "groups", "value": "XX=对照组, XY=低剂量"}],
            source="prefilled_from_memory",
        )
        resolved = data["resolved"]
        # groups value present
        assert resolved["groups"] == "XX=对照组, XY=低剂量"
        # provenance recorded alongside
        sources = resolved.get("sources", {})
        assert sources.get("groups") == "prefilled_from_memory"

    def test_apply_resolved_facts_user_current_turn(self):
        from deerflow.agents.middlewares.experiment_context import (
            _apply_resolved_facts,
        )

        data: dict = {"paradigm": "epm"}
        _apply_resolved_facts(
            data,
            [{"key": "groups", "value": "A=control, B=treatment"}],
            source="user_current_turn",
        )
        assert data["resolved"]["sources"]["groups"] == "user_current_turn"
