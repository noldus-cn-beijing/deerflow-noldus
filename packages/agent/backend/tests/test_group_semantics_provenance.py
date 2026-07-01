"""TDD tests for L4-1 group-semantics provenance gate (spec 2026-07-01).

Core invariant: a group's *interpretive* label (e.g. "低剂量", "高剂量") may
only survive into ``resolved.group_semantics`` if the user confirmed it THIS
turn (``source="user_current_turn"``). Anything the agent inferred
(``agent_inferred``) or pulled from memory historical prefs
(``prefilled_from_memory``) is deterministically **downgraded to a neutral name**
(``实验组 N``, numbered stably by group_structure order; a user-confirmed
对照组 retains its label) and marked ``confirmed=false``.

This is a SOFT gate: it degrades at write time. It does NOT block analysis —
group_structure (the subject→group mapping) is written verbatim so between-group
comparison still runs. The gate's job is to make it physically impossible for a
downstream report to write a dose-response narrative out of an unconfirmed
semantics label.

Covers spec 四 (1–8):
  1. agent_inferred "低剂量" → 中性名 "实验组N" + confirmed=false + 记 source.
  2. prefilled_from_memory → same downgrade (memory prefill ≠ user confirmation).
  3. user_current_turn "对照组" → label retained + confirmed=true (no false hurt).
  4. source=None (legacy) → untouched (back-compat).
  5. Neutral-name numbering is stable by group_structure order → same in, same out.
  6. Anti-vacuous: neuter the downgrade line in the pure fn → tests 1/2 go red.
  7. End-to-end structural invariant (assert structure, NOT substring): walk every
     item of resolved.group_semantics after a full write; each non-user-confirmed
     item's label MUST match the neutral-name pattern (or original group label),
     never the agent-fabricated semantics text.
  8. group_structure written verbatim + analysis path not denied (soft gate).
"""

import json
import re
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NEUTRAL_NAME_RE = re.compile(r"^实验组\d+$")


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
# Tests 1–5 — unit 2 pure function
# ---------------------------------------------------------------------------


class TestStampGroupSemanticsProvenance:
    """Spec 四.1–四.5: the pure provenance-stamping / downgrade function."""

    def test_agent_inferred_downgraded_to_neutral_and_unconfirmed(self):
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"XX": ["s1", "s2"], "XY": ["s3", "s4"]}
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "低剂量", "source": "agent_inferred"},
        }
        out = _stamp_group_semantics_provenance(semantics, structure, source=None)
        # user-confirmed control retained
        assert out["XX"]["label_text"] == "对照组"
        assert out["XX"]["confirmed"] is True
        # agent_inferred downgraded to neutral name, unconfirmed, source recorded
        assert out["XY"]["label_text"] != "低剂量"
        assert NEUTRAL_NAME_RE.match(out["XY"]["label_text"]), out["XY"]["label_text"]
        assert out["XY"]["confirmed"] is False
        assert out["XY"]["confirmed_source"] == "agent_inferred"

    def test_prefilled_from_memory_also_downgraded(self):
        """memory prefill is NOT a this-turn user confirmation → downgrade."""
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"A": ["s1"], "B": ["s2"]}
        semantics = {
            "A": {"label_text": "对照组", "source": "user_current_turn"},
            "B": {"label_text": "高剂量", "source": "prefilled_from_memory"},
        }
        out = _stamp_group_semantics_provenance(semantics, structure, source=None)
        assert out["B"]["label_text"] != "高剂量"
        assert NEUTRAL_NAME_RE.match(out["B"]["label_text"]), out["B"]["label_text"]
        assert out["B"]["confirmed"] is False
        assert out["B"]["confirmed_source"] == "prefilled_from_memory"

    def test_user_current_turn_label_retained_and_confirmed(self):
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"XX": ["s1"], "XY": ["s2"]}
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "应激组", "source": "user_current_turn"},
        }
        out = _stamp_group_semantics_provenance(semantics, structure, source=None)
        for key, expected in (("XX", "对照组"), ("XY", "应激组")):
            assert out[key]["label_text"] == expected
            assert out[key]["confirmed"] is True
            assert out[key]["confirmed_source"] == "user_current_turn"

    def test_legacy_per_item_source_none_left_untouched(self):
        """An item with no per-item source (= legacy / undeclared) is not
        downgraded — back-compat with callers that never declared provenance."""
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"A": ["s1"], "B": ["s2"]}
        semantics = {
            "A": {"label_text": "对照组"},
            "B": {"label_text": "实验组X"},
        }
        out = _stamp_group_semantics_provenance(semantics, structure, source=None)
        # No source declared → left as-is, confirmed not forced false.
        assert out["A"]["label_text"] == "对照组"
        assert out["B"]["label_text"] == "实验组X"

    def test_neutral_name_numbering_stable_by_structure_order(self):
        """Same input → same output; numbering follows group_structure order."""
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"XX": ["s1"], "XY": ["s2"], "YY": ["s3"], "YZ": ["s4"]}
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "低剂量", "source": "agent_inferred"},
            "YY": {"label_text": "中剂量", "source": "agent_inferred"},
            "YZ": {"label_text": "高剂量", "source": "agent_inferred"},
        }
        out1 = _stamp_group_semantics_provenance(semantics, structure, source=None)
        out2 = _stamp_group_semantics_provenance(
            {k: dict(v) for k, v in semantics.items()}, structure, source=None
        )
        assert out1 == out2  # stable
        # Neutral names numbered by structure order among non-confirmed items.
        assert out1["XY"]["label_text"] == "实验组1"
        assert out1["YY"]["label_text"] == "实验组2"
        assert out1["YZ"]["label_text"] == "实验组3"

    def test_legacy_call_level_source_downgrades_entries_without_per_item_source(self):
        """Spec 错误处理 / 边界: legacy caller fallback — entries with NO
        per-item source inherit the call-level ``source`` arg; a memory call-level
        source downgrades them just like an explicit per-item source would.
        (Back-compat with callers that only declare provenance at the call level.)"""
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"A": ["s1"], "B": ["s2"]}
        semantics = {
            "A": {"label_text": "对照"},
            "B": {"label_text": "高剂量"},
        }
        out = _stamp_group_semantics_provenance(
            semantics, structure, source="prefilled_from_memory"
        )
        # Both entries inherit the call-level memory source → downgraded.
        assert out["A"]["label_text"] == "实验组1"
        assert out["B"]["label_text"] == "实验组2"
        for k in ("A", "B"):
            assert out[k]["confirmed"] is False
            assert out[k]["confirmed_source"] == "prefilled_from_memory"


# ---------------------------------------------------------------------------
# Tests 7–8 — end-to-end structural invariant + soft gate (no analysis block)
# ---------------------------------------------------------------------------


class TestEndToEndOverclaimInvariant:
    """Spec 四.7 + 四.8: after a full set_experiment_paradigm write carrying
    group_structure + group_semantics, walk resolved.group_semantics and assert
    the STRUCTURAL invariant: every non-user-confirmed item carries a neutral
    name (or the original group label), never the fabricated semantics text.
    Asserts structure, not a substring."""

    def test_unconfirmed_semantics_neutralized_end_to_end(self, tmp_path):
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        # The dogfood bug: user only said XX=对照; agent fabricated XY/YY/YZ doses.
        structure = {
            "XX": ["s1", "s2"],
            "XY": ["s3", "s4"],
            "YY": ["s5"],
            "YZ": ["s6", "s7"],
        }
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "低剂量", "source": "agent_inferred"},
            "YY": {"label_text": "中剂量", "source": "agent_inferred"},
            "YZ": {"label_text": "高剂量", "source": "agent_inferred"},
        }
        raw = set_experiment_paradigm_tool.func(
            group_structure=structure,
            group_semantics=semantics,
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        assert json.loads(raw)["status"] == "ok"

        ctx = read_context(str(tmp_path))
        resolved = ctx["resolved"]
        gs = resolved["group_semantics"]

        fabricated = {"低剂量", "中剂量", "高剂量"}
        for key, entry in gs.items():
            label = entry["label_text"]
            source = entry.get("confirmed_source")
            if source == "user_current_turn":
                # user-confirmed items may keep their semantic label
                assert label == semantics[key]["label_text"], (key, label)
                continue
            # STRUCTURAL invariant: every non-user-confirmed item's label MUST
            # NOT be a fabricated semantics string; it must be a neutral name
            # (实验组N) or the original group label. Asserting the structure
            # (not `"剂量" not in str(...)`) — a substring check would pass on a
            # label the agent rephrased to "高浓度", a false green.
            assert label not in fabricated, (
                f"item {key!r} label {label!r} leaked an unconfirmed semantic"
            )
            is_neutral = bool(NEUTRAL_NAME_RE.match(label))
            is_original = label == key
            assert is_neutral or is_original, (
                f"item {key!r} label {label!r} must be a neutral name or the "
                f"original group label, not an unconfirmed semantic"
            )

    def test_group_structure_written_verbatim_soft_gate(self, tmp_path):
        """Spec 四.8: group_structure is written as-is; the soft gate does NOT
        block the analysis path (no raise / no error status)."""
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )

        runtime = _gate1_runtime(tmp_path)
        _gate1_seed(tmp_path, runtime)

        structure = {"XX": ["s1", "s2"], "XY": ["s3", "s4"]}
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "低剂量", "source": "agent_inferred"},
        }
        raw = set_experiment_paradigm_tool.func(
            group_structure=structure,
            group_semantics=semantics,
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        assert json.loads(raw)["status"] == "ok"  # soft gate: no block

        ctx = read_context(str(tmp_path))
        # group_structure preserved verbatim — between-group comparison unaffected.
        assert ctx["resolved"]["group_structure"] == structure


# ---------------------------------------------------------------------------
# Test 6 — anti-vacuous probe (assertion-only; the actual neuter is run by the
# implementer against the real source and observed red, per spec 四.6).
# ---------------------------------------------------------------------------


class TestAntiVacuousDowngradeLoadBearing:
    """Spec 四.6: the downgrade is the load-bearing behaviour. This test
    exercises the pure fn end-to-end so that neutering the downgrade inside it
    (commenting out the neutral-name assignment for agent_inferred/memory)
    makes this test go red. The implementer MUST additionally run the live
    neuter against the source and paste the red output before restoring it."""

    def test_downgrade_actually_replaces_fabricated_label(self):
        from deerflow.agents.middlewares.experiment_context import (
            _stamp_group_semantics_provenance,
        )

        structure = {"XX": ["s1"], "XY": ["s2"]}
        semantics = {
            "XX": {"label_text": "对照组", "source": "user_current_turn"},
            "XY": {"label_text": "低剂量", "source": "agent_inferred"},
        }
        out = _stamp_group_semantics_provenance(semantics, structure, source=None)
        # If the downgrade line were removed, label_text would stay "低剂量"
        # and this assertion would fail — proving the test is not vacuous.
        assert out["XY"]["label_text"] != "低剂量"
        assert NEUTRAL_NAME_RE.match(out["XY"]["label_text"])
        assert out["XY"]["confirmed"] is False
