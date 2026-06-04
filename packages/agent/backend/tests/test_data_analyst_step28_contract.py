"""Batch 1 contract tests — data-analyst step 2.8 empty-{} branching + step 3 positive seal framing.

These are prompt text contract tests: they lock the exact phrasing in the system prompt
so that future syncs or edits cannot silently revert the two changes from 2026-06-03 batch 1.

Test categories (per handoff §4):
  1. Empty params_{} early branching — step 2.8 must have split + direct-to-seal semantics
  2. No plan_metrics.json parameters_in_use as audit source — truth source locked to metrics_summary
  3. Layer 4 positive framing — step 3 completion = tool_call, no negative reverse-activation phrases
  4. Existing tests still green (verified by running the full suite)
"""

from __future__ import annotations

from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt() -> str:
    """Return the full data-analyst system prompt."""
    p = DATA_ANALYST_CONFIG.system_prompt
    assert p, "system_prompt must not be empty"
    return p


# ---------------------------------------------------------------------------
# 1. Empty parameters_used early branching contract
# ---------------------------------------------------------------------------

def test_step28_has_empty_params_early_branch():
    """step 2.8 must contain the all-empty → natural-complete → direct-seal branch."""
    p = _prompt()

    # The first step of 2.8 must declare "判断本轮是否有可审计的参数"
    assert "判断本轮是否有可审计的参数" in p

    # All-empty branch: explicitly state "所有 metric 的 parameters_used 都是空"
    assert "所有 metric 的 parameters_used 都是空" in p

    # Natural-complete conclusion
    assert "天然完成" in p

    # Empty findings array + direct-to-seal
    assert "parameter_audit_findings 为空数组" in p
    assert "随即进入 step 3 调 seal_data_analyst_handoff" in p


def test_step28_nonempty_branch_still_present():
    """The non-empty branch (original audit path) must still exist."""
    p = _prompt()

    # At-least-one-non-empty branch
    assert "至少有一个 metric 的 parameters_used 非空" in p

    # a-f sections still referenceable
    assert "比对方法见下方 a–f" in p


# ---------------------------------------------------------------------------
# 2. No plan_metrics.json parameters_in_use as audit source
# ---------------------------------------------------------------------------

def test_truth_source_is_metrics_summary_not_plan():
    """Prompt must lock metrics_summary[*].parameters_used as the sole truth source."""
    p = _prompt()

    # Truth source declaration
    assert "唯一真相源" in p
    assert "metrics_summary" in p
    assert "parameters_used" in p

    # plan_metrics.json parameters_in_use explicitly marked as NOT audit source
    assert "plan_metrics.json" in p
    assert "parameters_in_use" in p
    assert "计划要用的参数" in p
    assert "不能作为审计对象" in p

    # Positive anchor: "以 metrics_summary 的 parameters_used 为准"
    assert "以 metrics_summary 的 parameters_used 为准" in p


def test_a_f_sections_scoped_to_nonempty_only():
    """The a-f sections must have an explicit scope note that they only apply to non-empty params."""
    p = _prompt()

    assert "以下 a–f 仅适用于 parameters_used 非空的 metric" in p
    assert "已被第一步分流" in p


# ---------------------------------------------------------------------------
# 2b. Non-empty branch seal-must-fire pre-frame + criteria-missing shortcut
#     (2026-06-04 O-Maze fix): the empty-{} fast path from 4caa78b8 did not
#     cover the non-empty path. open_zones="in_zone" (discrete param, no
#     paradigm criteria) fell into the a-f marathon and the model burned its
#     budget in step-2.8 reasoning, treating the written-out finding as "done"
#     and never emitting the seal tool_call. These anchors lock the fix so a
#     future sync/edit cannot silently revert it.
# ---------------------------------------------------------------------------

def test_nonempty_branch_has_seal_must_fire_preframe():
    """Before a-f, the non-empty branch must pre-frame 'seal is mandatory, audit is best-effort'."""
    p = _prompt()

    # The pre-frame must appear (placed before the a-f scope note)
    assert "进入 a–f 之前先记住" in p
    # Bounded reasoning budget so the audit can't marathon
    assert "本段至多 2-3 轮思考" in p
    # Positive completion framing reused from the layer-4 fix
    assert "发出 seal_data_analyst_handoff 的 tool_call" in p
    assert "seal 是必达，审计是尽力" in p

    # The pre-frame must come BEFORE the a-f scope note (ordering matters:
    # the escape hatch has to be read before the marathon, not after).
    preframe_idx = p.index("进入 a–f 之前先记住")
    scope_note_idx = p.index("以下 a–f 仅适用于 parameters_used 非空的 metric")
    assert preframe_idx < scope_note_idx, "seal pre-frame must precede the a-f scope note"


def test_nonempty_branch_has_criteria_missing_shortcut():
    """Discrete/categorical param + no paradigm criteria → one info finding, skip the a-f weighing."""
    p = _prompt()

    # Shortcut header
    assert "捷径（命中即用，不必走完 a–f）" in p
    # Must name the discrete-param trigger that O-Maze hit
    assert "离散/类别参数" in p
    assert "open_zones" in p
    # Must route to a single info finding via category_mismatch + issue #63 suggestion
    assert "category_mismatch" in p
    assert "参见 issue #63" in p
    # Must explicitly tell the model to stop weighing Phase 2/Phase 1/mismatch_kind and seal
    assert "记完即进入 step 3 发 seal" in p



# ---------------------------------------------------------------------------
# 3. Layer 4 positive framing — step 3 must use "tool_call as seal" language
# ---------------------------------------------------------------------------

def test_step3_has_positive_tool_call_is_seal_framing():
    """Step 3 must frame completion as 'issuing the seal tool_call'."""
    p = _prompt()

    # Exact key phrase from the handoff B.2 text
    assert "本步骤的完成标志是" in p
    assert "发出一次 seal_data_analyst_handoff 的 tool_call" in p

    # Positive framing: "这一次工具调用本身，就是封存这个动作"
    assert "这一次工具调用本身" in p
    # The prompt uses ASCII double-quotes around 封存 in this sentence
    assert '就是"封存"这个动作' in p

    # Narrative vs reality distinction
    assert "是叙述，不会落库" in p
    assert "真正落库靠这一次 tool_call" in p


def test_no_negative_reverse_activation_phrases_remain():
    """The old negative phrasing '不能只在 thinking 里写封存' must be GONE from the entire prompt."""
    p = _prompt()

    # B.3 cleanup: the old negative half-sentence must not exist anywhere
    assert "不能只在 thinking 里写" not in p
    assert "不能只在 thinking" not in p

    # Also verify the new positive version is present (from B.3)
    assert "step 3 会通过发出 seal 工具调用来落库本次分析" in p


def test_step3_still_includes_field_checklist():
    """Step 3 must still list all required handoff fields (field completeness guard)."""
    p = _prompt()

    # Field list preserved
    assert "status/key_findings/outlier_findings/excluded_metrics/" in p
    assert "method_warnings/recommendations/errors/gate_signals/quality_warnings/parameter_audit_findings" in p

    # Empty-array fallback preserved
    assert "如果没有相应发现，用空数组" in p
    assert "不要省略字段" in p


# ---------------------------------------------------------------------------
# 4. Cross-section sanity: the old "step 2.8 新增" label is gone
# ---------------------------------------------------------------------------

def test_step28_no_longer_says_newly_added():
    """The step 2.8 header should use 'Sprint 3 —' not the old 'Sprint 3 新增 —' label.

    Note: 'Sprint 3 新增' still appears in gate_signals_contract comments which is fine —
    we only care about the step 2.8 header wording change (Change A, §A.3 of handoff).
    """
    p = _prompt()

    # step 2.8 header uses the new concise label
    assert "Sprint 3 — 只警告不调参" in p

    # The old combined label in step 2.8 context is gone:
    # the old header was 'Sprint 3 新增 — 只警告不调参，铁律。参数审计至多占 2-3 轮思考'
    # Check that the old unique combined phrase no longer exists
    old_step28_header_marker = "Sprint 3 新增 — 只警告不调参"
    assert old_step28_header_marker not in p
