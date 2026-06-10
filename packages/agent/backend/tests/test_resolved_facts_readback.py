"""Tests for Spec B: resolved facts readback (write-through + prompt injection).

Covers:
  - C1: scope isolation (only this thread's user_clarification facts are rendered)
  - C2: independent block with reuse rules
  - C3: last-writer-wins on same-key conflicts
  - §4: write-through dual landing (experiment-context.json authoritative + facts[] projection)
  - B1 injection into system prompt via apply_prompt_template
  - B4 non-regression: loop-detection / todo-middleware unchanged
"""

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THREAD_ID = "test-thread-abc"
THREAD_MARKER = f"[thread:{THREAD_ID}]"


def _make_fact(
    content: str,
    *,
    source: str = "user_clarification",
    category: str = "user_clarification",
    confidence: float = 1.0,
    created_at: str = "2026-06-10T10:00:00Z",
) -> dict:
    return {
        "id": "f-" + content[:8].replace(" ", "_"),
        "content": content,
        "category": category,
        "confidence": confidence,
        "source": source,
        "createdAt": created_at,
    }


def _make_memory(facts: list[dict] | None = None) -> dict:
    return {
        "version": "1.0",
        "lastUpdated": "",
        "user": {
            "workContext": {"summary": "", "updatedAt": ""},
            "personalContext": {"summary": "", "updatedAt": ""},
            "topOfMind": {"summary": "", "updatedAt": ""},
        },
        "history": {
            "recentMonths": {"summary": "", "updatedAt": ""},
            "earlierContext": {"summary": "", "updatedAt": ""},
            "longTermBackground": {"summary": "", "updatedAt": ""},
        },
        "facts": facts or [],
    }


# ---------------------------------------------------------------------------
# B1 / C1: scope isolation
# ---------------------------------------------------------------------------

def test_resolved_facts_context_renders_scoped_block():
    """C1: Only this thread's user_clarification facts are rendered."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    memory_data = _make_memory(
        [
            _make_fact(f"{THREAD_MARKER} groups: Trial1=control, Trial2=treatment"),
            _make_fact(f"{THREAD_MARKER} paradigm: OFT"),
            _make_fact("[thread:other-thread] groups: X=control, Y=treatment"),
            _make_fact("User likes Python", source="knowledge"),  # wrong source
        ]
    )

    # get_memory_data is lazily imported inside _get_resolved_facts_context via
    #   from deerflow.agents.memory import get_memory_data
    # so we patch the source module.
    with patch("deerflow.agents.memory.get_memory_data", return_value=memory_data):
        result = _get_resolved_facts_context(thread_id=THREAD_ID)

    # Only 2 facts (scoped + correct source)
    assert "groups: Trial1=control, Trial2=treatment" in result
    assert "paradigm: OFT" in result
    assert "other-thread" not in result
    assert "User likes Python" not in result
    assert result.count("<resolved_task_facts>") == 1
    assert result.count("</resolved_task_facts>") == 1


def test_resolved_facts_context_empty_when_no_thread_id():
    """Returns empty string when thread_id is None."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    result = _get_resolved_facts_context(thread_id=None)
    assert result == ""


def test_resolved_facts_context_empty_when_no_matching_facts():
    """Returns empty string when no user_clarification facts exist."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    memory_data = _make_memory(
        [_make_fact("User likes Python", source="knowledge")]
    )

    with patch("deerflow.agents.memory.get_memory_data", return_value=memory_data):
        result = _get_resolved_facts_context(thread_id=THREAD_ID)

    assert result == ""


# ---------------------------------------------------------------------------
# C3: last-writer-wins
# ---------------------------------------------------------------------------

def test_resolved_facts_last_writer_wins_on_conflict():
    """C3: Same key written twice → only latest (by createdAt) rendered."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    memory_data = _make_memory(
        [
            _make_fact(f"{THREAD_MARKER} groups: Trial1=control, Trial2=treatment",
                       created_at="2026-06-10T10:00:00Z"),
            _make_fact(f"{THREAD_MARKER} groups: X=control, Y=drug",
                       created_at="2026-06-10T11:00:00Z"),  # newer → wins
        ]
    )

    with patch("deerflow.agents.memory.get_memory_data", return_value=memory_data):
        result = _get_resolved_facts_context(thread_id=THREAD_ID)

    assert "X=control, Y=drug" in result
    assert "Trial1=control, Trial2=treatment" not in result
    assert "既定事实" in result  # reuse rule present


# ---------------------------------------------------------------------------
# C2: reuse rule in rendered block
# ---------------------------------------------------------------------------

def test_resolved_facts_block_carries_reuse_rule():
    """C2: Rendered block contains the consumption rules."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    memory_data = _make_memory(
        [_make_fact(f"{THREAD_MARKER} groups: A=control, B=treatment")]
    )

    with patch("deerflow.agents.memory.get_memory_data", return_value=memory_data):
        result = _get_resolved_facts_context(thread_id=THREAD_ID)

    assert "既定事实" in result
    assert "无需重读输入文件重新验证" in result
    assert "以此处为准" in result
    assert "不再 ask_clarification 重问" in result


# ---------------------------------------------------------------------------
# C2: separate block from memory
# ---------------------------------------------------------------------------

def test_resolved_facts_block_separate_from_memory():
    """B1: <resolved_task_facts> is independent from <memory> block."""
    from deerflow.agents.lead_agent.prompt import _get_resolved_facts_context

    memory_data = _make_memory(
        [_make_fact(f"{THREAD_MARKER} groups: A=control, B=treatment")]
    )

    with patch("deerflow.agents.memory.get_memory_data", return_value=memory_data):
        result = _get_resolved_facts_context(thread_id=THREAD_ID)

    # It should NOT contain <memory> tags
    assert "<memory>" not in result
    # It SHOULD have its own <resolved_task_facts> tags
    assert "<resolved_task_facts>" in result
    assert "</resolved_task_facts>" in result


# ---------------------------------------------------------------------------
# §4: dual landing
# ---------------------------------------------------------------------------

def test_clarification_answer_write_through_dual_landing():
    """§4: resolved_facts written to experiment-context.json (authoritative)
    AND projected to memory facts. groups.json is NOT touched."""
    from deerflow.agents.middlewares.experiment_context import (
        _apply_resolved_facts,
        _write_user_clarification_fact_to_memory,
    )

    # --- Authoritative: experiment-context.json ---
    data: dict = {"paradigm": "oft", "ev19_template": "OpenFieldRectangle"}
    resolved_facts = [
        {"key": "groups", "value": "Trial1=control, Trial2=treatment"},
        {"key": "分组", "value": "试验1=实验组, 试验2=对照组"},
    ]
    _apply_resolved_facts(data, resolved_facts)

    assert "resolved" in data
    assert data["resolved"]["groups"] == "Trial1=control, Trial2=treatment"
    assert data["resolved"]["分组"] == "试验1=实验组, 试验2=对照组"
    # Original fields preserved
    assert data["paradigm"] == "oft"

    # --- Projection: memory facts ---
    with (
        patch("deerflow.agents.memory.get_memory_data", return_value={"facts": []}),
        patch("deerflow.agents.memory.storage.get_memory_storage") as mock_storage,
    ):
        success = _write_user_clarification_fact_to_memory(
            key="groups",
            value="Trial1=control, Trial2=treatment",
            thread_id=THREAD_ID,
        )
        assert success is True
        assert mock_storage.return_value.save.called


# ---------------------------------------------------------------------------
# §4 / runtime: thread_id extraction must read the FLAT runtime.context shape
# (regression guard — a nested configurable.thread_id read returns None in
# production and silently disables the memory projection)
# ---------------------------------------------------------------------------

def test_thread_id_from_runtime_reads_flat_context_key():
    """thread_id is read from the FLAT runtime.context['thread_id'], matching
    memory_middleware / thread_data_middleware / loop_detection_middleware etc.

    Feeds a REALISTIC runtime.context shape (the gap that let the original
    bug ship: tests fed a controlled thread_id but never exercised the real
    extraction path)."""
    from deerflow.agents.middlewares.experiment_context import _thread_id_from_runtime

    # (a) flat context dict → extracts correctly
    runtime_flat = MagicMock()
    runtime_flat.context = {"thread_id": THREAD_ID, "user_id": "u1"}
    assert _thread_id_from_runtime(runtime_flat) == THREAD_ID

    # (b) WRONG nested shape (RunnableConfig 'configurable') must NOT be the
    #     path we read — ToolRuntime.context is flat. If someone regresses the
    #     helper back to ctx.get("configurable",{}).get("thread_id"), this
    #     realistic flat-context input would return None and this asserts fail.
    runtime_nested_only = MagicMock()
    runtime_nested_only.context = {"configurable": {"thread_id": THREAD_ID}}
    assert _thread_id_from_runtime(runtime_nested_only) is None, (
        "thread_id must come from the flat context key, not nested configurable"
    )

    # (c) robustness: None runtime, None/non-dict context
    assert _thread_id_from_runtime(None) is None
    runtime_no_ctx = MagicMock()
    runtime_no_ctx.context = None
    assert _thread_id_from_runtime(runtime_no_ctx) is None
    runtime_missing = MagicMock()
    runtime_missing.context = {"user_id": "u1"}  # no thread_id key
    assert _thread_id_from_runtime(runtime_missing) is None

def test_apply_prompt_template_injects_resolved_facts():
    """B1: System prompt contains <resolved_task_facts> when facts exist,
    and does NOT when there are none."""
    from deerflow.agents.lead_agent.prompt import apply_prompt_template

    memory_with_facts = _make_memory(
        [_make_fact(f"{THREAD_MARKER} groups: A=control, B=treatment")]
    )
    memory_empty = _make_memory([])

    # Case 1: facts exist → block injected
    # _get_memory_context and _get_prior_corrections_context are module-level
    # functions in prompt.py, so we patch them there.
    # get_memory_data is lazily imported from deerflow.agents.memory.
    with (
        patch("deerflow.agents.lead_agent.prompt._get_memory_context", return_value=""),
        patch("deerflow.agents.lead_agent.prompt._get_prior_corrections_context", return_value=""),
        patch("deerflow.agents.memory.get_memory_data", return_value=memory_with_facts),
    ):
        prompt = apply_prompt_template(thread_id=THREAD_ID)
        assert "<resolved_task_facts>" in prompt
        assert "groups: A=control, B=treatment" in prompt
        assert "既定事实" in prompt

    # Case 2: empty facts → no block content rendered
    with (
        patch("deerflow.agents.lead_agent.prompt._get_memory_context", return_value=""),
        patch("deerflow.agents.lead_agent.prompt._get_prior_corrections_context", return_value=""),
        patch("deerflow.agents.memory.get_memory_data", return_value=memory_empty),
    ):
        prompt = apply_prompt_template(thread_id=THREAD_ID)
        # The B3 prompt rule references <resolved_task_facts> literally as
        # an instruction to the lead, so the tag name always appears.
        # The block CONTENT (facts list + reuse rules) should NOT appear.
        assert "既定事实，按既定事实处理" not in prompt
        assert "（规则：这些是本任务的既定事实" not in prompt

    # Case 3: no thread_id → no block content rendered
    with (
        patch("deerflow.agents.lead_agent.prompt._get_memory_context", return_value=""),
        patch("deerflow.agents.lead_agent.prompt._get_prior_corrections_context", return_value=""),
    ):
        prompt = apply_prompt_template(thread_id=None)
        assert "既定事实，按既定事实处理" not in prompt
        assert "（规则：这些是本任务的既定事实" not in prompt


# ---------------------------------------------------------------------------
# B4: non-regression — loop-detection unchanged
# ---------------------------------------------------------------------------

def test_loop_detection_warn_threshold_unchanged():
    """B4: loop-detection warn_threshold is still 3."""
    from deerflow.config.loop_detection_config import LoopDetectionConfig

    config = LoopDetectionConfig()
    assert config.warn_threshold == 3, (
        "loop-detection warn_threshold must remain 3 (B4: don't weaken the safety net)"
    )


def test_todo_middleware_awaiting_clarification_unchanged():
    """B4: TodoMiddleware _is_awaiting_clarification still exists."""
    from deerflow.agents.middlewares import todo_middleware

    assert hasattr(todo_middleware, "_is_awaiting_clarification"), (
        "B4: _is_awaiting_clarification must still exist (don't touch 6-04 fix)"
    )


# ---------------------------------------------------------------------------
# Experiment-context.json resolved write-through
# ---------------------------------------------------------------------------

def test_set_experiment_paradigm_with_resolved_facts(tmp_path):
    """B2: resolved_facts merges into context data correctly."""
    from deerflow.agents.middlewares.experiment_context import _apply_resolved_facts

    context_path = tmp_path / "experiment-context.json"
    base_data = {
        "paradigm": "oft",
        "paradigm_cn": "旷场",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "OpenFieldRectangle",
        "gate_completed": ["gate1_paradigm"],
        "analysis_config_id": "test123",
    }
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(base_data, ensure_ascii=False))

    resolved_facts = [
        {"key": "groups", "value": "Trial1=control, Trial2=treatment"},
    ]
    data = json.loads(context_path.read_text())
    _apply_resolved_facts(data, resolved_facts)
    context_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    result = json.loads(context_path.read_text())
    assert result["resolved"]["groups"] == "Trial1=control, Trial2=treatment"
    assert result["paradigm"] == "oft"  # existing field preserved


def test_apply_resolved_facts_merges_multiple_calls():
    """resolved_facts merges across multiple calls (append semantics)."""
    from deerflow.agents.middlewares.experiment_context import _apply_resolved_facts

    data: dict = {"paradigm": "epm"}

    # First call
    _apply_resolved_facts(data, [{"key": "groups", "value": "A=control, B=treatment"}])
    assert data["resolved"]["groups"] == "A=control, B=treatment"

    # Second call adds new key, updates existing
    _apply_resolved_facts(data, [
        {"key": "groups", "value": "X=control, Y=drug"},  # update
        {"key": "分组中文", "value": "第一组=对照, 第二组=实验"},  # new key
    ])
    assert data["resolved"]["groups"] == "X=control, Y=drug"  # updated
    assert data["resolved"]["分组中文"] == "第一组=对照, 第二组=实验"  # new


# ---------------------------------------------------------------------------
# _write_user_clarification_fact_to_memory
# ---------------------------------------------------------------------------

def test_write_user_clarification_fact_structure():
    """Fact written to memory has correct shape: source, category, thread scoping."""
    from deerflow.agents.middlewares.experiment_context import _write_user_clarification_fact_to_memory

    with (
        patch("deerflow.agents.memory.get_memory_data", return_value={"facts": []}),
        patch("deerflow.agents.memory.storage.get_memory_storage") as mock_storage,
    ):
        success = _write_user_clarification_fact_to_memory(
            key="groups",
            value="A=control, B=treatment",
            thread_id=THREAD_ID,
        )
        assert success is True

        call_args = mock_storage.return_value.save.call_args
        saved_memory = call_args[0][0]
        facts = saved_memory.get("facts", [])
        assert len(facts) == 1
        fact = facts[0]
        assert fact["source"] == "user_clarification"
        assert fact["category"] == "user_clarification"
        assert fact["confidence"] == 1.0
        assert THREAD_MARKER in fact["content"]
        assert "groups: A=control, B=treatment" in fact["content"]


# ---------------------------------------------------------------------------
# _thread_id_from_runtime
# ---------------------------------------------------------------------------

def test_thread_id_from_runtime():
    """Extracts thread_id from the FLAT ToolRuntime.context['thread_id'].

    NOTE: ToolRuntime.context is a flat dict — thread_id is at the top level,
    NOT nested under 'configurable' (that nesting belongs to RunnableConfig).
    Reading the nested path returns None in production and silently disables
    the memory projection. This test guards the correct flat-key behavior;
    see also test_thread_id_from_runtime_reads_flat_context_key above.
    """
    from deerflow.agents.middlewares.experiment_context import _thread_id_from_runtime

    assert _thread_id_from_runtime(None) is None

    mock_runtime = MagicMock()
    mock_runtime.context = {"thread_id": "my-thread-123"}
    assert _thread_id_from_runtime(mock_runtime) == "my-thread-123"

    # Nested configurable shape is NOT the runtime.context path → None.
    mock_runtime.context = {"configurable": {"thread_id": "my-thread-123"}}
    assert _thread_id_from_runtime(mock_runtime) is None

    # No thread_id key at all → None.
    mock_runtime.context = {"user_id": "u1"}
    assert _thread_id_from_runtime(mock_runtime) is None
