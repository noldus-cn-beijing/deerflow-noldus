"""data-analyst step 3 positive seal framing 契约测试。

2026-06-18（spec data-analyst-thinking-overload）：step 2.8 参数适配性审计整段已从
data-analyst prompt 删除（判据行为学上造不出来，是 thinking 超时根因之一）。原锁定
step 2.8 措辞的 7 个测试随之移除——step 2.8 已不存在的反向契约见
``test_data_analyst_thinking_diet.py``。本文件只保留 **step 3 产出/交付合一** 的正面
framing 契约（与参数审计无关、step 3 在删 2.8 后仍成立）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# 加载 worktree 的 data_analyst.py（主仓 venv editable 指主仓，正常 import 读旧 prompt；
# 与 test_data_analyst_thinking_diet.py 同模式，见 memory
# feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib）。
_DATA_ANALYST_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "builtins" / "data_analyst.py"
)


def _load_config():
    spec = importlib.util.spec_from_file_location(
        "data_analyst_worktree_step28",
        _DATA_ANALYST_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.DATA_ANALYST_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt() -> str:
    """Return the full data-analyst system prompt."""
    p = _load_config().system_prompt
    assert p, "system_prompt must not be empty"
    return p


# ---------------------------------------------------------------------------
# step 3 positive framing — completion = issuing the seal tool_call (产出=交付合一)
# ---------------------------------------------------------------------------

def test_step3_has_positive_tool_call_is_seal_framing():
    """Step 3 must frame completion as 'issuing the seal tool_call' (产出=交付合一).

    Updated 2026-06-18: the separate '本步骤的完成标志是' / '这一次工具调用本身'
    phrasings were consolidated into the unified '产出与交付合一' framing (seal
    refactor — conclusion is first written directly into the seal tool args,
    no separate write-then-copy step). Locks the consolidated intent.
    """
    p = _prompt()

    # Step 3 header: production == issuing the seal tool_call
    assert "产出分析 = 发出 seal_data_analyst_handoff 的 tool_call" in p
    # 产出/交付合一 framing
    assert "产出与交付合一" in p

    # The "first written directly into tool args" positive framing
    assert "你的结论第一次成文" in p
    # This very tool call is both production and persistence
    assert "这次工具调用本身既是产出也是落库" in p
    # Hard rule: must go through the seal tool, never write_file
    assert "严禁直接 write_file 写 handoff_data_analyst.json" in p


def test_no_negative_reverse_activation_phrases_remain():
    """The old negative phrasing '不能只在 thinking 里写封存' must be GONE from the entire prompt.

    Updated 2026-06-18: the old positive companion sentence
    'step 3 会通过发出 seal 工具调用来落库本次分析' was replaced by the
    consolidated '这次工具调用本身既是产出也是落库' framing. The negative
    half-sentence must still be absent; the positive anchor is updated.
    """
    p = _prompt()

    # B.3 cleanup: the old negative half-sentence must not exist anywhere
    assert "不能只在 thinking 里写" not in p
    assert "不能只在 thinking" not in p

    # The current positive companion (consolidated 产出/交付合一 framing)
    assert "这次工具调用本身既是产出也是落库" in p


def test_step3_still_includes_field_checklist():
    """Step 3 must still list all required handoff fields (field completeness guard).

    Note (2026-06-18): step 2.8 removed, but step 3 still passes parameter_audit_findings
    as an empty array (schema field retained for forward compat). The field checklist
    therefore still names it — data-analyst must emit the field, just always [].
    """
    p = _prompt()

    # Field list preserved (parameter_audit_findings still in the list, emitted as [])
    assert "status/key_findings/outlier_findings/excluded_metrics/" in p
    assert "method_warnings/recommendations/errors/gate_signals/quality_warnings/parameter_audit_findings" in p

    # Empty-array fallback preserved
    assert "如果没有相应发现，用空数组" in p
    assert "不要省略字段" in p
