"""data-analyst thinking 瘦身静态契约（spec 2026-06-18-data-analyst-thinking-overload §4.3）。

data-analyst 曾在 thinking 里逐条搬运 63 条 outlier + 手跑参数审计 a–f 决策树 → 撑爆
50K 撞 900s 超时 → turn 永远不结束 → SealGate 救不了。本批锁死 prompt 的两条瘦身不变量：
  ① step 2.8 参数审计整段已删——prompt 不含其任何标志串（防 sync/编辑悄悄回退）。
  ② step 2.6 判据 read 硬前置 + step 2.7b「不搬运/不重算/不重映射」措辞在场。

加载策略：importlib 直接加载 worktree 的 data_analyst.py（主仓 venv editable 链接指向主仓，
正常 import 会读到主仓旧 prompt；conftest 又 mock 了 executor 导致循环导入被遮蔽——见
memory feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_DATA_ANALYST_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "builtins" / "data_analyst.py"
)


def _load_config():
    spec = importlib.util.spec_from_file_location(
        "data_analyst_worktree_diet",
        _DATA_ANALYST_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.DATA_ANALYST_CONFIG


def _prompt() -> str:
    p = _load_config().system_prompt
    assert p, "system_prompt must not be empty"
    return p


# ---------------------------------------------------------------------------
# ① step 2.8 参数审计已移除——任何标志串都不得回归
# ---------------------------------------------------------------------------

def test_prompt_has_no_step28_parameter_audit_markers():
    """step 2.8 参数适配性审计整段已删，其所有标志串必须从 prompt 消失。"""
    p = _prompt()
    # spec §4.3 列举的标志串（任一回归 = step 2.8 被回退）
    forbidden = [
        "参数适配性审计",
        "mismatch_kind",
        "Phase 2 优先路径",
        "比对方法见下方 a–f",
        "判断本轮是否有可审计的参数",
        "参数审计至多占",
        "Sprint 3 — 只警告不调参",
    ]
    for marker in forbidden:
        assert marker not in p, f"step 2.8 标志串回归到 prompt: {marker!r}"


def test_prompt_has_no_parameter_audit_decision_tree_markers():
    """参数审计决策树的 a–f 分支 / Phase 降级 / mismatch 五选一标志串必须消失。"""
    p = _prompt()
    forbidden = [
        "Phase 1 降级路径",
        "threshold_too_high",
        "threshold_too_low",
        "window_too_wide",
        "window_too_narrow",
        "category_mismatch",
        "降级字段填法",
        "参数审计【天然完成】",
        "天然完成",
    ]
    for marker in forbidden:
        assert marker not in p, f"参数审计决策树标志串回归: {marker!r}"


def test_gate_signals_contract_marks_audit_as_always_zero():
    """gate_signals 契约里两个 audit count 字段必须标注恒为 0（data-analyst 不再产出）。"""
    p = _prompt()
    assert "parameter_audit_findings_count: <int>" in p
    assert "parameter_audit_critical_count: <int>" in p
    # 至少其一的注释明示「恒为 0」
    assert "恒为 0" in p


def test_handoff_field_format_marks_audit_as_empty_array():
    """handoff 字段速查里 parameter_audit_findings 必须标注恒为空数组。"""
    p = _prompt()
    assert "parameter_audit_findings" in p
    assert "恒为空数组" in p


# ---------------------------------------------------------------------------
# ② step 2.6 判据 read 硬前置 + step 2.7b 不搬运措辞在场
# ---------------------------------------------------------------------------

def test_step26_paradigm_doc_is_hard_prerequisite():
    """step 2.6 必须把 by-experiment/<paradigm>.md read 标成「必读」硬前置。"""
    p = _prompt()
    # 路径硬编码在场
    assert "/mnt/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/<paradigm>.md" in p
    # 必读措辞
    assert "必读" in p
    # 判据来源措辞（混杂排查/解读方向/组间比较口径至少出现一处指向）
    assert "判读的判据来源" in p or "判据来源" in p


def test_step27b_no_carry_no_recompute_markers():
    """step 2.7b 必须含「不搬运/不重算/不重映射」三连措辞（thinking 只判读）。"""
    p = _prompt()
    # 引用而非搬运
    assert "不搬运" in p
    assert "不重算" in p
    assert "不重映射" in p
    # thinking 只做判断
    assert "thinking 只做判断" in p


def test_step27b_references_real_subject_identifier():
    """step 2.7b 必须明示 outlier 旁路成品的 subject 已是真名（如 Trial 3）。"""
    p = _prompt()
    # 统计层预填真名 + 引用成品
    assert "真实标识" in p
    assert "Trial 3" in p


def test_step27_intro_no_param_compare_phrase():
    """step 2.7 引导句不得再含「比对参数」（参数审计已删，旧措辞会诱导回归）。"""
    p = _prompt()
    # 旧 step 2.7 引导句含「比对参数」——删 step 2.8 后必须消失
    assert "比对参数" not in p
