"""#6b 文案守护测试（2026-06-16 EPM dogfood）：data-analyst prompt 的
"statistics 为空走描述性 partial + 立即 seal，不要手算" 规则存在性守护。

防 sync paraphrase 削弱（守 feedback_sync_protected_file_paraphrase_merge_weakens_constitution）：
data_analyst.py 是受保护文件，上游 sync 时若把这条规则改写没了，这里会 fail。

importlib 加载 worktree 源：worktree 共享主仓 venv，editable deerflow 指主仓，
直接 `from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG`
测主仓 prompt（worktree 改动不生效=假绿）。
守 feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL data_analyst.py source from this worktree
# ---------------------------------------------------------------------------
_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "builtins" / "data_analyst.py"
)

_MODULE: ModuleType | None = None


def _get_module() -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    assert _FILE.exists(), f"data_analyst.py not found at {_FILE}"
    spec = importlib.util.spec_from_file_location("data_analyst_worktree_6b", _FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not load worktree data_analyst.py: {e}")
    _MODULE = module
    return _MODULE


def _prompt() -> str:
    return _get_module().DATA_ANALYST_CONFIG.system_prompt


# ---------------------------------------------------------------------------
# 文案守护：关键句存在（防 paraphrase 削弱）
# ---------------------------------------------------------------------------


def test_prompt_has_empty_statistics_rule() -> None:
    """#6b：prompt 含"statistics 为空"规则（描述性 partial 路径触发条件）。"""
    text = _prompt()
    assert "statistics 为空" in text or "statistics 字段为空" in text, (
        "#6b 规则缺失：data-analyst prompt 应含 statistics 为空时的处理规则"
    )


def test_prompt_directs_immediate_finalize() -> None:
    """#6b：statistics 空时走 fill + finalize(partial) 封存（spec 2026-06-23-...-fill-template）。

    旧规则"立即调 seal_data_analyst_handoff"已移除（seal args 撞 max_tokens 狭颈），
    改为 fill key_findings → finalize(final_status="partial")。
    """
    text = _prompt()
    assert "finalize" in text and "partial" in text, (
        "#6b 规则缺失：应指示 statistics 空时 fill key_findings 后 finalize(partial)"
    )
    assert "seal_data_analyst_handoff" not in text, "旧一次性 seal 引用应已移除"


def test_prompt_forbids_manual_recomputation() -> None:
    """#6b：prompt 含"不要手算/手工重算"（堵手算补偿螺旋，#6b 主诉求）。"""
    text = _prompt()
    assert "不要" in text and ("手算" in text or "手工重算" in text), (
        "#6b 规则缺失：应明确禁止手工重算组间统计"
    )


def test_prompt_marks_partial_status() -> None:
    """#6b：prompt 指示 statistics 空走 status='partial'（三态对齐）。"""
    text = _prompt()
    # 规则块里应同时出现"描述性"与"partial"
    assert "描述性" in text, "#6b 规则缺失：应指示走描述性判读"
    assert 'status="partial"' in text or "status=partial" in text or '"partial"' in text, (
        "#6b 规则缺失：应指示 status=partial"
    )


def test_prompt_marks_descriptive_only_in_key_findings() -> None:
    """#6b：prompt 指示 key_findings 标注'仅描述性分析、未做推断检验'（让下游知情）。"""
    text = _prompt()
    assert "仅描述性分析" in text or "未做推断检验" in text, (
        "#6b 规则缺失：应指示在 key_findings 标注仅描述性"
    )
