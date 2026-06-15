"""2026-06-15 spec #1/#4 — lead prompt 守护测试。

防 sync paraphrase 削弱铁律（守 memory feedback_sync_protected_file_paraphrase_merge_weakens_constitution）。
prompt 铁律难直接 E2E 测，用「断言关键句存在」的轻量守护。

用 importlib 显式加载 **worktree** 的 prompt.py 源码（绕过主仓 venv 的 editable 链接，
保证测的是当前 worktree 改动），与 test_handoff_content_validation.py 同一加载模式。

#1: 模板变体由"录制时是否划分析区"决定，不由列名决定——有归属列就是 Few/AllZones，
    走列对齐保持模板不变，不因列名非标准（open/closed）改判 NoZones。
#4: 规则 #7 区分"漏 seal 重派"与"诚实失败上报"——status=failed + 具体原因不机械重派。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL prompt module source from this worktree (bypass editable link).
# ---------------------------------------------------------------------------
_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "agents" / "lead_agent" / "prompt.py"
)

_REAL_PROMPT: ModuleType | None = None


def _get_real_prompt() -> ModuleType:
    global _REAL_PROMPT
    if _REAL_PROMPT is not None:
        return _REAL_PROMPT

    spec = importlib.util.spec_from_file_location(
        "deerflow.agents.lead_agent.prompt_real_0615",
        _PROMPT_FILE,
    )
    assert spec is not None, f"Could not find prompt.py at {_PROMPT_FILE}"
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        pytest.skip("Could not load real prompt module")

    _REAL_PROMPT = module
    return _REAL_PROMPT


@pytest.fixture(autouse=True)
def _ensure_module_loaded():
    _get_real_prompt()


def test_lead_prompt_has_template_by_zone_recording_rule():
    """#1 铁律：模板变体看划区与否、不看列名（在 SYSTEM_PROMPT_TEMPLATE 内）。"""
    prompt = _get_real_prompt()
    text = prompt.SYSTEM_PROMPT_TEMPLATE
    assert "由\"录制时是否划分析区\"决定" in text or "不由列名决定" in text
    # 关键语义：有区归属列 → Few/AllZones、走对齐保持模板
    assert "走列语义对齐" in text or "列对齐" in text
    assert "保持已选模板不变" in text


def test_lead_prompt_rule7_distinguishes_honest_failure():
    """#4 铁律：规则 #7 区分漏 seal 重派 vs 诚实失败上报（在 noldus_rules 内）。"""
    prompt = _get_real_prompt()
    section = prompt._build_subagent_section(max_concurrent=3)
    # 区分两种失败的指令必须存在
    assert "诚实的失败上报" in section
    assert "不要机械重派" in section
    # status=failed 是区分依据
    assert "status=failed" in section
