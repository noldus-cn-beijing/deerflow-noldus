"""2026-06-29 dogfood 修复 — lead prompt suspect_columns 主动列对齐守护测试。

根因（thread 3dcac0a0 磁盘取证）：identify_ev19_template 第 0 步就输出
zone_info.suspect_columns=["open","closed"]，但 lead prompt 674-675 行只把
inspect_uploaded_file/列问题框定为「分组」问题，没指引 agent「suspect_columns
已在手 → 把列对齐折进首个合并反问 + 同一次 set_experiment_paradigm 带
column_semantics」。结果 agent 漏带 column_semantics → prep_metric_plan 报
columns_missing → 才回头重 inspect（一次失败 + 一次多余 inspect 的浪费来回）。

判定 ETHO-5（结构已对、只缺指引）→ 收紧 prompt。本测试守护该指引存在，纯
source-substring 断言（不依赖 LLM 输出），与 test_lead_prompt_template_zone_rule.py
同一 importlib 加载模式（绕过主仓 venv editable 链接，测 worktree 源）。
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
        "deerflow.agents.lead_agent.prompt_real_0629_suspect",
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


def test_suspect_columns_fold_into_first_clarification():
    """suspect_columns 已在手时,列对齐并入首个合并反问、不再额外 inspect。

    现在红:prompt 全文无 column_semantics。编辑后绿。
    """
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    assert "suspect_columns" in text
    assert "column_semantics" in text  # 载荷断言:全文今天 0 次出现 → 今天必红


def test_suspect_columns_guidance_is_colocated_with_inspect_rule():
    """防回归:指引须长在 paradigm_identification_system 段、与 inspect 规则同处,
    不是新加游离 reminder(守 HarnessX §6.6 反累加)。"""
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    section = (
        text.split("<paradigm_identification_system>")[1]
        .split("</paradigm_identification_system>")[0]
    )
    assert "suspect_columns" in section  # 今天 suspect_columns 仅在段外(589 行) → 今天必红
    assert "column_semantics" in section
    assert "inspect_uploaded_file" in section  # 确认是 tighten 674-675 而非另起
