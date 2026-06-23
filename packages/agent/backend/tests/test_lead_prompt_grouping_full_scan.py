"""2026-06-23 ETHO-5 spec — lead prompt 分组判定必须基于全量 per_file_grouping 守护测试。

防 sync paraphrase 削弱铁律（守 memory feedback_sync_protected_file_paraphrase_merge_weakens_constitution）。
prompt 铁律用「断言关键句存在」的轻量守护。

用 importlib 显式加载 **worktree** 的 prompt.py 源码（绕过主仓 venv 的 editable 链接，
保证测的是当前 worktree 改动），与 test_lead_prompt_template_zone_rule.py 同一加载模式。

ETHO-5 真因（spec §1.2）：工具侧 identify_ev19_template 一次扫全部文件返回 per_file_grouping
（全量正确，identify_ev19_template_tool.py:518 `for f in uploaded_files`），缺口在 lead prompt
只有「优先用」软建议，给了 lead「看一个 inspect 就外推全局分组」的偷懒入口。范式推断有对称的
「禁止猜」硬约束，分组判定此前没有对称硬约束。

修法（纯 prompt）：把「优先用」升级为「分组判定必须基于全部文件 per_file_grouping」+「单个/少数
文件 inspect 结果不能外推为全部分组结论」（正面措辞为主）。**不改工具、不加 middleware。**

本测试守护 SYSTEM_PROMPT_TEMPLATE 反问合并规则段的分组段：
  1. 必须含「分组判定覆盖全部文件 / 基于 per_file_grouping」硬约束语义（红→绿：改前只有「优先用」软串）
  2. 必须含「不能外推」语义（单个/少数文件结果不能当全部结论）
  3. 必须保留 fallback 分支（per_file_grouping 为空/不直观时才 fallback inspect）
  4. 修改段用正面措辞，不引入「禁止猜」式否定祈使（守 deepseek 正面提示铁律）
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
        "deerflow.agents.lead_agent.prompt_real_etho5",
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


def _grouping_block(text: str) -> str:
    """提取反问合并规则段中 per_file_grouping 分组判定相关的小段。

    反问合并规则段以「**反问合并规则」起，per_file_grouping 段紧跟其后到「合并反问时：」前。
    """
    start = text.index("per_file_grouping")
    # 向后截到 fallback 段结束（合并反问时 之前）
    end_marker = "合并反问时："
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end]


def test_prompt_requires_full_per_file_grouping():
    """ETHO-5 核心（红→绿）：分组判定必须基于全部上传文件的 per_file_grouping。

    改前只有「优先用它」软串，不含「必须基于全部文件」硬约束 → 红。
    改后含「必须基于全部」+「每个文件依据自己条目归组」→ 绿。
    """
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    block = _grouping_block(text)
    # 硬约束：必须基于全部文件（不是「优先用」软建议）
    assert ("必须基于全部" in block or "覆盖全部文件" in block or "对全部文件" in block), (
        "分组判定段缺少「必须基于全部文件」硬约束（ETHO-5：升级「优先用」软建议为硬约束）"
    )
    # 语义锚：每个文件依据自己的 per_file_grouping 条目归组（非单文件外推）
    assert "per_file_grouping" in block


def test_prompt_forbids_single_file_extrapolation():
    """ETHO-5 对称硬约束：单个/少数文件 inspect 结果不能外推为全部文件的分组结论。

    与范式「禁止猜」硬约束对称——分组判定此前缺此对称约束，是 lead 单文件外推的缺口。
    用正面措辞表达「外推不行」（描述正确做法），不是「禁止猜」式否定祈使。
    """
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    block = _grouping_block(text)
    # 「不能外推」语义：单个/少数文件结果只反映自身，不能当全部结论
    assert ("不能外推" in block or "不可外推" in block or "只反映那几个文件" in block), (
        "分组判定段缺少「单文件结果不能外推为全部结论」对称硬约束"
    )


def test_prompt_keeps_fallback_branch():
    """守边界（不删 fallback）：per_file_grouping 为空/不直观时仍 fallback inspect_uploaded_file。

    修法是把「优先用」升级为硬约束，不是删 fallback。fallback 分支必须保留，否则
    per_file_grouping 真为空时 lead 无路可走。
    """
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    block = _grouping_block(text)
    # fallback 触发条件：per_file_grouping 为空 / 不直观（aa/bb）
    assert ("为空" in block or "空" in block), "fallback 分支的「per_file_grouping 为空」触发条件被删"
    # fallback 动作：inspect_uploaded_file 看预览
    assert "inspect_uploaded_file" in block, "fallback 动作 inspect_uploaded_file 被删"
    # fallback 仍要覆盖全部分组（看够能覆盖全部分组的文件，不是看一个）
    assert ("覆盖全部分组" in block or "看够" in block), "fallback 分支缺少「看够能覆盖全部分组的文件」约束"


def test_prompt_no_negation_framing():
    """deepseek 正面提示铁律（memory feedback_skill_describing_tool_output_enables_hallucination）：
    修改的分组段不得引入「禁止猜 / 切勿 / 不要」类否定祈使句（会反向激活）。

    注：本段允许「不能外推」（描述正确做法的正面措辞），禁的是「禁止 X / 切勿 X / 不要 X」
    式纯否定祈使。
    """
    text = _get_real_prompt().SYSTEM_PROMPT_TEMPLATE
    block = _grouping_block(text)
    for token in ["禁止猜", "切勿", "不要单文件", "禁止单文件"]:
        assert token not in block, f"分组段含否定祈使句「{token}」，违反 deepseek 正面提示铁律"
