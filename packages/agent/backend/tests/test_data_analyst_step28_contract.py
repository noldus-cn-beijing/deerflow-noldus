"""data-analyst step 3 分步填模板（产物 + 封口）契约测试。

2026-06-23（spec data-analyst-seal-stepwise-fill-template）：step 3 从旧的「产出与交付
合一 = 一次性发出 seal_data_analyst_handoff tool_call」反转为「分步填模板」——判读不再
塞进 seal args（与 reasoning_tokens 共享 max_tokens 撞腰斩），改为 fill_* 逐字段填 +
finalize 封口。本文件锁定新 step 3 的正面 framing 契约。
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

def test_step3_uses_stepwise_fill_template_flow():
    """Step 3 = 分步填模板（产物 + 封口），不再是旧的一次性 seal（spec 2026-06-23）。

    旧「产出分析 = 发出 seal_data_analyst_handoff 的 tool_call（产出与交付合一）」已移除
    （seal args 撞 max_tokens 狭颈腰斩）。新流程：harness 预置 in_progress 模板 → fill_*
    逐字段填 → finalize 封口。
    """
    p = _prompt()

    # 旧「产出与交付合一」措辞必须消失
    assert "产出与交付合一" not in p
    assert "发出 seal_data_analyst_handoff 的 tool_call" not in p

    # 新流程核心：分步填模板 + finalize 封口
    assert "分步填模板" in p
    assert "fill_data_analyst_text_list" in p
    assert "finalize_data_analyst_handoff" in p
    # 模板已预置（省 agent 第一轮探查）
    assert "in_progress" in p
    # Hard rule: must go through fill_* + finalize, never write_file
    assert "严禁直接 write_file 写 handoff_data_analyst.json" in p


def test_old_produce_deliver_merged_phrasing_gone():
    """旧「产出与交付合一」「这次工具调用本身既是产出也是落库」措辞必须消失（spec 2026-06-23）。

    新流程的交付动作是 fill_* + finalize 序列，不再用「产出/交付合一」framing。
    """
    p = _prompt()
    assert "产出与交付合一" not in p
    assert "这次工具调用本身既是产出也是落库" not in p
    assert "你的结论第一次成文" not in p


def test_step3_lists_fill_tools_and_finalize():
    """Step 3 lists the fill_* tools + finalize as the delivery sequence (spec 2026-06-23).

    旧 step 3 的「字段清单」（status/key_findings/.../parameter_audit_findings）随一次性
    seal 一起移除；新 step 3 列的是 fill_* 工具序列 + finalize。空数组兜底语义保留（默认值
    已空、可不填该字段）。
    """
    p = _prompt()
    # fill 工具序列
    assert "fill_data_analyst_text_list" in p
    assert "fill_data_analyst_record_list" in p
    assert "fill_data_analyst_gate_signals" in p
    assert "finalize_data_analyst_handoff" in p
    # 空数组兜底语义保留
    assert "空数组" in p or "[]" in p
    # parameter_audit_findings 无 fill 入口（恒空、不产出）——旧清单措辞已移除
    assert "status/key_findings/outlier_findings/excluded_metrics/" not in p
