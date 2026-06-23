"""2026-06-23 ETHO-2 spec — lead prompt 批量扫描路径 + 无 bash subagent 指引守护测试。

防 sync paraphrase 削弱铁律（守 memory feedback_sync_protected_file_paraphrase_merge_weakens_constitution）。
prompt 铁律用「断言关键句存在」的轻量守护。

用 importlib 显式加载 **worktree** 的 prompt.py 源码（绕过主仓 venv 的 editable 链接，
保证测的是当前 worktree 改动），与 test_lead_prompt_template_zone_rule.py 同一加载模式。

ETHO-2 真因（spec §1.5）：lead 想批量扫所有上传文件的分组 → 尝试 task(subagent_type='bash')
→ 被 Pydantic schema 拒（bash 从来不在 _SubagentLiteral）→ 退化逐个 inspect_uploaded_file。
schema 层已正确拒绝（registry 自洽，见 test_subagent_registry_self_consistency.py），
唯一有效修法是 prompt 讲清批量扫描路径 + 明示没有 bash subagent 类型。

本测试守护 _build_subagent_section 渲染出的派遣段含：
  1. 没有 'bash' subagent 类型、shell 用 lead 自己的 bash tool（真正的新知识）
  2. 批量扫描全部上传文件的分组入口是 identify_ev19_template（pointer，机制 SSOT 在范式识别段）
  3. 9 条派遣硬约束未被改动（catastrophic forgetting 守卫）
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL prompt module source from this worktree (bypass editable link).
# ---------------------------------------------------------------------------
_PROMPT_FILE = Path(__file__).resolve().parents[1] / "packages" / "harness" / "deerflow" / "agents" / "lead_agent" / "prompt.py"

_REAL_PROMPT: ModuleType | None = None


def _get_real_prompt() -> ModuleType:
    global _REAL_PROMPT
    if _REAL_PROMPT is not None:
        return _REAL_PROMPT

    spec = importlib.util.spec_from_file_location(
        "deerflow.agents.lead_agent.prompt_real_etho2",
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


def test_lead_prompt_has_no_bash_subagent_type_guidance():
    """ETHO-2 核心：明示没有名为 bash 的 subagent 类型，shell 是 lead 自己的 bash tool。

    这一段是 schema 层拒绝 task(subagent_type='bash') 之后，prompt 层唯一能消除 lead
    想派 bash 动机的指引。必须在 _build_subagent_section 渲染的派遣段内（非 clarification 段）。
    """
    section = _get_real_prompt()._build_subagent_section(max_concurrent=3)
    # 关键：点明 bash 不是 subagent 类型（lead 形成派遣动机时可见）
    assert "没有名为 `bash` 的 subagent 类型" in section or '没有名为"bash"的 subagent 类型' in section, "派遣段必须明示没有 bash subagent 类型（ETHO-2 prompt 层解药）"
    # 关键：shell 走 lead 自己的 bash tool，不经 task 派遣
    assert "你自己的 `bash` tool" in section or "lead 本地工具" in section
    assert "task(subagent_type=...)" in section or "task(subagent_type=')" in section


def test_lead_prompt_has_batch_scan_pointer_to_identify():
    """ETHO-2：批量扫描全部上传文件的分组入口指向 identify_ev19_template（pointer）。

    守 SSOT：此处只做 pointer（入口工具名 + 「一次覆盖全部文件」），机制（per_file_grouping
    用法、fallback 条件）的 SSOT 在范式识别段（prompt.py ~L570）+ 反问合并规则段（~L493）。
    本测试只断言 pointer 存在，不断言机制复述（避免把 SSOT 复制钉死成契约）。
    """
    section = _get_real_prompt()._build_subagent_section(max_concurrent=3)
    assert "批量扫描全部上传文件的分组" in section
    assert "identify_ev19_template" in section
    # pointer 指向 SSOT 来源（范式识别段 / 反问合并规则），不复述机制
    assert "唯一来源" in section or "见" in section


def test_lead_prompt_dispatch_hard_constraints_preserved():
    """catastrophic forgetting 守卫：9 条派遣硬约束 + Guardrail anchor 未被新段扰动。

    新段是 ### 其他 subagent 与 ### 派遣硬约束 之间的兄弟段，绝不动 9 条编号规则。
    """
    section = _get_real_prompt()._build_subagent_section(max_concurrent=3)
    # 9 条编号规则完整存在
    import re

    nums = re.findall(r"^(\d+)\. \*\*", section, re.MULTILINE)
    assert nums[:9] == [str(i) for i in range(1, 10)], f"9 条派遣硬约束编号被破坏: {nums}"
    # 关键 Guardrail anchor 仍在（各至少 1 次）
    for anchor in [
        "Ev19TemplateGuardrailProvider",
        "InspectGateGuardrailProvider",
        "TaskHandoffAuthorizationProvider",
    ]:
        assert section.count(anchor) >= 1, f"Guardrail anchor 丢失: {anchor}"


def test_lead_prompt_batch_guidance_uses_positive_phrasing():
    """deepseek 正面提示铁律（memory feedback_skill_describing_tool_output_enables_hallucination）：
    新段不得用「禁止/不要/切勿」类否定祈使句（会反向激活）。"""
    section = _get_real_prompt()._build_subagent_section(max_concurrent=3)
    new_block = section[section.index("工具来源对齐") : section.index("派遣硬约束")]
    for token in ["禁止", "不要", "切勿"]:
        assert token not in new_block, f"新段含否定祈使句「{token}」，违反 deepseek 正面提示铁律"
