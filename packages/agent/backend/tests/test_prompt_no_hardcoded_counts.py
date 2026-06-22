"""Spec 2026-06-22 prompt-hardcoded-counts：prompt / 工具文案里硬编码数字根治。

防回滚契约测试。两类硬编码病：
- A 类：工具文案裸写会漂移的数字（「v0.1 已实现 5 个范式」），权威源
  SUPPORTED_PARADIGMS_V01 在旁却没引用。范式清单变了数字不变 → 漂移。
- B 类：code-executor prompt 没约束「摘要数字必须从结构化字段取」，致 LLM 嘴瓢。

本测试守护：文案里的范式数量 == len(SUPPORTED_PARADIGMS_V01)，且全量 prompt /
工具文案无「会漂移的裸数字」（无害描述性数字白名单除外）。后续新加 prompt 若再写
裸数字，CI 抓住。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_ROOT = _BACKEND_ROOT / "packages" / "harness" / "deerflow"
_TOOLS_BUILTINS_DIR = _HARNESS_ROOT / "tools" / "builtins"
_SUBAGENTS_BUILTINS_DIR = _HARNESS_ROOT / "subagents" / "builtins"
_LEAD_AGENT_DIR = _HARNESS_ROOT / "agents" / "lead_agent"

IDENTIFY_TOOL = _TOOLS_BUILTINS_DIR / "identify_ev19_template_tool.py"
PREP_TOOL = _TOOLS_BUILTINS_DIR / "prep_metric_plan_tool.py"
CODE_EXECUTOR = _SUBAGENTS_BUILTINS_DIR / "code_executor.py"
LEAD_PROMPT = _LEAD_AGENT_DIR / "prompt.py"

# A 类 ---------------------------------------------------------------


def _ev19_facts():
    """惰性导入 ethoinsight.ev19_facts（守 harness 可纯导入铁律）。"""
    return importlib.import_module("ethoinsight.ev19_facts")


def _load_identify_tool_from_worktree():
    """从 worktree 源文件加载 identify_ev19_template_tool 模块（单例，缓存在 sys.modules）。

    worktree 借主仓 venv，editable _editable_impl_deerflow_harness.pth 指向主仓，
    故 `import deerflow.tools.builtins.identify_ev19_template_tool` 会读到主仓旧代码（假绿）。
    用 importlib.util.spec_from_file_location 显式从 worktree 源加载，确保测的是改动后代码。

    缓存：重复调用返回同一模块实例，使 monkeypatch 模块级常量后能被后续 helper 观察到
    （否则每次 exec_module 会重置常量回源码原始值）。
    """
    mod_name = "identify_ev19_template_tool_worktree_under_test"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    mod_path = IDENTIFY_TOOL
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    assert spec and spec.loader, f"无法从 {mod_path} 构造 import spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _build_unsupported_via_pure_helper(paradigm_key: str, supported_overrides: set[str] | None = None):
    """调纯函数 _build_unsupported_result 构造 unsupported 结果。

    用 monkeypatched SUPPORTED_PARADIGMS_V01 时也走同一条文案生成路径，
    证明 message/hint 里的范式数量来自 len()，不是字面量。
    """
    identify = _load_identify_tool_from_worktree()
    facts = _ev19_facts()
    supported = supported_overrides if supported_overrides is not None else set(facts.SUPPORTED_PARADIGMS_V01)
    # 复用工具内已有的中文 label 权威 map（覆盖已支持 + 部分未支持范式）
    cn_map = identify._SUPPORTED_PARADIGM_CN_LABELS  # type: ignore[attr-defined]
    unsupported_label_map = identify._UNSUPPORTED_PARADIGM_CN_LABELS  # type: ignore[attr-defined]
    paradigm_label = unsupported_label_map.get(paradigm_key, paradigm_key)
    return identify._build_unsupported_result(
        paradigm_key=paradigm_key,
        paradigm_label=paradigm_label,
        supported_paradigms=supported,
        supported_cn_labels=cn_map,
    )


def test_identify_ev19_unsupported_message_uses_len_not_literal() -> None:
    """monkeypatch 扩到 7 个范式（含一个假的），文案必须变「7 个」+ 含第 7 个 label。

    改动前红：硬编码「5 个」。改动后绿：从 len() 派生。
    """
    facts = _ev19_facts()
    real_supported = set(facts.SUPPORTED_PARADIGMS_V01)
    fake_extra = "fake_tail_suspension_v2"
    extended = real_supported | {fake_extra}
    # 给假范式一个中文 label，证明清单是动态生成而非手举
    identify_mod = _load_identify_tool_from_worktree()
    original_labels = dict(identify_mod._SUPPORTED_PARADIGM_CN_LABELS)
    identify_mod._SUPPORTED_PARADIGM_CN_LABELS = {
        **identify_mod._SUPPORTED_PARADIGM_CN_LABELS,
        fake_extra: "假范式 (FAKE)",
    }
    try:
        result = _build_unsupported_via_pure_helper("shoaling", supported_overrides=extended)
    finally:
        identify_mod._SUPPORTED_PARADIGM_CN_LABELS = original_labels

    message = result["message"]
    assert "7 个范式" in message, f"message 应含动态计数「7 个范式」, 实际: {message!r}"
    assert "假范式 (FAKE)" in message, f"message 应含第 7 个范式的动态 label, 实际: {message!r}"
    # 反证：不再是旧的硬编码「5」
    assert "5 个范式" not in message


def test_identify_ev19_message_matches_supported_paradigms() -> None:
    """message 里的范式数量 == len(SUPPORTED_PARADIGMS_V01)（防回归）。"""
    facts = _ev19_facts()
    result = _build_unsupported_via_pure_helper("shoaling")
    expected_count = len(facts.SUPPORTED_PARADIGMS_V01)
    message = result["message"]
    assert f"{expected_count} 个范式" in message, (
        f"message 应含 len(SUPPORTED_PARADIGMS_V01)={expected_count} 个范式, 实际: {message!r}"
    )
    # hint 也应一致
    assert f"{expected_count} 个范式之一" in result["hint"], (
        f"hint 应含 {expected_count} 个范式之一, 实际: {result['hint']!r}"
    )


def test_identify_ev19_supported_paradigms_field_complete() -> None:
    """supported_paradigms 字段 == sorted(SUPPORTED_PARADIGMS_V01)（结构化字段不漂移）。"""
    facts = _ev19_facts()
    result = _build_unsupported_via_pure_helper("shoaling")
    assert result["supported_paradigms"] == sorted(facts.SUPPORTED_PARADIGMS_V01)


def test_prep_metric_plan_docstring_has_no_literal_count() -> None:
    """prep_metric_plan 的 paradigm docstring 不含裸「5 个」+ 不手举范式清单。

    docstring 经 @tool(parse_docstring=True) 进 LLM 可见工具描述，是漂移高发区。
    权威源是 SUPPORTED_PARADIGMS_V01 / identify_ev19_template 返回，docstring 只指向它。
    """
    text = PREP_TOOL.read_text(encoding="utf-8")
    # 提取 paradigm 参数的 docstring 段（Args 段里 paradigm: 开头到下一行 groups: 之前）
    paradigm_section = _extract_paradigm_docstring(text)
    assert "5 个" not in paradigm_section, (
        f"paradigm docstring 不应含裸「5 个」, 实际段落:\n{paradigm_section}"
    )
    # 不应出现「v0.1 仅支持以下 N 个」式完整清单声明（裸数字 + 清单）。
    # 「形如 'epm' / 'open_field' 等」「缩写如 'oft'/'fst'/'ldb'」是 canonical key
    # 格式的开放示例，不是能力声明，允许（漂移由下方「指向权威源」断言兜住）。
    import re as _re

    list_declaration = _re.search(r"仅支持.{0,12}\d+\s*个|支持以下\s*\d+\s*个", paradigm_section)
    assert list_declaration is None, (
        f"paradigm docstring 不应含「仅支持 N 个」式清单声明, 实际段落:\n{paradigm_section}"
    )
    # 应指向权威源（identify_ev19_template / SUPPORTED_PARADIGMS_V01 任一）
    assert (
        "identify_ev19_template" in paradigm_section
        or "SUPPORTED_PARADIGMS_V01" in paradigm_section
    ), f"paradigm docstring 应指向权威源, 实际段落:\n{paradigm_section}"


def _extract_paradigm_docstring(text: str) -> str:
    """从 prep_metric_plan_tool.py 源码里提取 docstring 内 paradigm 参数的说明段。

    定位 docstring 里（非函数签名）的 paradigm 参数：签名是 `paradigm: str,`，
    docstring 里是 `paradigm: 范式 canonical key...`。用「canonical key」作锚点。
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        # docstring 里的 paradigm 段：缩进 + paradigm: + 含中文说明（非签名的 str,）
        stripped = line.lstrip()
        if stripped.startswith("paradigm:") and "canonical key" in line:
            start = i
            break
    if start is None:
        return ""
    end = start + 1
    while end < len(lines):
        nxt = lines[end].strip()
        # 段落终止：遇到下一个参数（groups: / Returns: 等）或 docstring 结束
        if nxt.startswith("groups:") or nxt.startswith("Returns:") or nxt == '"""':
            break
        end += 1
    return "\n".join(lines[start:end])


# B 类 ---------------------------------------------------------------


def test_code_executor_prompt_has_number_discipline() -> None:
    """code_executor system_prompt 含 <summary_number_discipline> 段 + 「按 id 去重」口径约束。"""
    text = CODE_EXECUTOR.read_text(encoding="utf-8")
    assert "<summary_number_discipline>" in text, (
        "code_executor.py system_prompt 应含 <summary_number_discipline> 段（约束摘要数字从结构化字段取）"
    )
    assert "</summary_number_discipline>" in text
    # 约束指标计数口径（按 id 去重，不是 ×subject 展开）
    assert "按 id 去重" in text or "去重" in text, (
        "数字纪律段应明确指标计数口径（按 id 去重，非 ×subject 展开数）"
    )


# 全量扫描契约 --------------------------------------------------------


# 受 lint 的文件：所有 subagent system_prompt + 工具文案 + lead prompt
def _collect_prompt_files() -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(_SUBAGENTS_BUILTINS_DIR.glob("*.py")))
    files.extend(sorted(_TOOLS_BUILTINS_DIR.glob("*.py")))
    files.append(LEAD_PROMPT)
    # 排除 __init__ / 非模块
    return [f for f in files if f.name != "__init__.py"]


# 会漂移的裸数字模式（声称事实 + 权威源存在 + 随版本/数据变化）
_DRIFTING_PATTERNS = [
    r"已实现\s*\d+\s*个范式",
    r"已实现\s*\d+\s*个",
    r"仅支持.{0,12}\d+\s*个",
    r"支持以下\s*\d+\s*个",
    r"\d+\s*个当前支持",
    r"\d+\s*个范式",
]


@pytest.mark.parametrize("py_file", _collect_prompt_files(), ids=lambda p: p.name)
def test_no_hardcoded_drifting_counts_in_prompts(py_file: Path) -> None:
    """全量扫描：所有 subagent prompt + 工具文案 + lead prompt 不含会漂移的裸数字。

    无害描述性数字（「140 行 bash」「2-3 个 AI message」「每条 <80 字」）不匹配上述模式，
    不受影响。本测试是防回归契约：以后新加 prompt 若再写「N 个范式」裸数字，CI 抓住。
    """
    import re

    text = py_file.read_text(encoding="utf-8")
    for pattern in _DRIFTING_PATTERNS:
        matches = re.findall(pattern, text)
        # lead prompt 的「当前支持的范式范围」段允许出现范式清单（来自运行时注入的占位符
        # 上下文），但禁止裸「N 个」计数。此处 regex 已只匹配裸数字，清单本身不触发。
        assert not matches, (
            f"{py_file.name}: 含会漂移的裸数字 {matches!r} (模式 {pattern!r})。"
            f" 凡声称事实的范式/指标/任务计数必须引用权威源 (SUPPORTED_PARADIGMS_V01 /"
            f" plan 计数 / run_metric_plan 统计), 不能写字面量。"
            f" 详见 docs/superpowers/specs/2026-06-22-prompt-hardcoded-counts-spec.md"
        )


def test_lead_prompt_no_hardcoded_paradigm_list() -> None:
    """lead prompt 不手举具体范式清单作为「当前支持」事实声明。

    权威源是 SUPPORTED_PARADIGMS_V01（ev19_facts）/ identify_ev19_template 返回。
    lead prompt L299 已声明「以 catalog 为准」——L283 能力声明主文 + L456 example +
    L540 options 都不应手举会漂移的清单。few-shot options 可用占位符示意。
    """
    text = LEAD_PROMPT.read_text(encoding="utf-8")
    # L283 能力声明段：不应出现「(6 个 ... 范式)」式硬编码计数标题
    assert "(6 个" not in text and "(5 个" not in text, (
        "lead prompt 不应在能力声明标题里硬编码范式计数；应指向 SUPPORTED_PARADIGMS_V01"
    )
    # 能力声明段不应手举范式缩写清单（「EPM/OFT/FST/LDB/Zero Maze/TST」串）。
    # 权威源是 SUPPORTED_PARADIGMS_V01，指向它即可，不必列举。
    import re

    paradigm_abbr_chain = re.search(r"EPM\s*/\s*OFT\s*/\s*(FST|LDB)", text)
    assert paradigm_abbr_chain is None, (
        f"lead prompt 不应手举范式缩写清单 {paradigm_abbr_chain.group(0)!r}; 指向 SUPPORTED_PARADIGMS_V01 即可"
    )
    # few-shot options 不应把真实范式清单当作固定菜单（范式扩展会漂移）
    # 允许出现单个范式名（措辞性），但不允许把 6 个范式名串成 options=[...] 字面菜单
    import re

    options_menu = re.search(r'options\s*=\s*\[([^\]]{40,})\]', text)
    if options_menu:
        menu_inner = options_menu.group(1)
        # 若菜单里同时出现 EPM + OFT + FST + LDB 多个范式名 = 真实清单当菜单 = 漂移
        paradigm_hits = sum(
            1 for name in ["EPM", "OFT", "FST", "LDB", "Zero Maze", "TST"] if name in menu_inner
        )
        assert paradigm_hits < 3, (
            f"lead prompt few-shot options 把 {paradigm_hits} 个真实范式名当固定菜单,"
            f" 范式扩展会漂移。应改占位符示意或运行时注入。菜单内容: {menu_inner!r}"
        )
