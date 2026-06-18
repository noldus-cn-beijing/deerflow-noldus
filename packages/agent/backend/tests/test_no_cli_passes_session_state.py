"""Spec 4 (P4) §4 — 守护测试：B 类 session 状态绝不作为 LLM 可传入参。

spec §0/§1 的核心纪律：横切 session 状态（column_aliases / parameter_overrides）
是「一个 thread 一份」的共享态，必须每个 tool 从自己的 session 注入态（
``runtime.state["thread_data"]`` + ``read_context``）自取，**禁止让 LLM 在工具签名
里通过参数传**——传参 = 把状态传递交给调用方记性，下一路径就漏接（见 §6 元收口）。

本守护用 AST 解析 prep 工具的函数签名（@tool 装饰、暴露给 LLM 的入参），断言
``column_aliases`` / ``parameter_overrides`` **永远不出现在 LLM 可传参数名里**。
有人在签名里加这种参数就当场红，避免「靠 prompt 记得拼」的回归。

注意区分：
- ``column_aliases`` / ``parameter_overrides`` → B 类 session 状态，**禁止** LLM 传，
  必须自读 context（本测试守护）。
- ``groups`` / ``paradigm`` → 合法的 lead 翻译口子（lead 把用户 ask_clarification
  答案翻译成 groups dict / 把 handoff 的 paradigm 原样传入），**不算**泄漏，不在此守护。
"""

from __future__ import annotations

import ast
from pathlib import Path

# B 类 session 状态关键词：绝不应作为 @tool 工具的 LLM 可传入参出现。
# （prep_* 工具体内从 read_context 自取这些，是正范例；签名暴露才是回归。）
FORBIDDEN_LLM_PARAM_NAMES = {
    "column_aliases",
    "column_aliases_file",  # CLI 文件注入口仅供测试/调试，工具层不该暴露给 LLM
    "parameter_overrides",
}

# prep_* 工具所在目录 —— spec §0 的正范例（prep_metric_plan / prep_chart_plan）。
TOOLS_BUILTINS_DIR = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "harness"
    / "deerflow"
    / "tools"
    / "builtins"
)

# runtime / workspace_dir 是 harness 注入或合法配置口子，不算 LLM 传 B 类状态。
_INJECTED_PARAMS = {"runtime", "workspace_dir"}


def _tool_function_param_names(func_node: ast.FunctionDef) -> set[str]:
    """提取一个 @tool 函数暴露的参数名（排除 runtime 等注入项）。"""
    names: set[str] = set()
    args = func_node.args
    for a in list(args.args) + list(args.posonlyargs) + list(args.kwonlyargs):
        if isinstance(a, ast.arg):
            names.add(a.arg)
    return names - _INJECTED_PARAMS


def _iter_tool_functions(directory: Path):
    """遍历目录下所有 @tool 装饰的函数（返回 (file, func_name, param_names)）。"""
    for src in sorted(directory.glob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # 检测 @tool 装饰器（langchain.tools.tool）
            is_tool = any(
                (isinstance(d, ast.Call) and getattr(d.func, "id", None) == "tool")
                or (isinstance(d, ast.Name) and d.id == "tool")
                for d in node.decorator_list
            )
            if is_tool:
                yield src.name, node.name, _tool_function_param_names(node)


def test_no_tool_exposes_session_state_as_llm_param():
    """守护：@tool 工具的 LLM 可传签名里不含 column_aliases / parameter_overrides。

    spec §0/§1：这些是 session 级共享态，必须 ``runtime.state`` / ``read_context``
    自取。红了 = 有人在工具签名里加了 B 类状态参数（把状态传递交给 LLM 记性）。
    """
    violations: list[str] = []
    found_any_tool = False
    for fname, func_name, params in _iter_tool_functions(TOOLS_BUILTINS_DIR):
        found_any_tool = True
        leaked = params & FORBIDDEN_LLM_PARAM_NAMES
        if leaked:
            violations.append(
                f"{fname}::{func_name} 暴露了 B 类 session 状态参数: {sorted(leaked)}"
                f" —— 改为函数体内从 read_context / runtime.state 自取。"
            )
    assert found_any_tool, "未发现任何 @tool 函数，AST 扫描可能失效，请检查本测试。"
    assert not violations, "B 类 session 状态泄漏到 LLM 工具签名:\n" + "\n".join(violations)


def test_prep_tools_self_read_context():
    """守护：prep_metric_plan / prep_chart_plan 必须在函数体内调 read_context
    自取 column_aliases（spec §0 正范例）。红了 = 某个 prep 工具退化为靠参数传。
    """
    targets = {
        "prep_metric_plan_tool.py": "prep_metric_plan_tool",
        "prep_chart_plan_tool.py": "prep_chart_plan_tool",
    }
    for fname, func_name in targets.items():
        src = TOOLS_BUILTINS_DIR / fname
        assert src.exists(), f"missing {fname}"
        tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name]
        assert funcs, f"{fname} 缺 {func_name}"
        body_src = ast.get_source_segment(src.read_text(encoding="utf-8"), funcs[0]) or ""
        assert "read_context" in body_src, (
            f"{fname}::{func_name} 未调用 read_context 自取 session 状态 —— "
            f"违反 spec §0 正范例（工具内部确定性自读）。"
        )
