"""2026-06-23 ETHO-2 spec §四 测试 1/2/3/4 —— subagent registry 自洽性守护。

spec §1.2 三重实证推翻了原 infra 根因：bash 从来不在 ``get_available_subagent_names()``
返回值里，``task(subagent_type='bash')`` 当前已在 Pydantic schema 层（``_SubagentLiteral``）
被正确拒绝。本测试文件**坐实现状 + 守不变式**，非红→绿（spec §四 明示）。

四条测试：
  1. test_available_names_excludes_bash —— 坐实 bash 不在 available、get_subagent_config('bash') is None
  2. test_registry_self_consistency_invariant —— 防御性不变式：每个 available name 都有 config
  3. test_subagent_literal_excludes_bash —— _SubagentLiteral 生成的 Literal 不含 bash（schema 层会拒）
  4. test_runtime_fallback_message_preserved —— task_tool runtime 兜底文案仍含「请改用」指引

注：worktree 借主仓 venv（editable 指主仓），但本文件测的是 registry/task_tool 的**行为不变式**，
主仓与 worktree 在这些模块上一致（本 spec 不改它们的逻辑，只加注释）。conftest 的 executor mock
破了 subagents 导入环，使 task_tool 可正常 import。
"""

from __future__ import annotations

import importlib
from typing import get_args

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.registry import (
    get_available_subagent_names,
    get_subagent_config,
)

task_tool_module = importlib.import_module("deerflow.tools.builtins.task_tool")


# ---------------------------------------------------------------------------
# 测试 1：坐实现状（改前已绿）—— bash 不在 available、无 config
# ---------------------------------------------------------------------------
def test_available_names_excludes_bash():
    """文档化 spec §1.2 实证：bash 从来不是可派 subagent。

    这条不是红→绿，它坐实「现状已正确」——schema 层据此拒绝 task(subagent_type='bash')。
    """
    available = get_available_subagent_names()
    assert "bash" not in available, f"bash 不应在 available names 中, 实际: {available}"
    # BUILTIN_SUBAGENTS 是 available 的来源之一，也不含 bash
    assert "bash" not in BUILTIN_SUBAGENTS
    # bash 没有对应 config（runtime 兜底会返回 Unknown subagent type）
    assert get_subagent_config("bash") is None


# ---------------------------------------------------------------------------
# 测试 2：防御性不变式（spec §2.2）—— 校验集合恒 = 可派集合
# ---------------------------------------------------------------------------
def test_registry_self_consistency_invariant():
    """每个 available name 都必须 get_subagent_config(name) is not None。

    守 spec §2.2 的结构性不变式：schema 校验集合（_SubagentLiteral，module-load 时由
    get_available_subagent_names 生成）必须等于可派集合（get_subagent_config 非 None）。
    将来若有人往 custom_agents 注入一个 get_subagent_config 查不到的名字，会重现
    「schema 放行又 runtime 失败」的不自洽——此测试红，提示去修。
    """
    available = get_available_subagent_names()
    assert available, "available names 不应为空"
    undispatchable = [n for n in available if get_subagent_config(n) is None]
    assert not undispatchable, f"自洽不变式被破坏: 这些 name 通过了 schema 校验但 get_subagent_config 返回 None (schema 放行又 runtime 失败): {undispatchable}"


# ---------------------------------------------------------------------------
# 测试 3：_SubagentLiteral 不含 bash（间接验 schema 层会拒）
# ---------------------------------------------------------------------------
def test_subagent_literal_excludes_bash():
    """task_tool 的 _SubagentLiteral（烘进 JSON Schema enum）不含 'bash'。

    spec §1.2 实证 3：_SubagentLiteral 在 task_tool module load 时由
    get_available_subagent_names() 生成。它不含 bash → task(subagent_type='bash')
    在进函数体前就被 Pydantic 拒（报 'Input should be ...'）。
    """
    literal = task_tool_module._SubagentLiteral
    literal_args = get_args(literal)
    assert "bash" not in literal_args, f"_SubagentLiteral 不应含 bash (否则 task(subagent_type='bash') 会绕过 schema), 实际: {literal_args}"
    # 它应等于 get_available_subagent_names()（自洽）
    assert set(literal_args) == set(get_available_subagent_names())


# ---------------------------------------------------------------------------
# 测试 4：runtime 兜底文案保留「请改用」指引（第二道防线，spec §2.4）
# ---------------------------------------------------------------------------
def test_runtime_fallback_message_preserved(monkeypatch):
    """模拟 app_config 漂移（非法名绕过 schema 进函数体），runtime 兜底仍返回指引文案。

    spec §1.3 路径②：非法名若绕过 schema 进 task_tool 函数体，get_subagent_config=None
    兜底（task_tool.py L405-411）返回含「请改用 task(...)」的中文指引。本 spec 不改 task_tool，
    此测试守这道第二防线不被破坏（memory feedback_deny_messages_must_direct）。
    """
    # 模拟一个绕过 schema 的非法名（如运行期 custom_agents 注入了不可派名字）
    monkeypatch.setattr(task_tool_module, "get_subagent_config", lambda *_a, **_k: None)
    monkeypatch.setattr(task_tool_module, "get_available_subagent_names", lambda *_, **__: ["chart-maker"])

    coroutine = getattr(task_tool_module.task_tool, "coroutine", None)
    kwargs = dict(
        runtime=None,
        description="执行任务",
        prompt="do work",
        subagent_type="bash",  # 模拟绕过 schema 的非法名
        tool_call_id="tc-1",
    )
    import asyncio

    result = asyncio.run(coroutine(**kwargs)) if coroutine is not None else task_tool_module.task_tool.func(**kwargs)

    assert "Error: Unknown subagent type 'bash'" in result
    # 守 memory feedback_deny_messages_must_direct：deny 文案必须含明确「请改用」指引
    assert "请改用" in result
    assert "task(subagent_type=" in result
