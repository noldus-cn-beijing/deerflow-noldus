"""Test thinking_enabled propagation for data-analyst subagent.

Spec: 2026-06-03-data-analyst-enable-thinking-spec.md, §4.1
- Config field default value: SubagentConfig().thinking_enabled is False
- Only data-analyst has thinking_enabled=True; all other builtins remain False
- Executor passes self.config.thinking_enabled to create_chat_model
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from deerflow.subagents.builtins import (
    BASH_AGENT_CONFIG,
    CHART_MAKER_CONFIG,
    CODE_EXECUTOR_CONFIG,
    DATA_ANALYST_CONFIG,
    GENERAL_PURPOSE_CONFIG,
    KNOWLEDGE_ASSISTANT_CONFIG,
    REPORT_WRITER_CONFIG,
    BUILTIN_SUBAGENTS,
)
from deerflow.subagents.config import SubagentConfig

# ── 4.1(1): config 字段默认值（向后兼容锚点） ──────────────────────────────


def test_thinking_enabled_default_false():
    """新建 SubagentConfig 实例 thinking_enabled 默认 False，向后兼容。"""
    cfg = SubagentConfig(name="probe", description="d")
    assert cfg.thinking_enabled is False


def test_thinking_enabled_can_be_set_true():
    """显式设 True 后正确返回。"""
    cfg = SubagentConfig(name="probe", description="d", thinking_enabled=True)
    assert cfg.thinking_enabled is True


# ── 4.1(2): 仅 data-analyst 开、其余全关（最关键的"不一开全开"锚点） ──


# Collect every builtin config exported in BUILTIN_SUBAGENTS registry,
# plus the raw module-level constants, so we test from both angles.
_ALL_BUILTIN_CONFIGS_MODULE = {
    "data-analyst": DATA_ANALYST_CONFIG,
    "code-executor": CODE_EXECUTOR_CONFIG,
    "chart-maker": CHART_MAKER_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
    "knowledge-assistant": KNOWLEDGE_ASSISTANT_CONFIG,
}


def test_no_builtin_has_thinking_enabled():
    """所有 builtin subagent 的 thinking_enabled 均为 False。

    历史：spec 2026-06-03-data-analyst-enable-thinking-spec.md 曾为 data-analyst
    单独开 thinking（当时唯一开 think 的洞察型 subagent）。**2026-06-18 第四轮 EPM
    dogfood 推翻该决策**：data-analyst 判读判据已 100% 在 by-experiment skill +
    outlier 旁路文件里（查表+选择，无现场多步推理），thinking 在此纯属把整套判读
    在每个 model turn 重演一遍，撞穿 max_tokens=4096 输出预算（与 seal tool_call
    arguments 共享），导致 ~9 轮空转。data-analyst 已改 model=deepseek-v4-pro-summary
    （无 thinking），与 code-executor/chart-maker/report-writer 一致。

    这条测试现在防止的是"未来有人又顺手把某个 builtin 的 thinking 开了"。
    """
    for name, cfg in _ALL_BUILTIN_CONFIGS_MODULE.items():
        assert cfg.thinking_enabled is False, (
            f"{name} 必须为 False（2026-06-18 起所有 builtin 均不开 thinking；"
            f"data-analyst 的 thinking 因撞 4096 输出预算空转已撤销）"
        )


def test_builtin_registry_matches_module_constants():
    """BUILTIN_SUBAGENTS 中每个 config 与模块级常量一致。"""
    registry = dict(BUILTIN_SUBAGENTS)
    assert registry["data-analyst"] is DATA_ANALYST_CONFIG
    assert registry["code-executor"] is CODE_EXECUTOR_CONFIG
    assert registry["chart-maker"] is CHART_MAKER_CONFIG
    assert registry["report-writer"] is REPORT_WRITER_CONFIG


def test_builtin_registry_no_thinking_enabled():
    """遍历 BUILTIN_SUBAGENTS 注册表断言所有 builtin thinking_enabled 全 False。

    见 test_no_builtin_has_thinking_enabled 的 docstring：2026-06-18 起 data-analyst
    的 thinking 已撤销，全部 builtin 走非 thinking 模型。
    """
    for name, cfg in BUILTIN_SUBAGENTS.items():
        assert cfg.thinking_enabled is False, (
            f"registry {name} 必须为 False（2026-06-18 起所有 builtin 均不开 thinking）"
        )


def test_data_analyst_uses_non_thinking_model():
    """data-analyst 用 deepseek-v4-pro-summary（无 thinking），与三个产出型 subagent 一致。

    根因锚点（2026-06-18 第四轮 EPM dogfood）：原 model="inherit"（v4-pro+thinking）
    每轮在 thinking 里重演整套判读，撞 4096 输出预算 → seal arguments 被腰斩 → 空转。
    判读判据全在 skill，无现场推理，换非 thinking 模型后预算全留给 arguments。
    """
    assert DATA_ANALYST_CONFIG.model == "deepseek-v4-pro-summary", (
        f"data-analyst 必须用非 thinking 模型，实际 {DATA_ANALYST_CONFIG.model}"
    )
    assert DATA_ANALYST_CONFIG.thinking_enabled is False


# ── 4.1(3): executor 传递 config 值给 create_chat_model ─────────────────

# conftest.py mocks ``deerflow.subagents.executor`` to break a circular import
# chain. To test the *real* SubagentExecutor._create_agent, we must undo that
# mock and supply our own mocks for the modules that executor.py depends on.
# The pattern follows test_subagent_executor.py's session fixture.

_EXECUTOR_DEP_MOCKS = [
    "deerflow.agents",
    "deerflow.agents.thread_state",
    "deerflow.agents.middlewares",
    "deerflow.agents.middlewares.thread_data_middleware",
    "deerflow.sandbox",
    "deerflow.sandbox.middleware",
    "deerflow.sandbox.security",
    "deerflow.models",
    "deerflow.models.factory",
    "deerflow.config",
    "deerflow.config.app_config",
    "deerflow.subagents.guardrails",
    "deerflow.subagents.guardrails.executor",
    "deerflow.agents.middlewares.guardrail_middleware",
    "deerflow.skills",
    "deerflow.mcp",
]


@pytest.fixture()
def _real_executor():
    """Import the real SubagentExecutor with its deps mocked (one shot)."""
    # Save originals
    original_deps = {name: sys.modules.get(name) for name in _EXECUTOR_DEP_MOCKS}
    original_exec = sys.modules.get("deerflow.subagents.executor")

    # Remove conftest mock
    if "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]

    for name in _EXECUTOR_DEP_MOCKS:
        sys.modules[name] = MagicMock()

    # ThreadState needs to look like a real class for langchain's create_agent
    thread_state_mock = sys.modules["deerflow.agents.thread_state"]
    thread_state_mock.ThreadState = type("ThreadState", (), {"__annotations__": {"messages": list}})

    # Avoid needing a fully-real langchain create_agent — we only care about the
    # create_chat_model call; the downstream create_agent can be a no-op.
    import langchain.agents
    langchain.agents.create_agent = MagicMock(return_value=MagicMock())

    from deerflow.subagents.executor import SubagentExecutor

    yield SubagentExecutor

    # Restore
    for name in _EXECUTOR_DEP_MOCKS:
        if original_deps[name] is not None:
            sys.modules[name] = original_deps[name]
        elif name in sys.modules:
            del sys.modules[name]

    if original_exec is not None:
        sys.modules["deerflow.subagents.executor"] = original_exec
    elif "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]


class TestExecutorThinkingPropagation:
    """验证 SubagentExecutor._create_agent 把 config.thinking_enabled 传给 create_chat_model。"""

    def test_passes_thinking_enabled_true(self, _real_executor):
        """thinking_enabled=True 的 config → create_chat_model 收到 thinking_enabled=True。"""
        SubagentExecutor = _real_executor
        cfg = SubagentConfig(name="test", description="d", thinking_enabled=True)
        executor = SubagentExecutor(config=cfg, tools=[], trace_id="test-true")
        with patch.object(executor, "_build_middlewares", return_value=[]):
            with patch("deerflow.subagents.executor.create_chat_model") as mock_create:
                executor._create_agent()
                mock_create.assert_called_once()
                _, kwargs = mock_create.call_args
                assert kwargs["thinking_enabled"] is True, (
                    f"必须传 thinking_enabled=True，实际 {kwargs.get('thinking_enabled')}"
                )

    def test_passes_thinking_enabled_false(self, _real_executor):
        """thinking_enabled=False 的 config → create_chat_model 收到 thinking_enabled=False。"""
        SubagentExecutor = _real_executor
        cfg = SubagentConfig(name="test", description="d", thinking_enabled=False)
        executor = SubagentExecutor(config=cfg, tools=[], trace_id="test-false")
        with patch.object(executor, "_build_middlewares", return_value=[]):
            with patch("deerflow.subagents.executor.create_chat_model") as mock_create:
                executor._create_agent()
                mock_create.assert_called_once()
                _, kwargs = mock_create.call_args
                assert kwargs["thinking_enabled"] is False, (
                    f"必须传 thinking_enabled=False，实际 {kwargs.get('thinking_enabled')}"
                )

    def test_default_config_passes_false(self, _real_executor):
        """未显式设 thinking_enabled 的 config（默认 False）→ 传 False。"""
        SubagentExecutor = _real_executor
        cfg = SubagentConfig(name="test", description="d")  # no thinking_enabled kwarg
        executor = SubagentExecutor(config=cfg, tools=[], trace_id="test-default")
        with patch.object(executor, "_build_middlewares", return_value=[]):
            with patch("deerflow.subagents.executor.create_chat_model") as mock_create:
                executor._create_agent()
                mock_create.assert_called_once()
                _, kwargs = mock_create.call_args
                assert kwargs["thinking_enabled"] is False
