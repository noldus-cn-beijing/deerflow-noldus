"""Test that SubagentExecutor accepts authorized_handoff_paths and stores it."""

import sys
from unittest.mock import MagicMock

import pytest

_MOCKED_MODULE_NAMES = [
    "deerflow.agents",
    "deerflow.agents.thread_state",
    "deerflow.agents.middlewares",
    "deerflow.agents.middlewares.thread_data_middleware",
    "deerflow.sandbox",
    "deerflow.sandbox.middleware",
    "deerflow.sandbox.security",
    "deerflow.models",
]


@pytest.fixture(scope="module")
def _setup_executor():
    """Replace mock with real executor module."""
    original_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULE_NAMES}
    original_executor = sys.modules.get("deerflow.subagents.executor")

    if "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]

    for name in _MOCKED_MODULE_NAMES:
        sys.modules[name] = MagicMock()

    yield

    for name in _MOCKED_MODULE_NAMES:
        if original_modules.get(name) is not None:
            sys.modules[name] = original_modules[name]
        elif name in sys.modules:
            del sys.modules[name]

    if original_executor is not None:
        sys.modules["deerflow.subagents.executor"] = original_executor
    elif "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]


@pytest.fixture
def SubagentExecutor_cls(_setup_executor):
    from deerflow.subagents.executor import SubagentExecutor
    return SubagentExecutor


@pytest.fixture
def SubagentConfig_cls(_setup_executor):
    from deerflow.subagents.config import SubagentConfig
    return SubagentConfig


def _minimal_config(SubagentConfig_cls):
    return SubagentConfig_cls(
        name="test-agent",
        description="test",
        system_prompt="you are a test",
    )


def test_executor_accepts_authorized_handoff_paths(SubagentExecutor_cls, SubagentConfig_cls):
    """Executor accepts the new parameter as a keyword argument."""
    config = _minimal_config(SubagentConfig_cls)
    executor = SubagentExecutor_cls(
        config=config,
        tools=[],
        authorized_handoff_paths={"/mnt/user-data/workspace/handoff_code_executor.json"},
    )
    assert executor.authorized_handoff_paths == {
        "/mnt/user-data/workspace/handoff_code_executor.json"
    }


def test_executor_authorized_paths_defaults_to_empty_set(SubagentExecutor_cls, SubagentConfig_cls):
    """When the parameter is omitted, attribute defaults to empty set."""
    config = _minimal_config(SubagentConfig_cls)
    executor = SubagentExecutor_cls(config=config, tools=[])
    assert executor.authorized_handoff_paths == set()


def test_executor_authorized_paths_none_normalized_to_empty(SubagentExecutor_cls, SubagentConfig_cls):
    """Passing None explicitly normalizes to empty set."""
    config = _minimal_config(SubagentConfig_cls)
    executor = SubagentExecutor_cls(
        config=config, tools=[], authorized_handoff_paths=None
    )
    assert executor.authorized_handoff_paths == set()
