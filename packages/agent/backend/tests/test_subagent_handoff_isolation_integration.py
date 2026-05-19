"""Integration: HandoffIsolationProvider attached to SubagentExecutor blocks
unauthorized handoff reads at the middleware level.

We don't run a full subagent here; we verify the provider constructs correctly
and would deny an unauthorized read. Full middleware-chain integration is
tested implicitly when the existing executor tests pass (they verify
_create_agent completes without error, and the HandoffIsolationProvider
is in the chain)."""

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
        name="data-analyst",
        description="test",
        system_prompt="you are a test",
    )


def test_executor_stores_authorized_paths_and_name(SubagentExecutor_cls, SubagentConfig_cls):
    """Executor stores authorized_handoff_paths and config.name for
    HandoffIsolationProvider construction in _create_agent."""
    config = _minimal_config(SubagentConfig_cls)
    executor = SubagentExecutor_cls(
        config=config,
        tools=[],
        authorized_handoff_paths={"/mnt/user-data/workspace/handoff_code_executor.json"},
    )
    assert executor.authorized_handoff_paths == {
        "/mnt/user-data/workspace/handoff_code_executor.json"
    }
    assert executor.config.name == "data-analyst"


def test_handoff_isolation_provider_blocks_unauthorized_peer_read():
    """Provider unit test: unauthorized path → denied."""
    from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
    from deerflow.guardrails.provider import GuardrailRequest

    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    decision = p.evaluate(GuardrailRequest(
        tool_name="read_file",
        tool_input={"file_path": "/mnt/user-data/workspace/handoff_code_executor.json"},
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is False
    assert decision.reasons[0].code == "handoff_isolation.unauthorized"
