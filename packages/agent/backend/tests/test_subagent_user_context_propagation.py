"""Regression test: ContextVar (user_id) must propagate from lead agent to subagent.

This test guards against the bug where subagent execution lost user context
because contextvars do not propagate across ThreadPoolExecutor / new event
loop boundaries unless explicitly carried via copy_context(). See
docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# user_context has no circular import issues — safe to import directly
from deerflow.runtime.user_context import (
    DEFAULT_USER_ID,
    get_effective_user_id,
    reset_current_user,
    set_current_user,
)

# Modules that cause circular imports when executor is loaded
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


@pytest.fixture(scope="session")
def _executor():
    """Import the real executor module, working around conftest.py circular-import mocks."""
    original_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULE_NAMES}
    original_executor = sys.modules.get("deerflow.subagents.executor")

    if "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]
    for name in _MOCKED_MODULE_NAMES:
        sys.modules[name] = MagicMock()

    import deerflow.subagents.executor as _mod

    yield _mod

    # Restore
    for name in _MOCKED_MODULE_NAMES:
        if original_modules[name] is not None:
            sys.modules[name] = original_modules[name]
        elif name in sys.modules:
            del sys.modules[name]
    if original_executor is not None:
        sys.modules["deerflow.subagents.executor"] = original_executor
    elif "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]


@dataclass
class _StubUser:
    """Minimal user object that satisfies the CurrentUser protocol."""

    id: str


def test_get_effective_user_id_returns_default_when_unset():
    """Sanity: with no user set, fallback is DEFAULT_USER_ID."""
    # Reset any leftover user context from other tests in the session
    from deerflow.runtime.user_context import _current_user

    _current_user.set(None)
    assert get_effective_user_id() == DEFAULT_USER_ID


def test_get_effective_user_id_returns_set_value():
    user = _StubUser(id="alice-uuid")
    token = set_current_user(user)
    try:
        assert get_effective_user_id() == "alice-uuid"
    finally:
        reset_current_user(token)


def test_contextvar_does_not_propagate_across_naive_thread_pool():
    """Verifies the underlying Python behaviour our fix compensates for.

    This is the *bug* condition: a ThreadPoolExecutor.submit without
    contextvars.copy_context() loses the parent's ContextVar state.
    """
    from concurrent.futures import ThreadPoolExecutor

    user = _StubUser(id="bob-uuid")
    token = set_current_user(user)
    try:
        captured: list[str] = []

        def child_thread():
            captured.append(get_effective_user_id())

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(child_thread).result(timeout=5)

        # Without copy_context(), child thread sees default, not "bob-uuid".
        assert captured == [DEFAULT_USER_ID], "If this test starts failing, Python's ThreadPoolExecutor began propagating contextvars by default; review whether the _submit_to_isolated_loop_in_context wrapper is still needed."
    finally:
        reset_current_user(token)


def test_submit_to_isolated_loop_preserves_user_context(_executor):
    """The fix: _submit_to_isolated_loop_in_context must carry ContextVar.

    This is the regression test that would have caught the original bug.
    """
    _get_isolated_subagent_loop = _executor._get_isolated_subagent_loop
    _submit_to_isolated_loop_in_context = _executor._submit_to_isolated_loop_in_context

    user = _StubUser(id="carol-uuid")
    token = set_current_user(user)
    try:
        # Force the persistent isolated loop to start.
        _get_isolated_subagent_loop()

        async def read_user_in_isolated_loop():
            return get_effective_user_id()

        from contextvars import copy_context

        parent_context = copy_context()
        future = _submit_to_isolated_loop_in_context(
            parent_context,
            lambda: read_user_in_isolated_loop(),
        )
        seen_user_id = future.result(timeout=10)

        assert seen_user_id == "carol-uuid", (
            f"ContextVar should propagate from parent task to isolated "
            f"loop coroutine, but got {seen_user_id!r} (DEFAULT_USER_ID = "
            f"{DEFAULT_USER_ID!r}). This means the fix in "
            f"docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md "
            f"has regressed."
        )
    finally:
        reset_current_user(token)


def test_isolated_loop_thread_is_daemon(_executor):
    """The persistent loop must run on a daemon thread.

    Otherwise, atexit cleanup or test teardown can hang waiting for the loop.
    """
    _get_isolated_subagent_loop = _executor._get_isolated_subagent_loop
    _get_isolated_subagent_loop()  # ensure it's started
    target_threads = [t for t in threading.enumerate() if t.name == "subagent-persistent-loop"]
    assert len(target_threads) >= 1, f"Expected exactly one subagent-persistent-loop thread, found {len(target_threads)}: {[t.name for t in threading.enumerate()]}"
    assert target_threads[0].daemon is True, "subagent-persistent-loop must be a daemon thread to allow clean shutdown"


@pytest.mark.asyncio
async def test_subagent_executor_propagates_user_context_in_running_loop(_executor):
    """End-to-end: an async parent invoking SubagentExecutor.execute() must
    carry user_id into the subagent's coroutine.

    This simulates the production path where lead_agent (running inside
    LangGraph's bg-loop) invokes the task tool, which invokes
    SubagentExecutor.execute_async() which uses _scheduler_pool.

    For this test we don't spin up a real LLM. We instead instantiate a
    SubagentExecutor with a stub _aexecute that captures user_id.
    """
    SubagentExecutor = _executor.SubagentExecutor
    SubagentResult = _executor.SubagentResult
    SubagentStatus = _executor.SubagentStatus

    from deerflow.subagents.config import SubagentConfig

    captured_user_id: list[str] = []

    user = _StubUser(id="dave-uuid")
    token = set_current_user(user)
    try:
        config = SubagentConfig(
            name="stub",
            description="stub for ContextVar propagation test",
            system_prompt="(unused)",
            max_turns=1,
            timeout_seconds=10,
        )
        executor = SubagentExecutor(config=config, tools=[])

        async def stub_aexecute(task, result_holder=None):
            captured_user_id.append(get_effective_user_id())
            return SubagentResult(
                task_id="stub",
                trace_id="stub",
                status=SubagentStatus.COMPLETED,
                result="stub-result",
            )

        # Monkey-patch _aexecute to skip LLM/agent creation
        executor._aexecute = stub_aexecute  # type: ignore[method-assign]

        # Call the sync execute() from inside a running event loop.
        # This forces the "isolated loop" path that was broken.
        result = await asyncio.to_thread(executor.execute, "stub-task")

        assert result.status == SubagentStatus.COMPLETED
        assert captured_user_id == ["dave-uuid"], f"SubagentExecutor.execute must carry the parent task's user_id into _aexecute, but got {captured_user_id!r}. This is the exact bug class that broke shoaling pipeline e2e on 2026-05-08."
    finally:
        reset_current_user(token)
