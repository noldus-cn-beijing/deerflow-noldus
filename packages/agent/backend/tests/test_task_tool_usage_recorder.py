"""Regression tests for `_find_usage_recorder` callback normalization.

Background: LangGraph runtime's ``config["callbacks"]`` is NOT guaranteed to be
a ``list``. In async execution paths, LangChain often passes an
``AsyncCallbackManager`` instance (subclass of ``BaseCallbackManager``), which
has no ``__iter__`` method. Earlier code did ``for cb in callbacks:`` which
raised ``TypeError: 'AsyncCallbackManager' object is not iterable``, breaking
the entire subagent token-usage reporting path and (transitively) causing the
task tool to return an error to the lead agent even after the subagent
completed successfully.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    BaseCallbackManager,
    CallbackManager,
)

task_tool_module = importlib.import_module("deerflow.tools.builtins.task_tool")
_find_usage_recorder = task_tool_module._find_usage_recorder


class _Recorder:
    """Stand-in for a real journal — has the duck-typed attribute we look for."""

    def record_external_llm_usage_records(self, records: list) -> None:  # pragma: no cover - method existence is what matters
        return None


class _NonRecorder:
    """Callback handler without the recorder attribute."""


def _make_runtime(callbacks):
    return SimpleNamespace(config={"callbacks": callbacks})


def test_runtime_none_returns_none():
    assert _find_usage_recorder(None) is None


def test_runtime_without_dict_config_returns_none():
    runtime = SimpleNamespace(config=None)
    assert _find_usage_recorder(runtime) is None


def test_callbacks_missing_returns_none():
    runtime = SimpleNamespace(config={})
    assert _find_usage_recorder(runtime) is None


def test_callbacks_empty_list_returns_none():
    runtime = _make_runtime([])
    assert _find_usage_recorder(runtime) is None


def test_callbacks_list_with_recorder_returns_recorder():
    recorder = _Recorder()
    runtime = _make_runtime([_NonRecorder(), recorder])
    assert _find_usage_recorder(runtime) is recorder


def test_callbacks_list_without_recorder_returns_none():
    runtime = _make_runtime([_NonRecorder(), _NonRecorder()])
    assert _find_usage_recorder(runtime) is None


def test_async_callback_manager_with_recorder_in_handlers():
    """Regression: AsyncCallbackManager is not iterable but has .handlers list.

    Reproduces the production failure path: LangGraph runtime hands us an
    AsyncCallbackManager (no __iter__). _find_usage_recorder must look inside
    its `.handlers` list rather than blindly iterating the manager.
    """
    recorder = _Recorder()
    manager = AsyncCallbackManager(handlers=[_NonRecorder(), recorder])
    assert isinstance(manager, BaseCallbackManager)
    assert not hasattr(manager, "__iter__")

    runtime = _make_runtime(manager)
    assert _find_usage_recorder(runtime) is recorder


def test_sync_callback_manager_with_recorder_in_handlers():
    """Same protection for the sync CallbackManager variant."""
    recorder = _Recorder()
    manager = CallbackManager(handlers=[recorder])
    runtime = _make_runtime(manager)
    assert _find_usage_recorder(runtime) is recorder


def test_async_callback_manager_without_recorder_returns_none():
    manager = AsyncCallbackManager(handlers=[_NonRecorder()])
    runtime = _make_runtime(manager)
    assert _find_usage_recorder(runtime) is None


def test_async_callback_manager_with_empty_handlers_returns_none():
    manager = AsyncCallbackManager(handlers=[])
    runtime = _make_runtime(manager)
    assert _find_usage_recorder(runtime) is None
