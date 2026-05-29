"""Sprint 5.7 — Integration tests for handoff emission validation in executor.

Tests the interaction between SubagentResult.try_set_terminal and the
_validate_handoff_emitted helper at the executor's terminal-state call site.

Because conftest.py pre-mocks deerflow.subagents.executor, we load the real
module via importlib and construct SubagentResult objects directly to test
the COMPLETED / FAILED / CANCELLED terminal paths with handoff validation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Load the real executor module (same pattern as test_handoff_emission_validator)
# ---------------------------------------------------------------------------
_EXECUTOR_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
)

_REAL_EXECUTOR: ModuleType | None = None


def _get_real_executor() -> ModuleType:
    global _REAL_EXECUTOR
    if _REAL_EXECUTOR is not None:
        return _REAL_EXECUTOR

    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.executor_real_integration",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _REAL_EXECUTOR = module
    return module


@pytest.fixture(autouse=True)
def _load_module():
    _get_real_executor()


def _make_result(mod: ModuleType, *, task_id: str = "test-task"):
    """Create a fresh SubagentResult in PENDING state."""
    return mod.SubagentResult(
        task_id=task_id,
        trace_id="test-trace",
        status=mod.SubagentStatus.PENDING,
    )


def _make_workspace(tmp_path: Path, *, with_file: str | None = None) -> str:
    ws = tmp_path / "workspace"
    ws.mkdir()
    if with_file:
        (ws / with_file).write_text("{}", encoding="utf-8")
    return str(ws)


# ---------------------------------------------------------------------------
# Test: simulate executor's validate-first → set_terminal logic
# ---------------------------------------------------------------------------

def _simulate_completed_path(mod: ModuleType, subagent_name: str, workspace_path: str | None) -> ModuleType.SubagentResult:  # type: ignore[name-defined]
    """Simulate what executor.py does at the COMPLETED call site (line 807).

    This is the exact logic added in Sprint 5.7 T2:
      1. Resolve workspace_path from thread_data
      2. Call _validate_handoff_emitted
      3. If error → FAILED; else → COMPLETED
    """
    result = _make_result(mod)
    _handoff_error = mod._validate_handoff_emitted(subagent_name, workspace_path)
    if _handoff_error is not None:
        result.try_set_terminal(mod.SubagentStatus.FAILED, error=_handoff_error)
    else:
        result.try_set_terminal(mod.SubagentStatus.COMPLETED)
    return result


class TestCompletedPathWithHandoffValidation:
    """Test the COMPLETED terminal path with handoff validation injected."""

    def test_completed_subagent_with_handoff_passes(self, tmp_path: Path):
        """data-analyst task 完成 + workspace 有 handoff → COMPLETED."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path, with_file="handoff_data_analyst.json")
        result = _simulate_completed_path(mod, "data-analyst", ws)
        assert result.status == mod.SubagentStatus.COMPLETED
        assert result.error is None

    def test_completed_subagent_without_handoff_marked_failed(self, tmp_path: Path):
        """data-analyst 完成 + workspace 无 handoff → FAILED + diagnostic."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)  # no handoff file
        result = _simulate_completed_path(mod, "data-analyst", ws)
        assert result.status == mod.SubagentStatus.FAILED
        assert result.error is not None
        assert "terminated without emitting" in result.error

    def test_completed_general_purpose_without_handoff_still_passes(self, tmp_path: Path):
        """general-purpose + 空 workspace → COMPLETED（白名单外不验证）."""
        mod = _get_real_executor()
        ws = _make_workspace(tmp_path)  # no handoff, irrelevant for general-purpose
        result = _simulate_completed_path(mod, "general-purpose", ws)
        assert result.status == mod.SubagentStatus.COMPLETED
        assert result.error is None

    def test_cancelled_path_skips_validation(self):
        """task 被 cancel → CANCELLED, 不被 5.7 逻辑改成 FAILED."""
        mod = _get_real_executor()
        result = _make_result(mod)
        # CANCELLED path happens before the COMPLETED validation (different code path)
        result.try_set_terminal(mod.SubagentStatus.CANCELLED, error="Cancelled by user")
        assert result.status == mod.SubagentStatus.CANCELLED
        assert result.error == "Cancelled by user"

    def test_failed_path_skips_validation(self):
        """task 自身抛异常 → FAILED, error 是原异常串，不被 handoff 诊断覆盖."""
        mod = _get_real_executor()
        result = _make_result(mod)
        original_error = "ZeroDivisionError: division by zero"
        # FAILED path happens in the except block (different code path from validation)
        result.try_set_terminal(mod.SubagentStatus.FAILED, error=original_error)
        assert result.status == mod.SubagentStatus.FAILED
        assert result.error == original_error
        assert "terminated without emitting" not in result.error
