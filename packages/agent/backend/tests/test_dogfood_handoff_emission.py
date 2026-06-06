"""Sprint 5.7 — Dogfood simulation test for handoff emission validation.

Simulates the exact failure scenario from 2026-05-28 dogfood thread 68f6da40...
row 14: LLM outputs final AIMessage with content like "现在封存分析结果："
but no tool_call → executor previously marked COMPLETED → handoff file missing
→ downstream broke.

After Sprint 5.7, the executor should mark this as FAILED with the diagnostic
string containing "terminated without emitting".
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


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
        "deerflow.subagents.executor_real_dogfood",
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


class TestSimulatedLLMForgetsSealTool:
    """Simulate the real dogfood failure: LLM finishes without calling seal tool."""

    def test_simulated_llm_forgets_seal_tool(self, tmp_path: Path):
        """Reproduces 2026-05-28 dogfood failure: LLM says '封存' but no tool_call.

        Scenario:
        1. data-analyst's LLM produces a final AIMessage with content
           "现在封存分析结果：" — no tool_call for seal_data_analyst_handoff.
        2. Executor previously would mark COMPLETED (no new tool_calls → done).
        3. After Sprint 5.7: executor checks for handoff_data_analyst.json
           in workspace → missing → marks FAILED with diagnostic.
        4. The diagnostic contains "terminated without emitting" which the
           lead's retry rule can match on.
        """
        mod = _get_real_executor()

        # Create an empty workspace (no handoff file produced by LLM)
        ws = tmp_path / "workspace"
        ws.mkdir()

        # Simulate executor's validate-first COMPLETED path
        result = mod.SubagentResult(
            task_id="dogfood-task",
            trace_id="68f6da40",
            status=mod.SubagentStatus.PENDING,
        )

        _handoff_error = mod._validate_handoff_emitted("data-analyst", str(ws))
        if _handoff_error is not None:
            result.try_set_terminal(mod.SubagentStatus.FAILED, error=_handoff_error)
        else:
            result.try_set_terminal(mod.SubagentStatus.COMPLETED)

        # Assertions:
        # 1. Task should be FAILED (not silently COMPLETED)
        assert result.status == mod.SubagentStatus.FAILED, (
            "Sprint 5.7 regression: LLM forgot seal tool but executor marked COMPLETED"
        )

        # 2. Error message must contain the keyword the lead retry rule matches on
        assert result.error is not None
        assert "terminated without emitting" in result.error, (
            "Diagnostic string missing 'terminated without emitting' keyword"
        )

        # 3. Error should mention the specific file and seal tool
        assert "handoff_data_analyst.json" in result.error
        assert "seal_data_analyst_handoff" in result.error

        # 4. Error should contain the instruction for lead to retry
        assert "Lead should re-dispatch" in result.error
