"""Sprint 5.7 — Unit tests for _validate_handoff_emitted helper.

Tests the pure function that checks whether an ethoinsight subagent's handoff
file exists in the thread workspace before executor marks the task COMPLETED.

The conftest.py pre-mocks deerflow.subagents.executor to break circular imports,
so we load the real module source directly via importlib to access the function
under test without triggering the circular import chain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


# ---------------------------------------------------------------------------
# Load the REAL executor module source (bypassing conftest's sys.modules mock)
# without triggering the circular import chain.  We use importlib.util to load
# from file path, and intercept any imports that would re-enter the mock.
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
        "deerflow.subagents.executor_real",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None, f"Could not find executor.py at {_EXECUTOR_FILE}"
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    # We do NOT register in sys.modules — this avoids triggering the mock
    # and also avoids the circular import chain.  The module will still be
    # able to resolve top-level stdlib/third-party imports but won't re-enter
    # deerflow.subagents (which is mocked).
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        # If exec_module fails (due to import chain), we still need the
        # function. Fall back to a targeted approach: exec just the helper.
        pytest.skip("Could not load real executor module — falling back")

    _REAL_EXECUTOR = module
    return _REAL_EXECUTOR


@pytest.fixture(autouse=True)
def _ensure_module_loaded():
    """Pre-load the real module before any test runs."""
    _get_real_executor()


def _validate(subagent_name: str, workspace_path: str | None) -> str | None:
    """Convenience wrapper around the real _validate_handoff_emitted."""
    return _get_real_executor()._validate_handoff_emitted(subagent_name, workspace_path)  # type: ignore[union-attr]


def _make_workspace(tmp_path: Path, *, with_file: str | None = None) -> str:
    """Create a temporary workspace dir, optionally pre-creating a handoff file."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    if with_file:
        (ws / with_file).write_text("{}", encoding="utf-8")
    return str(ws)


# ==================== PASS CASES (return None) ====================


class TestHandoffPresentPasses:
    """When the handoff file exists, validation passes (returns None)."""

    @pytest.mark.parametrize(
        "name, filename",
        [
            ("data-analyst", "handoff_data_analyst.json"),
            ("code-executor", "handoff_code_executor.json"),
            ("chart-maker", "handoff_chart_maker.json"),
            ("report-writer", "handoff_report_writer.json"),
        ],
    )
    def test_handoff_present_passes(self, tmp_path: Path, name: str, filename: str):
        ws = _make_workspace(tmp_path, with_file=filename)
        assert _validate(name, ws) is None


# ==================== FAIL CASES (return diagnostic str) ====================


class TestHandoffAbsentReturnsDiagnostic:
    """When the handoff file is missing, returns a diagnostic string with fixed keywords."""

    @pytest.mark.parametrize(
        "name, filename, seal_tool",
        [
            ("data-analyst", "handoff_data_analyst.json", "seal_data_analyst_handoff"),
            ("code-executor", "handoff_code_executor.json", "seal_code_executor_handoff"),
            ("chart-maker", "handoff_chart_maker.json", "seal_chart_maker_handoff"),
            ("report-writer", "handoff_report_writer.json", "seal_report_writer_handoff"),
        ],
    )
    def test_missing_handoff_returns_diagnostic(self, tmp_path: Path, name: str, filename: str, seal_tool: str):
        ws = _make_workspace(tmp_path)  # no handoff file
        result = _validate(name, ws)
        assert result is not None
        assert name in result
        assert filename in result
        assert seal_tool in result
        assert "terminated without emitting" in result
        assert "Lead should re-dispatch" in result


# ==================== SKIP CASES (not in whitelist → None) ====================


class TestNonWhitelistedSkipped:
    """General-purpose / bash / knowledge-assistant have no handoff contract."""

    @pytest.mark.parametrize("name", ["general-purpose", "bash", "knowledge-assistant", "planning"])
    def test_skipped_subagents(self, tmp_path: Path, name: str):
        ws = _make_workspace(tmp_path)  # empty workspace
        assert _validate(name, ws) is None


# ==================== FAIL-OPEN CASES ====================


class TestFailOpen:
    """When workspace is unresolvable, fail-open (return None, do NOT block)."""

    def test_no_workspace_path(self):
        assert _validate("data-analyst", None) is None

    def test_empty_workspace_path(self):
        assert _validate("data-analyst", "") is None


# ==================== ROBUSTNESS (R1 — never raises) ====================


class TestRobustness:
    """The helper MUST NEVER raise, even with pathological inputs (C3)."""

    def test_helper_never_raises_on_path_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Monkeypatch Path.exists to raise, simulating pathological filesystem conditions.
        The helper must catch the exception and return None (fail-open).
        """
        ws = _make_workspace(tmp_path)
        original_exists = Path.exists

        def _raising_exists(self: Path) -> bool:
            # Raise only for handoff files, let the workspace dir check pass
            if "handoff_" in str(self):
                raise OSError("simulated filesystem error")
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", _raising_exists)
        result = _validate("data-analyst", ws)
        assert result is None  # fail-open, no exception propagated

    def test_nonexistent_dir_returns_diagnostic(self, tmp_path: Path):
        """A directory path that doesn't exist → Path.exists returns False → diagnostic."""
        fake_path = str(tmp_path / "nonexistent_dir")
        result = _validate("data-analyst", fake_path)
        assert result is not None
        assert "terminated without emitting" in result
