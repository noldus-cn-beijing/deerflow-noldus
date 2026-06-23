"""Spec 2026-06-23 ETHO-1 — SealGate after_agent fallback for reconstructable artifacts.

When a seal-requiring subagent (report-writer / chart-maker) terminates without
calling seal_<name>_handoff AND the handoff's core fields are mechanically
reconstructable from output files, the SealGate ``after_agent`` hook performs a
deterministic auto-seal at the termination point. This closes the L1 reminder-cap
release valve for reconstructable producers and eliminates the ``Task failed``
intermediate state the user would otherwise see (lead would otherwise retry a
whole round).

Cognitive producers (data-analyst) are deliberately NOT auto-sealed here:
``after_agent`` can neither ``jump_to`` (no ``can_jump_to``) nor fabricate an
interpretation. data-analyst seal-miss remains an observable degradation handled
by L1 (after_model nudge) + L2 (seal-resume) + L4 (lead retry).

Because conftest.py pre-mocks ``deerflow.subagents.executor`` AND the main-repo
venv's editable link resolves ``deerflow.*`` to the MAIN repo (not this worktree),
we load the worktree middleware module via importlib so these tests exercise the
worktree source (same pattern as test_auto_seal_from_artifacts / test_seal_resume).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

# ---------------------------------------------------------------------------
# Load the WORKTREE seal_gate_middleware module (editable link resolves to main
# repo otherwise — see module docstring).
# ---------------------------------------------------------------------------
_MIDDLEWARE_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "agents" / "middlewares" / "seal_gate_middleware.py"
)

_REAL_MW: ModuleType | None = None


def _get_mw() -> ModuleType:
    global _REAL_MW
    if _REAL_MW is not None:
        return _REAL_MW
    spec = importlib.util.spec_from_file_location(
        "deerflow.agents.middlewares.seal_gate_middleware_worktree",
        _MIDDLEWARE_FILE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _REAL_MW = module
    return module


# Likewise load the worktree executor module for _attempt_auto_seal_from_artifacts
# (the after_agent hook calls it; tests assert the handoff file appears).
_EXECUTOR_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "executor.py"
)


def _get_real_executor() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.executor_real_after_agent",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(autouse=True)
def _worktree_executor_in_sysmodules(monkeypatch):
    """Inject the WORKTREE executor module as ``deerflow.subagents.executor`` so the
    middleware's lazy ``from deerflow.subagents.executor import _attempt_auto_seal_from_artifacts``
    resolves to the worktree source (with the new ``sealed_by`` kwarg), not the
    main-repo editable install. Restored after each test.
    """
    real_exec = _get_real_executor()
    monkeypatch.setitem(sys.modules, "deerflow.subagents.executor", real_exec)
    return real_exec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_thread(tmp_path: Path):
    """Create thread dir structure: user-data/{workspace,outputs}.

    Returns (workspace_path_str, outputs_path).
    """
    user_data = tmp_path / "user-data"
    ws = user_data / "workspace"
    out = user_data / "outputs"
    ws.mkdir(parents=True)
    out.mkdir(parents=True)
    return str(ws), out


def _state_with_workspace(messages: list, workspace: str | None) -> dict:
    """Build a subagent state with messages + thread_data.workspace_path."""
    state: dict = {"messages": messages}
    if workspace is not None:
        state["thread_data"] = {"workspace_path": workspace}
    return state


def _runtime() -> MagicMock:
    rt = MagicMock()
    rt.run_id = "test-run"
    return rt


# ===================================================================
# 1. Core red→green: reconstructable miss → after_agent auto-seals
# ===================================================================

class TestAfterAgentAutoSealsReconstructable:
    """report-writer / chart-maker miss + artifacts present → after_agent auto-seals."""

    def test_report_writer_auto_seals_on_miss(self, tmp_path: Path, monkeypatch):
        """report-writer: outputs/report.md exists, no seal in history → after_agent
        calls _attempt_auto_seal_from_artifacts and the handoff file appears.

        Red before the after_agent hook exists (AttributeError / noop); green after.
        """
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# Results\n\nsome body\n", encoding="utf-8")

        gate = mw.SealGateMiddleware("report-writer")
        state = _state_with_workspace(
            messages=[AIMessage(content="报告已完成")],  # pure text, wants to exit, no seal
            workspace=ws,
        )

        assert hasattr(gate, "after_agent"), "SealGateMiddleware must expose after_agent"
        result = gate.after_agent(state, _runtime())

        # after_agent cannot jump; it returns None (side-effect only).
        assert result is None
        # The handoff file was deterministically created.
        handoff = Path(ws) / "handoff_report_writer.json"
        assert handoff.exists(), "after_agent should have auto-sealed from report.md"
        payload = json.loads(handoff.read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert payload["report_path"] == "/mnt/user-data/outputs/report.md"

    def test_chart_maker_auto_seals_on_miss(self, tmp_path: Path):
        """chart-maker: outputs/plot_*.png exist, no seal → after_agent auto-seals."""
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        (out / "plot_epm_open_arm.png").write_bytes(b"\x89PNG fake")
        # chart-maker schema requires paradigm; auto-seal reads it from
        # handoff_code_executor.json if present. Seed one so the payload validates.
        (Path(ws) / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm"}), encoding="utf-8",
        )

        gate = mw.SealGateMiddleware("chart-maker")
        state = _state_with_workspace(
            messages=[AIMessage(content="图表已生成")], workspace=ws,
        )
        result = gate.after_agent(state, _runtime())
        assert result is None

        handoff = Path(ws) / "handoff_chart_maker.json"
        assert handoff.exists()
        payload = json.loads(handoff.read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert any("plot_epm_open_arm.png" in p for p in payload["chart_files"])


# ===================================================================
# 2. Boundary: data-analyst (cognitive) must NOT be auto-sealed
# ===================================================================

class TestAfterAgentSkipsCognitiveProducers:
    """data-analyst is a cognitive producer — after_agent must NOT fabricate a seal."""

    def test_data_analyst_not_auto_sealed(self, tmp_path: Path):
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        gate = mw.SealGateMiddleware("data-analyst")
        state = _state_with_workspace(
            messages=[AIMessage(content="分析完成")], workspace=ws,
        )
        result = gate.after_agent(state, _runtime())
        # No jump, no auto-seal, no fabricated handoff.
        assert result is None
        assert not (Path(ws) / "handoff_data_analyst.json").exists()

    def test_code_executor_not_in_reconstructable_set(self):
        """code-executor is structurally excluded (run_metric_plan is produce-and-deliver)."""
        mw = _get_mw()
        assert "code-executor" not in mw._RECONSTRUCTABLE
        assert "data-analyst" not in mw._RECONSTRUCTABLE
        # Only the two mechanically-reconstructable producers.
        assert set(mw._RECONSTRUCTABLE) == {"report-writer", "chart-maker"}


# ===================================================================
# 3. Noop when already sealed
# ===================================================================

class TestAfterAgentNoopWhenAlreadySealed:
    def test_history_has_seal_tool_message_skips(self, tmp_path: Path):
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# Results\n", encoding="utf-8")
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "seal_report_writer_handoff", "args": {}, "id": "tc1"}],
        )
        tm = ToolMessage(
            content="ok", tool_call_id="tc1", name="seal_report_writer_handoff",
        )
        state = _state_with_workspace(
            messages=[ai, tm, AIMessage(content="done")], workspace=ws,
        )
        gate = mw.SealGateMiddleware("report-writer")
        result = gate.after_agent(state, _runtime())
        assert result is None
        # Should not have created a handoff (already sealed).
        assert not (Path(ws) / "handoff_report_writer.json").exists()


# ===================================================================
# 4. Fail-open: bad workspace / exception → no throw, returns None
# ===================================================================

class TestAfterAgentFailOpen:
    def test_workspace_none_returns_none(self, tmp_path: Path):
        mw = _get_mw()
        gate = mw.SealGateMiddleware("report-writer")
        state = _state_with_workspace(
            messages=[AIMessage(content="done")], workspace=None,
        )
        # Must not raise.
        assert gate.after_agent(state, _runtime()) is None

    def test_missing_thread_data_returns_none(self, tmp_path: Path):
        mw = _get_mw()
        gate = mw.SealGateMiddleware("report-writer")
        # state with no thread_data at all
        state = {"messages": [AIMessage(content="done")]}
        assert gate.after_agent(state, _runtime()) is None

    def test_auto_seal_exception_is_swallowed(self, tmp_path: Path, monkeypatch):
        """If _attempt_auto_seal_from_artifacts raises, after_agent must not propagate."""
        mw = _get_mw()

        # Force the lazy import inside after_agent to return a function that throws.
        import sys

        class _BoomExecutor:
            @staticmethod
            def _attempt_auto_seal_from_artifacts(name, workspace):
                raise RuntimeError("boom")

        # after_agent lazily imports deerflow.subagents.executor; inject our boom.
        monkeypatch.setitem(sys.modules, "deerflow.subagents.executor", _BoomExecutor)

        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# x\n", encoding="utf-8")
        gate = mw.SealGateMiddleware("report-writer")
        state = _state_with_workspace(
            messages=[AIMessage(content="done")], workspace=ws,
        )
        assert gate.after_agent(state, _runtime()) is None


# ===================================================================
# 5. Smoke: SealGate instantiates + both hooks run on synthetic state
# ===================================================================

class TestSmokeSealGate:
    def test_seal_gate_instantiates_and_runs(self, tmp_path: Path):
        mw = _get_mw()
        gate = mw.SealGateMiddleware("report-writer")
        ws, out = _mk_thread(tmp_path)
        state = _state_with_workspace(
            messages=[AIMessage(content="done")], workspace=ws,
        )
        rt = _runtime()
        # Both hooks callable, neither raises.
        assert gate.after_model(state, rt) is not None or True  # after_model may inject
        assert gate.after_agent(state, rt) is None


# ===================================================================
# 6. Observability: sealed_by recorded on the reconstructable handoffs
# ===================================================================

class TestSealedByRecorded:
    """after_agent auto-seal must stamp sealed_by='after_agent_artifacts' for
    trace richness (HarnessX) so fallback trigger rate is observable."""

    def test_report_writer_sealed_by_after_agent_artifacts(self, tmp_path: Path):
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# Results\n", encoding="utf-8")
        gate = mw.SealGateMiddleware("report-writer")
        state = _state_with_workspace(
            messages=[AIMessage(content="done")], workspace=ws,
        )
        gate.after_agent(state, _runtime())
        payload = json.loads(
            (Path(ws) / "handoff_report_writer.json").read_text(encoding="utf-8"),
        )
        assert payload.get("sealed_by") == "after_agent_artifacts"

    def test_chart_maker_sealed_by_after_agent_artifacts(self, tmp_path: Path):
        mw = _get_mw()
        ws, out = _mk_thread(tmp_path)
        (out / "plot_x.png").write_bytes(b"\x89PNG fake")
        (Path(ws) / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm"}), encoding="utf-8",
        )
        gate = mw.SealGateMiddleware("chart-maker")
        state = _state_with_workspace(
            messages=[AIMessage(content="done")], workspace=ws,
        )
        gate.after_agent(state, _runtime())
        payload = json.loads(
            (Path(ws) / "handoff_chart_maker.json").read_text(encoding="utf-8"),
        )
        assert payload.get("sealed_by") == "after_agent_artifacts"


# ===================================================================
# 7. Scope guard: _MAX_REMINDERS unchanged (spec §2.4)
# ===================================================================

class TestScopeGuard:
    def test_max_reminders_unchanged(self):
        mw = _get_mw()
        assert mw._MAX_REMINDERS == 2, (
            "spec §2.4: cap must stay 2; raising it burns more jumps on "
            "thinking-overload turns (handled by a separate spec line)."
        )

    def test_after_agent_not_can_jump_to(self):
        """after_agent must NOT declare can_jump_to — it runs after the agent has
        already decided to terminate and physically cannot jump (spec §1.4)."""
        mw = _get_mw()
        # The hook_config metadata, if any, must not advertise can_jump_to.
        hook = getattr(mw.SealGateMiddleware.after_agent, "hook_config", None)
        if hook:
            assert not hook.get("can_jump_to"), (
                "after_agent must be side-effect only; can_jump_to is forbidden "
                "(physical boundary — after_agent runs post-termination)."
            )
