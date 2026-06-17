"""Spec 2026-06-17 — column_semantics 独立写入通道 + guardrail 模板锁泛化 + Gate1 不 clobber。

复刻 agent 真实「分步调用」序列（Gate1 → 单独 column_semantics → 单独 resolved_facts），
不许把多步合一步（现有 test_column_semantics.py 把多步合一恰好掩盖 Bug2 = 假绿）。

⚠️ worktree 共享主仓 venv 的 editable 链固定指向**主仓** harness 源码（不是 worktree），
普通 ``import deerflow`` 会加载主仓旧代码 → 假绿。本文件**全部用 importlib 显式加载 worktree
源码**（守 memory feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Load the REAL worktree source for experiment_context + ev19_template_provider.
# Each module loaded under a unique synthetic name so it does not collide with
# the editable-installed (main-repo) ``deerflow`` package in sys.modules.
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_ROOT = _BACKEND_ROOT / "packages" / "harness"

_EXPERIMENT_CONTEXT_FILE = _HARNESS_ROOT / "deerflow" / "agents" / "middlewares" / "experiment_context.py"
_GUARDRAIL_PROVIDER_FILE = _HARNESS_ROOT / "deerflow" / "guardrails" / "ev19_template_provider.py"
_PROVIDER_FILE = _HARNESS_ROOT / "deerflow" / "guardrails" / "provider.py"


def _load_module(name: str, path: Path) -> ModuleType:
    """importlib-load a single module file from the worktree, isolated under ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not find {path}"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # so its own ``from deerflow...`` re-exports resolve within
    spec.loader.exec_module(module)
    return module


_LOADED: dict[str, ModuleType] = {}


def _experiment_context_module() -> ModuleType:
    if "experiment_context" not in _LOADED:
        _LOADED["experiment_context"] = _load_module(
            "tcs_experiment_context_isolated",
            _EXPERIMENT_CONTEXT_FILE,
        )
    return _LOADED["experiment_context"]


def _guardrail_provider_module() -> tuple[ModuleType, ModuleType]:
    """Load provider.py first (defines GuardrailRequest/GuardrailDecision), then the provider."""
    if "provider" not in _LOADED:
        _LOADED["provider"] = _load_module("tcs_provider_isolated", _PROVIDER_FILE)
    if "ev19_provider" not in _LOADED:
        _LOADED["ev19_provider"] = _load_module(
            "tcs_ev19_template_provider_isolated",
            _GUARDRAIL_PROVIDER_FILE,
        )
    return _LOADED["ev19_provider"], _LOADED["provider"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime_with_workspace(workspace: Path) -> MagicMock:
    """Build a MagicMock runtime whose .state.thread_data.workspace_path = workspace.

    ``runtime.context`` stays a MagicMock (not a dict) → _thread_id_from_runtime returns
    None → memory projection is skipped (no storage needed in unit tests).
    """
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": str(workspace)}}
    return runtime


def _gate1_full(module, workspace: Path, runtime: MagicMock) -> None:
    """Drive a full Gate 1 confirmation (all paradigm fields, no column_semantics)."""
    module.set_experiment_paradigm_tool.func(
        paradigm="epm",
        paradigm_cn="高架十字迷宫",
        category="anxiety",
        subject="rodent",
        ev19_template="PlusMaze-FewZones",
        workspace_dir=str(workspace),
        runtime=runtime,
    )


# ===========================================================================
# Bug 2 — independent column_semantics write channel (core blocker)
# ===========================================================================


class TestStandaloneColumnSemanticsChannel:
    """set_experiment_paradigm(column_semantics={...}) with NO paradigm fields must
    write column_semantics/column_aliases while preserving the Gate 1 state."""

    def test_standalone_column_semantics_after_gate1_writes(self, tmp_path):
        """Gate1 → then column_semantics-only call writes columns and preserves paradigm."""
        ec = _experiment_context_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _runtime_with_workspace(ws)

        # Step 1: Gate 1 (full fields, no column_semantics)
        _gate1_full(ec, ws, runtime)

        # Step 2: column_semantics ONLY (no paradigm fields) — the real agent sequence.
        cs = {
            "columns": {
                "open": {"raw_name": "open", "resolves_to": "open_arms", "meaning_zh": "开放臂", "confirmed": True},
                "closed": {"raw_name": "closed", "resolves_to": "closed_arms", "meaning_zh": "闭合臂", "confirmed": True},
            },
        }
        result_raw = ec.set_experiment_paradigm_tool.func(
            column_semantics=cs,
            workspace_dir=str(ws),
            runtime=runtime,
        )
        result = json.loads(result_raw)

        # Must succeed (current code returns "Missing required fields for Gate 1" → red).
        assert result["status"] == "ok", f"expected ok, got: {result}"
        assert result.get("column_semantics_saved") is True

        ctx = ec.read_context(str(ws))
        assert ctx is not None
        # column_semantics + aliases written
        assert "column_semantics" in ctx
        assert ctx["column_aliases"]["open"] == "open_arms"
        assert ctx["column_aliases"]["closed"] == "closed_arms"
        # Gate 1 state preserved (NOT clobbered)
        assert ctx["paradigm"] == "epm"
        assert ctx["ev19_template"] == "PlusMaze-FewZones"

    def test_standalone_column_semantics_without_existing_errors(self, tmp_path):
        """column_semantics-only call with NO existing context must return a clear error."""
        ec = _experiment_context_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _runtime_with_workspace(ws)

        cs = {"columns": {"open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True}}}
        result_raw = ec.set_experiment_paradigm_tool.func(
            column_semantics=cs,
            workspace_dir=str(ws),
            runtime=runtime,
        )
        result = json.loads(result_raw)
        assert result["status"] == "error"
        # Nothing written to disk
        assert not (ws / "experiment-context.json").exists()


# ===========================================================================
# Bug 1 — guardrail template lock generalized to pass-through when no template field
# ===========================================================================


def _provider_with_workspace(ws: Path):
    ev19_mod, _provider_mod = _guardrail_provider_module()
    return ev19_mod.Ev19TemplateGuardrailProvider(workspace_resolver=lambda: str(ws))


def _make_request(ev19_mod, provider_mod, tool_name: str, args: dict):
    return provider_mod.GuardrailRequest(
        tool_name=tool_name,
        tool_input=args,
        agent_id=None,
        timestamp="2026-06-17T00:00:00Z",
    )


class TestGuardrailTemplateLockGeneralized:
    def _ws_with_template(self, tmp_path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        ctx = {"paradigm": "epm", "ev19_template": "PlusMaze-FewZones", "gate_completed": ["gate1_paradigm"]}
        (ws / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
        return ws

    def test_allows_column_semantics_only_when_template_set(self, tmp_path):
        """column_semantics-only call (no confirm_template_change) must pass the lock."""
        ev19_mod, provider_mod = _guardrail_provider_module()
        ws = self._ws_with_template(tmp_path)
        p = _provider_with_workspace(ws)
        req = _make_request(
            ev19_mod,
            provider_mod,
            "set_experiment_paradigm",
            {"column_semantics": {"columns": {"open": {"resolves_to": "open_arms"}}}},
        )
        decision = p.evaluate(req)
        assert decision.allow is True
        assert not any(r.code == "ethoinsight.template_already_set" for r in decision.reasons)

    def test_allows_resolved_facts_only_when_template_set(self, tmp_path):
        """resolved_facts-only call must also pass the lock (generalization co-benefit)."""
        ev19_mod, provider_mod = _guardrail_provider_module()
        ws = self._ws_with_template(tmp_path)
        p = _provider_with_workspace(ws)
        req = _make_request(
            ev19_mod,
            provider_mod,
            "set_experiment_paradigm",
            {"resolved_facts": [{"key": "groups", "value": "x=control"}]},
        )
        decision = p.evaluate(req)
        assert decision.allow is True
        assert not any(r.code == "ethoinsight.template_already_set" for r in decision.reasons)

    def test_still_blocks_real_template_change(self, tmp_path):
        """Passing a DIFFERENT ev19_template without confirm_template_change still blocked."""
        ev19_mod, provider_mod = _guardrail_provider_module()
        ws = self._ws_with_template(tmp_path)
        p = _provider_with_workspace(ws)
        req = _make_request(
            ev19_mod,
            provider_mod,
            "set_experiment_paradigm",
            {"ev19_template": "PlusMaze-AllZones"},
        )
        decision = p.evaluate(req)
        assert decision.allow is False
        assert any(r.code == "ethoinsight.template_already_set" for r in decision.reasons)

    def test_ambiguous_check_still_fires_when_template_passed(self, tmp_path):
        """Ambiguous template check must NOT be bypassed by the generalization."""
        ev19_mod, provider_mod = _guardrail_provider_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        ctx = {"paradigm": "epm", "ev19_template": "PlusMaze-FewZones", "gate_completed": ["gate1_paradigm"]}
        (ws / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
        # ambiguous candidates (file name matches _read_template_candidates)
        cand = {
            "status": "ambiguous",
            "candidates": [
                {"template_id": "PlusMaze-FewZones"},
                {"template_id": "PlusMaze-AllZones"},
            ],
        }
        (ws / "template_candidates.json").write_text(json.dumps(cand), encoding="utf-8")

        p = _provider_with_workspace(ws)
        # Passing a candidate ev19_template without user_confirmed_template → still blocked.
        req = _make_request(
            ev19_mod,
            provider_mod,
            "set_experiment_paradigm",
            {"ev19_template": "PlusMaze-FewZones", "confirm_template_change": True},
        )
        decision = p.evaluate(req)
        assert decision.allow is False
        assert any(r.code == "ethoinsight.template_not_confirmed" for r in decision.reasons)


# ===========================================================================
# Bug 3 — Gate 1 rerun inherits existing, does not clobber
# ===========================================================================


class TestGate1RerunPreservesState:
    def test_gate1_rerun_preserves_resolved_facts(self, tmp_path):
        """Gate1 → standalone resolved_facts (groups) → full Gate1 rerun keeps resolved."""
        ec = _experiment_context_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _runtime_with_workspace(ws)

        _gate1_full(ec, ws, runtime)

        # Standalone resolved_facts → write groups
        ec.set_experiment_paradigm_tool.func(
            resolved_facts=[{"key": "groups", "value": "control=7,treatment=21"}],
            workspace_dir=str(ws),
            runtime=runtime,
        )

        # Full Gate 1 rerun (no resolved_facts) — current code clobbers resolved.
        ec.set_experiment_paradigm_tool.func(
            paradigm="epm",
            paradigm_cn="高架十字迷宫",
            category="anxiety",
            subject="rodent",
            ev19_template="PlusMaze-FewZones",
            workspace_dir=str(ws),
            runtime=runtime,
        )

        ctx = ec.read_context(str(ws))
        assert ctx is not None
        assert ctx.get("resolved", {}).get("groups") == "control=7,treatment=21"

    def test_gate1_rerun_preserves_column_semantics(self, tmp_path):
        """Gate1 → standalone column_semantics → full Gate1 rerun keeps column_semantics/aliases."""
        ec = _experiment_context_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _runtime_with_workspace(ws)

        _gate1_full(ec, ws, runtime)

        cs = {"columns": {"open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True}}}
        ec.set_experiment_paradigm_tool.func(
            column_semantics=cs,
            workspace_dir=str(ws),
            runtime=runtime,
        )

        # Full Gate 1 rerun (no column_semantics)
        ec.set_experiment_paradigm_tool.func(
            paradigm="epm",
            paradigm_cn="高架十字迷宫",
            category="anxiety",
            subject="rodent",
            ev19_template="PlusMaze-FewZones",
            workspace_dir=str(ws),
            runtime=runtime,
        )

        ctx = ec.read_context(str(ws))
        assert ctx is not None
        assert "column_semantics" in ctx
        assert ctx["column_aliases"]["open"] == "open_arms"

    def test_column_semantics_plus_resolved_facts_same_call(self, tmp_path):
        """Plan A boundary: column_semantics + resolved_facts in ONE call (no paradigm)."""
        ec = _experiment_context_module()
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _runtime_with_workspace(ws)

        _gate1_full(ec, ws, runtime)

        cs = {"columns": {"open": {"raw_name": "open", "resolves_to": "open_arms", "confirmed": True}}}
        result_raw = ec.set_experiment_paradigm_tool.func(
            column_semantics=cs,
            resolved_facts=[{"key": "groups", "value": "control=7,treatment=21"}],
            workspace_dir=str(ws),
            runtime=runtime,
        )
        result = json.loads(result_raw)
        assert result["status"] == "ok"
        assert result.get("column_semantics_saved") is True
        assert result.get("resolved_facts_saved") == 1

        ctx = ec.read_context(str(ws))
        assert ctx is not None
        assert ctx["column_aliases"]["open"] == "open_arms"
        assert ctx["resolved"]["groups"] == "control=7,treatment=21"
