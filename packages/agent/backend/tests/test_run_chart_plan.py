"""Spec 2026-06-24-chart-maker-run-chart-plan — Tests for run_chart_plan first-party tool.

Covers the T1–T14 matrix from spec §五. Strategy mirrors test_run_metric_plan.py:
- Inject a synchronous ``_TASK_RUNNER_OVERRIDE`` so tests never touch
  ProcessPoolExecutor's fork/pickle semantics. The runner simulates what a real
  plot script does: write a png to the --output path (or not), return
  (task_id, rc, error). This exercises the real disk-truth + seal/reconcile paths
  while keeping tests deterministic and fast.
- thread_data maps /mnt/user-data/{workspace,outputs} onto real tmp dirs so
  replace_virtual_path resolves chart output virtual paths to the test's tmp
  outputs/ (the schema's _validate_chart_paths requires /mnt/user-data/outputs/
  prefix, and _reconcile_chart_files_against_disk checks disk via the sibling
  outputs dir).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

# Load the tool module fresh via importlib (bypasses conftest sys.modules mock of
# deerflow.subagents.executor; per memory feedback_worktree_shares_main_venv_... the
# worktree's editable pth points at the MAIN repo, so importlib.spec_from_file_location
# is required to load the WORKTREE source under test).
_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_chart_plan_tool.py"
)


def _load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.tools.builtins.run_chart_plan_tool_real",
        _TOOL_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TOOL = _load_tool_module()


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _runtime(workspace: Path, outputs: Path) -> ToolRuntime:
    """ToolRuntime with thread_data mapping /mnt/user-data onto real tmp dirs."""
    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "outputs_path": str(outputs),
                "uploads_path": str(workspace.parent / "uploads"),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _plan(
    charts: list[dict],
    *,
    paradigm: str = "epm",
    budget_remaining: list[dict] | None = None,
) -> dict:
    """Build a plan_charts.json dict.

    Each chart's ``output`` is a /mnt/user-data/outputs/<name>.png virtual path
    (the runner writes a real png into the mapped tmp outputs dir). ``args`` carry
    --output so the runner knows where to write.
    """
    for c in charts:
        c.setdefault("output_mode", "per_subject")
        c.setdefault("subject_index", 0)
        if "output" not in c:
            c["output"] = f"/mnt/user-data/outputs/{c['id']}.png"
        if "args" not in c:
            c["args"] = ["--input", "/mnt/user-data/uploads/fake.txt", "--output", c["output"]]
        c.setdefault("script", "ethoinsight.scripts.epm.plot_fake")
    plan = {
        "schema_version": "1.1",
        "paradigm": paradigm,
        "ev19_template": paradigm,
        "generated_at": "2026-06-24T00:00:00",
        "inputs": {"raw_files": ["/mnt/user-data/uploads/fake.txt"], "groups_file": None, "columns_file": None},
        "charts": charts,
        "charts_fallback_available": [],
        "charts_budget_remaining": budget_remaining or [],
        "skipped": [],
        "user_intent": None,
        "notes": [],
    }
    return plan


def _write_plan(workspace: Path, plan: dict) -> None:
    (workspace / "plan_charts.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8"
    )
    # experiment-context.json for _seal_handoff_to_workspace (analysis_config_id).
    (workspace / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "test-config-chart"}), encoding="utf-8"
    )


def _runner_success(*, outputs: Path):
    """Factory: a runner that writes a tiny png to the --output path (rc=0)."""

    def _r(script: str, args: list[str], task_id: str):
        out = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                out = args[i + 1]
        # out is a /mnt/user-data/outputs/<name>.png virtual path → map to outputs dir.
        if out:
            name = out.rsplit("/", 1)[-1]
            (outputs / name).parent.mkdir(parents=True, exist_ok=True)
            (outputs / name).write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal png header bytes
        return (task_id, 0, "")

    return _r


def _runner_fail_on(fail_ids: set[str], *, outputs: Path):
    """Factory: runner that fails (rc=1, no png) for chart_ids in fail_ids."""
    success = _runner_success(outputs=outputs)

    def _r(script: str, args: list[str], task_id: str):
        if task_id in fail_ids:
            return (task_id, 1, f"planned failure on {task_id}")
        return success(script, args, task_id)

    return _r


def _runner_rc0_no_png(fail_ids: set[str], *, outputs: Path):
    """Factory: runner that returns rc=0 but writes NO png for chart_ids in fail_ids
    (simulates a half-successful script — disk is the only truth)."""
    success = _runner_success(outputs=outputs)

    def _r(script: str, args: list[str], task_id: str):
        if task_id in fail_ids:
            # rc=0 but do NOT write png → disk truth catches it as failed.
            return (task_id, 0, "")
        return success(script, args, task_id)

    return _r


@pytest.fixture
def ws_and_outputs(tmp_path):
    """Real tmp workspace + sibling outputs dir (mirrors prod layout)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return workspace, outputs


@pytest.fixture
def sync_runner(monkeypatch, ws_and_outputs):
    """Default: success runner installed. Per-test override via _set_runner."""
    workspace, outputs = ws_and_outputs
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))
    yield outputs
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", None)


def _set_runner(monkeypatch, runner):
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", runner)


def _call(runtime, **kwargs):
    """Invoke the underlying function (bypass @tool wrapper)."""
    return _TOOL.run_chart_plan_tool.func(runtime, **kwargs)


# ===================================================================
# T1 全画成功
# ===================================================================


class TestAllSuccess:
    def test_t1_all_rendered_completed(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
            {"id": "trajectory_s1", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "completed", res
        assert res["n_total"] == 3
        assert res["n_rendered"] == 3
        assert res["n_failed"] == 0
        assert res["failures"] == []
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert h["sealed_by"] == "run_plan"  # T9
        assert h["status"] == "completed"
        assert len(h["chart_files"]) == 3
        # chart_files are virtual /mnt/user-data/outputs/ paths (T8 disk truth)
        assert all(p.startswith("/mnt/user-data/outputs/") for p in h["chart_files"])


# ===================================================================
# T2 部分失败 → partial
# ===================================================================


class TestPartialFailure:
    def test_t2_one_per_subject_fails_partial(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        _set_runner(monkeypatch, _runner_fail_on({"trajectory_s1"}, outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
            {"id": "trajectory_s1", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "partial", res
        assert res["n_rendered"] == 2
        assert res["n_failed"] == 1
        assert res["failures"] == [{"chart_id": "trajectory_s1", "reason": "planned failure on trajectory_s1"}]
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert len(h["chart_files"]) == 2
        assert len(h["failed_charts"]) == 1
        assert h["failed_charts"][0]["chart_id"] == "trajectory_s1"


# ===================================================================
# T3 aggregate 缺失降级（per_subject 全画，aggregate 缺 → partial）
# ===================================================================


class TestAggregateMissingDegrade:
    def test_t3_aggregate_missing_partial(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        _set_runner(monkeypatch, _runner_fail_on({"box_open_arm"}, outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
            {"id": "trajectory_s1", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs))
        # aggregate missing + per_subject rendered → partial (count未满)
        assert res["status"] == "partial", res
        assert res["n_rendered"] == 2
        assert res["n_failed"] == 1


# ===================================================================
# T4 only_chart_ids 子集
# ===================================================================


class TestOnlyChartIds:
    def test_t4_subset_filter(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
            {"id": "trajectory_s1", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs), only_chart_ids=["box_open_arm"])
        assert res["status"] == "completed", res
        assert res["n_total"] == 1
        assert res["n_rendered"] == 1


# ===================================================================
# T5 plan 缺失 → seal failed + plan_missing
# ===================================================================


class TestPlanMissing:
    def test_t5_plan_missing_seals_failed(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        # No plan_charts.json written. experiment-context.json for seal.
        (workspace / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config-chart"}), encoding="utf-8"
        )
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "failed"
        assert res["error_code"] == "plan_missing"
        # disk has a failed handoff
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert h["status"] == "failed"


# ===================================================================
# T6 plan 空 charts[] → seal failed + empty_plan
# ===================================================================


class TestEmptyPlan:
    def test_t6_empty_charts_seals_failed(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        _write_plan(workspace, _plan([]))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "failed"
        assert res["error_code"] == "empty_plan"
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert h["status"] == "failed"


# ===================================================================
# T7 rc=0 但 png 缺失（脚本半成功）
# ===================================================================


class TestRc0ButPngMissing:
    def test_t7_half_success_goes_to_failed(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        _set_runner(monkeypatch, _runner_rc0_no_png({"trajectory_s0"}, outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "partial", res
        assert res["n_rendered"] == 1
        # the half-success chart is in failed_charts with the disk-truth reason
        assert any(f["chart_id"] == "trajectory_s0" for f in res["failures"])
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        fc = {f["chart_id"]: f["reason"] for f in h["failed_charts"]}
        assert "png missing" in fc["trajectory_s0"]


# ===================================================================
# T8 chart_files 磁盘真相 + reconcile 协同（ETHO-10 真实性门）
# ===================================================================


class TestDiskTruthReconcile:
    def test_t8_handoff_passes_reality_invariant(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "completed", res
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        # Every chart_file is a real png on disk (reconcile 2.0 truth invariant
        # passes cleanly because run_chart_plan only lists rendered pngs).
        for virtual in h["chart_files"]:
            name = virtual.rsplit("/", 1)[-1]
            assert (outputs / name).exists(), f"{name} claimed but not on disk"
        # All rendered aggregate pngs exist (2.2 gate satisfied).
        assert (outputs / "box_open_arm.png").exists()


# ===================================================================
# T9 sealed_by 枚举（schema 校验）
# ===================================================================


class TestSealedByEnum:
    def test_t9_run_plan_enum_accepted(self, ws_and_outputs, sync_runner):
        workspace, outputs = ws_and_outputs
        _write_plan(workspace, _plan([{"id": "box_open_arm", "output_mode": "aggregate"}]))
        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "completed"
        # If the enum value were missing, _seal_handoff_to_workspace would raise
        # ValueError → res would be seal_failed. Confirmed by the seal succeeding.
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert h["sealed_by"] == "run_plan"


# ===================================================================
# T10 on_error=abort
# ===================================================================


class TestOnErrorAbort:
    def test_t10_abort_stops_after_first_failure(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        # box_open_arm (first) fails → abort cancels the rest.
        _set_runner(monkeypatch, _runner_fail_on({"box_open_arm"}, outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
            {"id": "trajectory_s1", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        res = _call(_runtime(workspace, outputs), on_error="abort")
        assert res["status"] == "failed"  # 0 rendered
        failures_text = json.dumps(res["failures"], ensure_ascii=False)
        assert "aborted" in failures_text  # remaining tasks carry aborted marker


# ===================================================================
# T11 MPLBACKEND — worker initializer sets Agg
# ===================================================================


class TestMplBackend:
    def test_t11_worker_init_sets_mplbackend_agg(self, monkeypatch):
        # _worker_init must setdefault MPLBACKEND=Agg so forked workers inherit Agg
        # regardless of parent-process backend state.
        monkeypatch.delenv("MPLBACKEND", raising=False)
        _TOOL._worker_init({})
        assert os.environ.get("MPLBACKEND") == "Agg"

    def test_t11_worker_init_does_not_override_explicit(self, monkeypatch):
        # If an operator explicitly set a backend, worker respects it (setdefault).
        monkeypatch.setenv("MPLBACKEND", "module://custom")
        _TOOL._worker_init({})
        assert os.environ.get("MPLBACKEND") == "module://custom"


# ===================================================================
# T12 import 环（裸导入两生产入口）
# ===================================================================


class TestImportNoCycle:
    def test_t12_gateway_import_no_cycle(self):
        # Bare import of the two production entrypoints (no conftest mock).
        # Run in-subprocess to mirror tests/test_gateway_import_no_cycle.py.
        import subprocess
        import sys

        backend_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = backend_root
        for entry in ("import app.gateway", "from deerflow.agents import make_lead_agent"):
            r = subprocess.run([sys.executable, "-c", entry], cwd=backend_root, env=env, capture_output=True)
            assert r.returncode == 0, f"{entry} failed: {r.stderr.decode()}"


# ===================================================================
# T13 装配链可见性（#187 教训）
# ===================================================================


class TestAssemblyChainVisibility:
    def test_t13_run_chart_plan_in_builtin_tools(self):
        # #187 lesson: a tool can have @tool def + __init__ export + prompt mention
        # but STILL not be in BUILTIN_TOOLS → get_available_tools won't include it
        # → chart-maker can't call it. Assert the assembly chain end-to-end.
        from deerflow.tools.tools import BUILTIN_TOOLS

        names = [getattr(t, "name", "") for t in BUILTIN_TOOLS]
        assert "run_chart_plan" in names, f"run_chart_plan missing from BUILTIN_TOOLS: {names}"
        assert "run_metric_plan" in names  # parity sanity

    def test_t13_chart_maker_tools_whitelist_has_run_chart_plan(self):
        from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG

        assert "run_chart_plan" in CHART_MAKER_CONFIG.tools


# ===================================================================
# T14 SealGate 放行（chart-maker 移出 _REQUIRES_SEAL，回归）
# ===================================================================


class TestSealGatePassThrough:
    def test_t14_chart_maker_not_in_requires_seal(self):
        from deerflow.agents.middlewares.seal_gate_middleware import _RECONSTRUCTABLE, _REQUIRES_SEAL

        # chart-maker moved out (aligned with code-executor: run_chart_plan
        # produces-and-delivers in one tool, structurally cannot miss seal).
        assert "chart-maker" not in _REQUIRES_SEAL
        assert "chart-maker" not in _RECONSTRUCTABLE
        # data-analyst / report-writer must remain (not over-deleted).
        assert "data-analyst" in _REQUIRES_SEAL
        assert "report-writer" in _REQUIRES_SEAL
        assert "report-writer" in _RECONSTRUCTABLE

    def test_t14_seal_gate_after_model_passes_chart_maker(self):
        # Construct a state where chart-maker ended on pure text without calling
        # seal. After moving out of _REQUIRES_SEAL, after_model rule 1 returns None
        # (the gate does not intercept — run_chart_plan already sealed deterministically).
        from langchain_core.messages import AIMessage

        from deerflow.agents.middlewares.seal_gate_middleware import SealGateMiddleware

        gate = SealGateMiddleware("chart-maker")
        state = {"messages": [AIMessage(content="OK: charts written")]}
        runtime = None  # rule 1 short-circuits before runtime is touched.
        assert gate._check(state, runtime) is None
        assert gate.after_agent(state, runtime) is None  # also no auto-seal
