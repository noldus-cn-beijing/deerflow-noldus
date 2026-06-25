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
    """Invoke the underlying function and unpack the result dict from the Command.

    Spec 2026-06-25-run-chart-plan-auto-register-artifacts：run_chart_plan_tool 现返
    ``Command(update={"messages":[ToolMessage(json(result))...]})``。本 helper 把首条
    ToolMessage 的 json content 解回 result dict，让既有读 ``res["status"]`` /
    ``res["n_rendered"]`` / ``res["failures"]`` / ``res["error_code"]`` 的断言（T1-T14
    + F1 + M2）零改动继续工作。
    """
    cmd = _TOOL.run_chart_plan_tool.func(runtime, tool_call_id="test-tcid", **kwargs)
    msgs = cmd.update.get("messages", [])
    assert msgs, f"Command 缺 ToolMessage: {cmd.update}"
    return json.loads(msgs[0].content)


def _call_command(runtime, **kwargs):
    """Invoke the underlying function, return the raw Command.

    用于直接断言 ``cmd.update["artifacts"]`` / ``cmd.update["charts_status"]`` /
    ``ToolMessage.tool_call_id`` 透传的测试（_call 会把这些信息丢掉）。
    """
    return _TOOL.run_chart_plan_tool.func(runtime, tool_call_id="test-tcid", **kwargs)


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


# ===================================================================
# Spec 2026-06-24-run-chart-plan-permissionerror — F1 argv 预解析
# (T1-T5：进程池把 plot 脚本 args 原样透传 /mnt → savefig 崩塌；F1 在 Step 5
#  对 args 逐项跑 replace_virtual_path 预解析，对齐 bash 重写语义)
# ===================================================================


class TestArgvPreResolved:
    """F1：喂 worker 的 args 必须已是真实物理路径（无 /mnt 前缀）。

    Spec §四 T1/T3：注入 runner 捕获实际收到的 args，断言 /mnt 虚拟路径已被
    replace_virtual_path 预解析。改前红 = runner 收到 ``--output
    /mnt/user-data/outputs/x.png``；改后绿 = 收到 ``<workspace_parent>/outputs/x.png``。
    """

    def _capturing_runner(self, captured: dict):
        """Runner that records the args it received keyed by task_id, rc=0, writes png."""

        def _r(script: str, args: list[str], task_id: str):
            captured[task_id] = list(args)
            return (task_id, 0, "")

        return _r

    def test_f1_t1_output_argv_pre_resolved(self, ws_and_outputs, monkeypatch):
        # T1 核心：runner 收到的 --output 已是真实路径（无 /mnt 前缀）。
        workspace, outputs = ws_and_outputs
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", self._capturing_runner(captured))
        charts = [{"id": "trajectory_s0", "output_mode": "per_subject"}]
        _write_plan(workspace, _plan(charts))
        _call(_runtime(workspace, outputs))

        args = captured["trajectory_s0"]
        # The --output value must be the real physical path (mapped to outputs/),
        # NOT the raw /mnt/user-data/outputs/... virtual path.
        out_val = args[args.index("--output") + 1]
        assert not out_val.startswith("/mnt/"), (
            f"F1 red: --output not pre-resolved, still virtual: {out_val}"
        )
        assert out_val == str(outputs / "trajectory_s0.png"), out_val

    def test_f1_t2_non_path_args_preserved(self, ws_and_outputs, monkeypatch):
        # T2：非路径项（--parameters-json, JSON 字符串, --dpi, 150）原样不变。
        workspace, outputs = ws_and_outputs
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", self._capturing_runner(captured))
        charts = [
            {
                "id": "box_open_arm",
                "output_mode": "aggregate",
                "args": [
                    "--inputs", "/mnt/user-data/uploads/inputs.json",
                    "--output", "/mnt/user-data/outputs/box_open_arm.png",
                    "--parameters-json", '{"open_arm_zones": ["open"]}',
                    "--dpi", "150",
                ],
            }
        ]
        _write_plan(workspace, _plan(charts))
        _call(_runtime(workspace, outputs))

        args = captured["box_open_arm"]
        # JSON string preserved verbatim (not a /mnt path → unchanged).
        assert '{"open_arm_zones": ["open"]}' in args
        # Numeric / flag args preserved verbatim.
        assert args[args.index("--dpi") + 1] == "150"
        # --parameters-json flag itself unchanged (not a path).
        assert "--parameters-json" in args

    def test_f1_t3_multiple_subjects_all_resolved(self, ws_and_outputs, monkeypatch):
        # T3：28 个 per_subject chart 各含不同 /mnt 路径 → 全部 args 解析（不只首个）。
        workspace, outputs = ws_and_outputs
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", self._capturing_runner(captured))
        charts = [
            {
                "id": f"trajectory_s{i}",
                "output_mode": "per_subject",
                "args": [
                    "--inputs", f"/mnt/user-data/uploads/inputs_s{i}.json",
                    "--output", f"/mnt/user-data/outputs/plot_s{i}.png",
                ],
            }
            for i in range(28)
        ]
        _write_plan(workspace, _plan(charts))
        _call(_runtime(workspace, outputs))

        # Every chart's args must have NO /mnt-prefixed item remaining.
        for tid, args in captured.items():
            mnt_items = [a for a in args if a.startswith("/mnt/")]
            assert not mnt_items, f"F1 red: {tid} still has unresolved /mnt args: {mnt_items}"

    def test_f1_t4_thread_data_none_passthrough(self, ws_and_outputs, monkeypatch):
        # T4：thread_data 缺 outputs_path 映射 → replace_virtual_path 原样返回，不 crash。
        # （Step 1 仍保证 workspace_path 存在，但 outputs_path 缺失时 outputs 虚拟路径
        # 不被解析——这是 thread_data 映射不全的退化场景，F1 绝不能 crash。）
        workspace, outputs = ws_and_outputs
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", self._capturing_runner(captured))
        charts = [{"id": "trajectory_s0", "output_mode": "per_subject"}]
        _write_plan(workspace, _plan(charts))
        # thread_data WITHOUT outputs_path → outputs 虚拟路径无法解析。
        runtime_no_outputs = ToolRuntime(
            state={"thread_data": {"workspace_path": str(workspace)}},
            context=None,
            config={},
            stream_writer=None,
            tool_call_id="test-id",
            store=None,
        )
        _call(runtime_no_outputs)

        # Did not crash (F1 ran replace_virtual_path per-arg). The output arg stays
        # virtual because no outputs mapping — but that's the pre-existing thread_data
        # gap, not F1's job to fix. F1's contract: apply replace_virtual_path without
        # raising.
        assert "trajectory_s0" in captured  # runner was reached (no crash in Step 5)

    def test_f1_t5_chart_files_still_virtual(self, ws_and_outputs, monkeypatch):
        # T5：F1 改的是喂 worker 的 args（真实路径），但 chart_meta["output"] 仍存虚拟
        # 路径 → Step 7 核磁盘后 chart_files 里是 /mnt/user-data/outputs/ 前缀
        # （守 ChartMakerHandoff _validate_chart_paths 契约）。两处解耦不冲突。
        workspace, outputs = ws_and_outputs
        # Use the success runner so pngs actually land on disk (real path).
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        _call(_runtime(workspace, outputs))

        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        # chart_files MUST stay virtual (the schema's _validate_chart_paths enforces
        # /mnt/user-data/outputs/ prefix — F1 pre-resolves only the worker args, not
        # the stored chart_files contract).
        assert all(p.startswith("/mnt/user-data/outputs/") for p in h["chart_files"]), h["chart_files"]
        assert h["status"] == "completed"


# ===================================================================
# Spec 2026-06-25-chart-maker-seal-once-and-results-key-uniqueness — M2 (R2)
# per_subject 多 chart 共享同一 chart id → results dict key 必须唯一（task index），
# 否则 112 个结果互相覆盖成最后写入的少数几个。R1/M1 修了 double-seal 后，run_chart_plan
# 路径不能再靠 chart-maker 手调 seal 的 LLM 自报"救"计数缺口，故 R2 必须同 PR 修。
# ===================================================================


class TestPerSubjectSameIdAllReconciled:
    """R2 红→绿：per_subject 多 chart 同 id 全核盘（不丢图）。

    构造 plan：1 个 chart id × N subject（真实 dogfood 形态：open_arm_time_ratio_bar
    被复用 28 次，靠 subject_index 区分）。全 rc=0 落盘。
    红态：results dict 按 tid（=chart id）覆盖 → 只 1 entry → Step 7 核盘循环只核 1 次
          → chart_files 只 1 张。
    绿态：results 按 task index → chart_files N 张全核出。
    """

    def test_t4_per_subject_same_id_all_reconciled(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        # 28 个 chart 共享同一 id，靠 subject_index + 不同 output 文件名区分
        # （镜像 dogfood thread a6e3775c 的 open_arm_time_ratio_bar × 28 subject 形态）。
        charts = [
            {
                "id": "open_arm_time_ratio_bar",
                "output_mode": "per_subject",
                "subject_index": i,
                "output": f"/mnt/user-data/outputs/plot_open_arm_ratio_s{i}.png",
                "args": [
                    "--input", "/mnt/user-data/uploads/fake.txt",
                    "--output", f"/mnt/user-data/outputs/plot_open_arm_ratio_s{i}.png",
                ],
            }
            for i in range(28)
        ]
        _write_plan(workspace, _plan(charts))
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))

        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "completed", res
        # 核心断言：28 张同 id chart 全核盘，不丢图。
        assert res["n_total"] == 28, res
        assert res["n_rendered"] == 28, (
            f"R2 red: 同 id 多 chart 只核盘 {res['n_rendered']} 张（应为 28）"
        )
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert len(h["chart_files"]) == 28, (
            f"R2 red: handoff chart_files 只 {len(h['chart_files'])} 条（应为 28）"
        )


class TestSameIdPartialFailuresAllReasonsKept:
    """R2：同 id 部分失败 reason 不丢。

    28 个同 id chart，其中 3 个失败（不同 reason）→ failed_charts 应含 3 条独立失败。
    红态：results 按 id 覆盖 → 同 id 只留最后 1 个结果（可能是成功也可能是失败）。
    绿态：results 按 index → 3 个失败全保留，failed_charts 含 3 条。
    """

    def test_t5_same_id_partial_failures_all_reasons_kept(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs

        # 用 subject_index 标记每个 chart；失败的 3 个用 task_id 区分。
        # runner 按 task_id 决定成败，task_id 这里用 chart id + subject_index 拼成唯一
        # 但 chart_meta 仍按同 id 建（复现多对一）。为了让 runner 能区分，我们让 plan 里
        # 每个 chart 的 id 都相同，但 runner 按 output 文件名（隐式唯一）落盘。
        # _runner_fail_on 按 task_id 匹配；task_id = chart id（相同）→ 无法按 subject 区分。
        # 故改用 capturing runner：按 output 路径里的 subject index 判失败。
        fail_subjects = {3, 7, 11}
        success = _runner_success(outputs=outputs)

        def _r(script: str, args: list[str], task_id: str):
            # 从 --output 提取 subject index（plot_open_arm_ratio_s{N}.png）。
            out = next((args[i + 1] for i, a in enumerate(args) if a == "--output"), "")
            name = out.rsplit("/", 1)[-1]
            # parse s{N}
            import re as _re

            m = _re.search(r"_s(\d+)\.png$", name)
            if m and int(m.group(1)) in fail_subjects:
                return (task_id, 1, f"planned failure subject {m.group(1)}")
            return success(script, args, task_id)

        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _r)

        charts = [
            {
                "id": "open_arm_time_ratio_bar",
                "output_mode": "per_subject",
                "subject_index": i,
                "output": f"/mnt/user-data/outputs/plot_open_arm_ratio_s{i}.png",
                "args": [
                    "--input", "/mnt/user-data/uploads/fake.txt",
                    "--output", f"/mnt/user-data/outputs/plot_open_arm_ratio_s{i}.png",
                ],
            }
            for i in range(28)
        ]
        _write_plan(workspace, _plan(charts))

        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "partial", res
        assert res["n_rendered"] == 25, res  # 28 - 3 failed
        assert res["n_failed"] == 3, res
        # failed_charts 必须含 3 条独立失败 reason（不丢）。
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert len(h["failed_charts"]) == 3, (
            f"R2 red: 同 id 失败 reason 被覆盖，只留 {len(h['failed_charts'])} 条（应 3）: {h['failed_charts']}"
        )
        reasons = {fc["reason"] for fc in h["failed_charts"]}
        assert len(reasons) == 3, f"R2: 3 条 reason 应各不同，实得 {reasons}"


class TestAllSuccessHandoffConsistent:
    """R2 + R1 联合回归（复现 dogfood 正路径）。

    112 图全 rc=0 落盘 → status=completed, chart_files=112, failed_charts=0,
    sealed_by=run_plan（不是 model!）。这是 dogfood thread a6e3775c 应走的正路径：
    封存只允许一次（M1）后 chart-maker 不能再覆盖成 model；results key 唯一化（M2）后
    112 张全核盘不靠 LLM 自报。
    """

    def test_t6_112_charts_all_success_handoff_consistent(self, ws_and_outputs, monkeypatch):
        workspace, outputs = ws_and_outputs
        # 4 个唯一 chart id × 28 subject = 112 张（复现 dogfood 形态）。
        chart_ids = [
            "open_arm_time_ratio_bar",
            "zone_entry_distribution",
            "center_time_ratio_bar",
            "closed_arm_time_ratio_bar",
        ]
        charts = []
        for cid in chart_ids:
            for i in range(28):
                charts.append({
                    "id": cid,
                    "output_mode": "per_subject",
                    "subject_index": i,
                    "output": f"/mnt/user-data/outputs/plot_{cid}_s{i}.png",
                    "args": [
                        "--input", "/mnt/user-data/uploads/fake.txt",
                        "--output", f"/mnt/user-data/outputs/plot_{cid}_s{i}.png",
                    ],
                })
        assert len(charts) == 112
        _write_plan(workspace, _plan(charts))
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))

        res = _call(_runtime(workspace, outputs))
        assert res["status"] == "completed", res
        assert res["n_total"] == 112, res
        assert res["n_rendered"] == 112, (
            f"R2 red: 112 张同 id（4×28）只核盘 {res['n_rendered']} 张（应为 112）"
        )
        h = json.loads((workspace / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert h["status"] == "completed"
        assert len(h["chart_files"]) == 112, (
            f"R2 red: handoff chart_files 只 {len(h['chart_files'])} 条（应为 112）"
        )
        assert h["failed_charts"] == []
        # R1：sealed_by 是 run_plan（确定性），不是 model（LLM 手调覆盖）。
        assert h["sealed_by"] == "run_plan", h["sealed_by"]


# ===================================================================
# Spec 2026-06-25-run-chart-plan-auto-register-artifacts —— run_chart_plan
# 返 Command 自己确定性登记 artifacts（不再依赖 chart-maker 逐张 present_files）
# ===================================================================


class TestArtifactsRegistered:
    """run_chart_plan 必须把 chart_files 全量登记进 state.artifacts（修「113 张图只显示 1 张」）。

    红态：run_chart_plan 返 dict、从不更新 artifacts → ``cmd.update`` 里没有 ``artifacts`` 键。
    绿态：run_chart_plan 返 Command(update={"artifacts": [...113 meta...], ...})，经与
          present_files 同款 merge_artifacts reducer（按 path 去重）上行到 lead thread state。
    """

    def _build_113_charts(self):
        """复现 dogfood 形态：1 aggregate + 112 per_subject（4×28）。"""
        charts = [{"id": "box_open_arm", "output_mode": "aggregate"}]
        chart_ids = [
            "open_arm_time_ratio_bar",
            "zone_entry_distribution",
            "center_time_ratio_bar",
            "closed_arm_time_ratio_bar",
        ]
        for cid in chart_ids:
            for i in range(28):
                charts.append({
                    "id": cid,
                    "output_mode": "per_subject",
                    "subject_index": i,
                    "output": f"/mnt/user-data/outputs/plot_{cid}_s{i}.png",
                    "args": [
                        "--input", "/mnt/user-data/uploads/fake.txt",
                        "--output", f"/mnt/user-data/outputs/plot_{cid}_s{i}.png",
                    ],
                })
        return charts

    def test_113_charts_all_registered_as_artifacts(self, ws_and_outputs, monkeypatch):
        # 核心：113 张图全进 cmd.update["artifacts"]（不是依赖 LLM 逐张 present）。
        workspace, outputs = ws_and_outputs
        charts = self._build_113_charts()
        assert len(charts) == 113
        _write_plan(workspace, _plan(charts))
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))

        cmd = _call_command(_runtime(workspace, outputs))
        artifacts = cmd.update.get("artifacts")
        assert artifacts is not None, f"red: Command 没登记 artifacts: {cmd.update}"
        assert len(artifacts) == 113, (
            f"red: 只登记了 {len(artifacts)} 张 artifact（应 113）: {artifacts}"
        )
        # 全是 /mnt/user-data/outputs/ 前缀路径（命中 plan by_output 的升级成 meta dict，
        # 未命中的退回裸 string——但这里 113 张全命中 plan，故全是 meta dict）。
        for meta in artifacts:
            assert isinstance(meta, dict), f"113 张全命中 plan 应升级成 meta dict: {meta}"
            assert meta["path"].startswith("/mnt/user-data/outputs/"), meta
            assert meta["kind"] == "chart", meta
        # aggregate 那张 output_mode=="aggregate"。
        agg = [m for m in artifacts if m.get("output_mode") == "aggregate"]
        assert len(agg) == 1, f"应有 1 张 aggregate，实得 {len(agg)}"

    def test_tool_message_carries_status_and_tool_call_id(self, ws_and_outputs, monkeypatch):
        # ToolMessage content 是 json(result dict)，chart-maker 决策树 parse 它零感知变化。
        # tool_call_id 正确透传（InjectedToolCallId，LLM 不可见但 ToolMessage 必须绑定）。
        workspace, outputs = ws_and_outputs
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success(outputs=outputs))

        cmd = _call_command(_runtime(workspace, outputs))
        msg = cmd.update["messages"][0]
        assert msg.tool_call_id == "test-tcid", msg.tool_call_id
        payload = json.loads(msg.content)
        assert payload["status"] == "completed", payload
        assert payload["n_rendered"] == 2, payload
        # result dict 形态与决策树（chart_maker.py step 9 parse status/failures）零感知变化。
        assert "failures" in payload and "gate_signals" in payload, payload

    def test_charts_status_into_state_on_partial(self, ws_and_outputs, monkeypatch):
        # partial（有失败）→ charts_status 摘要进 state（前端拿失败原因）。
        workspace, outputs = ws_and_outputs
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"trajectory_s0"}, outputs=outputs))
        charts = [
            {"id": "box_open_arm", "output_mode": "aggregate"},
            {"id": "trajectory_s0", "output_mode": "per_subject"},
        ]
        _write_plan(workspace, _plan(charts))

        cmd = _call_command(_runtime(workspace, outputs))
        cs = cmd.update.get("charts_status")
        assert cs is not None, f"partial 应带 charts_status 进 state: {cmd.update}"
        assert cs["n_rendered"] == 1, cs
        assert any(f["chart_id"] == "trajectory_s0" for f in cs["failed"]), cs

    def test_error_path_returns_command_no_artifacts(self, ws_and_outputs, sync_runner):
        # 6 个错误早退点也返 Command（签名 -> Command 后混返 dict 会退化），
        # 且错误路径**不写 artifacts**（不污染画廊）。
        workspace, outputs = ws_and_outputs
        _write_plan(workspace, _plan([]))  # empty_plan

        cmd = _call_command(_runtime(workspace, outputs))
        # 错误路径返 Command（有 ToolMessage）。
        msg = cmd.update["messages"][0]
        payload = json.loads(msg.content)
        assert payload["status"] == "failed", payload
        assert payload["error_code"] == "empty_plan", payload
        # 错误路径不写 artifacts（不要把失败画图的空集/部分集污染画廊）。
        assert "artifacts" not in cmd.update, (
            f"red: 错误路径不应写 artifacts: {cmd.update}"
        )
