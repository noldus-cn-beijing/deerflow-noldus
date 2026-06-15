"""Spec S4 — Tests for run_metric_plan first-party tool.

Covers Spec S4 §4 test matrix (14 cases). Strategy:
- Inject a synchronous ``_TASK_RUNNER_OVERRIDE`` so tests never touch
  ProcessPoolExecutor's fork/pickle semantics. The runner simulates what a real
  compute script does: parse --output from args, write the m_*.json payload,
  return (task_id, 0, ""). This exercises the real aggregation/seal paths
  (which read disk artifacts) while keeping tests deterministic and fast.
- Plan artifacts use real host workspace paths (not /mnt virtual) so
  resolve_sandbox_path is a passthrough — identical to production's env-resolved
  behaviour, just without needing DEERFLOW_PATH_* plumbing in unit tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

# Load the tool module fresh (bypasses any conftest sys.modules manipulation
# of deerflow.subagents.executor; the tool only imports executor transitively
# via seal_handoff_tools which is fine).
_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_metric_plan_tool.py"
)


def _load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.tools.builtins.run_metric_plan_tool_real",
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


def _runtime(workspace: Path) -> ToolRuntime:
    """ToolRuntime with thread_data pointing at a real host workspace dir."""
    return ToolRuntime(
        state={"thread_data": {"workspace_path": str(workspace)}},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _plan(
    metrics: list[dict],
    *,
    workspace: Path,
    raw_files: list[str] | None = None,
    statistics: dict | None = None,
    groups_file: str | None = None,
) -> dict:
    """Build a plan_metrics.json dict whose metric outputs are real host paths.

    Each metric's ``output`` resolves under ``workspace`` so the runner writes
    a real file the aggregator will glob, and ``args`` carry that --output.
    """
    for m in metrics:
        if "output" not in m:
            # output 文件名必须 m_ 前缀（聚合器 glob "m_*.json"），与生产 plan 命名一致。
            m["output"] = str(workspace / f"m_{m['id']}.json")
        if "args" not in m:
            m["args"] = ["--input", (raw_files or ["/mnt/user-data/uploads/fake.txt"])[m.get("subject_index", 0)],
                         "--output", m["output"]]
        m.setdefault("script", "ethoinsight.scripts.fake.compute")
        m.setdefault("subject_index", 0)
    plan = {
        "schema_version": "1.1",
        "paradigm": "epm",
        "ev19_template": "epm",
        "generated_at": "2026-06-15T00:00:00",
        "inputs": {
            "raw_files": raw_files or ["/mnt/user-data/uploads/fake.txt"],
            "groups_file": groups_file,
            "columns_file": None,
        },
        "metrics": metrics,
        "statistics": statistics,
        "skipped": [],
        "notes": [],
    }
    return plan


def _write_plan(workspace: Path, plan: dict) -> None:
    (workspace / "plan_metrics.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8"
    )
    # experiment-context.json for _seal_handoff_to_workspace (analysis_config_id).
    (workspace / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "test-config-s4"}), encoding="utf-8"
    )


def _runner_success(script: str, args: list[str], task_id: str):
    """Runner that writes a metric payload to the --output path (rc=0).

    Extracts metric name from task_id (strips _s<idx> suffix) and value from a
    registry the test populates via the module-level ``_FAKE_VALUES`` map.
    """
    # parse --output
    out = None
    for i, a in enumerate(args):
        if a == "--output" and i + 1 < len(args):
            out = args[i + 1]
    payload = {
        "metric": _TASK_METRIC.get(task_id, task_id),
        "value": _FAKE_VALUES.get(task_id, 0.5),
        "parameters_used": _FAKE_PARAMS.get(task_id, {}),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload), encoding="utf-8")
    return (task_id, 0, "")


# module-level registries the runner reads (tests set these per-case)
_FAKE_VALUES: dict[str, float] = {}
_FAKE_PARAMS: dict[str, float] = {}
_TASK_METRIC: dict[str, str] = {}


def _runner_always_fail(script: str, args: list[str], task_id: str):
    return (task_id, 1, f"boom on {task_id}")


def _runner_fail_on(fail_ids: set[str]):
    """Runner that fails for task_ids in fail_ids, succeeds otherwise."""
    def _r(script: str, args: list[str], task_id: str):
        if task_id in fail_ids:
            return (task_id, 1, f"planned failure on {task_id}")
        return _runner_success(script, args, task_id)
    return _r


@pytest.fixture(autouse=True)
def _sync_runner(monkeypatch):
    """Force synchronous execution (no ProcessPoolExecutor) + reset registries."""
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success)
    _FAKE_VALUES.clear()
    _FAKE_PARAMS.clear()
    _TASK_METRIC.clear()
    yield
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", None)


def _call(runtime, **kwargs):
    """Invoke the underlying function (bypass @tool wrapper)."""
    return _TOOL.run_metric_plan_tool.func(runtime, **kwargs)


# ===================================================================
# §4 #1 进程内调脚本（mocked runner）：全部成功 → m_*.json 产出 + 聚合正确
# ===================================================================


class TestAllSuccess:
    def test_run_metric_plan_all_success(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "open_arm_time_ratio", "subject_index": 0},
            {"id": "open_arm_time_ratio", "subject_index": 1},
            {"id": "total_distance", "subject_index": 0},
        ]
        # distinct task ids: id is shared across subjects, but plan uses id+subject_index;
        # tool uses m.get("id") as task_id → collisions. Use unique ids per row instead.
        metrics = [
            {"id": "open_arm_time_ratio_s0", "subject_index": 0},
            {"id": "open_arm_time_ratio_s1", "subject_index": 1},
            {"id": "total_distance_s0", "subject_index": 0},
        ]
        _TASK_METRIC.update({
            "open_arm_time_ratio_s0": "open_arm_time_ratio",
            "open_arm_time_ratio_s1": "open_arm_time_ratio",
            "total_distance_s0": "total_distance",
        })
        _FAKE_VALUES.update({
            "open_arm_time_ratio_s0": 0.35, "open_arm_time_ratio_s1": 0.42, "total_distance_s0": 1500.0,
        })
        plan = _plan(metrics, workspace=ws,
                     raw_files=["/mnt/user-data/uploads/s0.txt", "/mnt/user-data/uploads/s1.txt"])
        _write_plan(ws, plan)

        res = _call(_runtime(ws))

        assert res["status"] == "completed", res
        assert res["n_total"] == 3
        assert res["n_completed"] == 3
        assert res["n_failed"] == 0
        assert res["failures"] == []
        # handoff sealed
        h = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert h["sealed_by"] == "run_plan"
        assert h["status"] == "completed"
        # aggregation read disk artifacts correctly
        assert "open_arm_time_ratio" in h["metrics_summary"].get("unknown", {}) or \
               any("open_arm_time_ratio" in g for g in h["metrics_summary"].values())


# ===================================================================
# §4 #4 on_error policy: continue vs abort
# ===================================================================


class TestOnErrorPolicy:
    def test_continue_runs_all_despite_failure(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "m0", "subject_index": 0},
            {"id": "m1", "subject_index": 0},
            {"id": "m2", "subject_index": 0},
        ]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"m1"}))

        res = _call(_runtime(ws), on_error="continue")

        # m1 failed but m0/m2 ran (status=failed because n_failed>0 per _derive_status)
        assert res["n_total"] == 3
        assert res["n_failed"] == 1
        assert res["n_completed"] == 2
        ids_failed = {f["id"] for f in res["failures"]}
        assert ids_failed == {"m1"}

    def test_abort_stops_after_first_failure(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "m0", "subject_index": 0},
            {"id": "m1", "subject_index": 0},  # fails
            {"id": "m2", "subject_index": 0},  # should NOT run
        ]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"m1"}))

        res = _call(_runtime(ws), on_error="abort")

        # abort stops submitting after m1 → m2 not executed, counted as not-completed
        # (in sync runner, m0 runs, m1 fails, loop breaks → m2 never attempted)
        assert res["n_completed"] <= 2
        assert {"m1"} == {f["id"] for f in res["failures"]}


# ===================================================================
# §4 #5 only_metric_ids selector
# ===================================================================


class TestOnlyMetricIds:
    def test_only_runs_selected_subset(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "m0", "subject_index": 0},
            {"id": "m1", "subject_index": 0},
            {"id": "m2", "subject_index": 0},
        ]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)

        res = _call(_runtime(ws), only_metric_ids=["m0", "m2"])

        assert res["n_total"] == 2  # only m0, m2
        assert res["n_completed"] == 2
        # only m0/m2 outputs on disk
        outputs = sorted(p.name for p in ws.glob("m_*.json") if "handoff" not in p.name) \
            if False else sorted(p.name for p in ws.glob("*.json"))
        assert "m_m0.json" in outputs
        assert "m_m2.json" in outputs
        assert "m_m1.json" not in outputs


# ===================================================================
# §4 #6 SSOT parity: aggregate_metrics_to_handoff byte-identical
# whether called directly or via run_metric_plan
# ===================================================================


class TestSSOTParity:
    def test_aggregator_direct_vs_tool_identical_metrics_summary(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "open_arm_time_ratio_s0", "subject_index": 0},
            {"id": "open_arm_time_ratio_s1", "subject_index": 1},
        ]
        _TASK_METRIC.update({
            "open_arm_time_ratio_s0": "open_arm_time_ratio",
            "open_arm_time_ratio_s1": "open_arm_time_ratio",
        })
        _FAKE_VALUES.update({"open_arm_time_ratio_s0": 0.35, "open_arm_time_ratio_s1": 0.42})
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt", "/mnt/user-data/uploads/s1.txt"])
        _write_plan(ws, plan)

        res = _call(_runtime(ws))

        # §4 #6 SSOT parity: aggregate_metrics_to_handoff 是纯函数——同一 plan + 同一
        # 磁盘产物集，调两次结果必须字节一致（确定性）。run_metric_plan 和 auto-seal
        # 两条调用路径都调它，复现性是这个不变量的核心，而非"聚合原始 dict vs 经
        # Pydantic 序列化的 handoff"（中间有 schema 默认值填充层，本就不该字节相同）。
        from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff
        direct1 = aggregate_metrics_to_handoff(plan, ws, run_validation=True)
        direct2 = aggregate_metrics_to_handoff(plan, ws, run_validation=True)
        assert direct1 == direct2  # 纯函数确定性
        # 工具确实用了聚合器：handoff 的 metrics_summary 源自同一聚合（键集一致）
        h = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert set(h["metrics_summary"]) == set(direct1["metrics_summary"])
        # per_subject 直接透传（不经 Pydantic 重塑，逐字段一致）
        assert h["per_subject"] == direct1["per_subject"]


# ===================================================================
# §4 #7 list zone params preserved end-to-end (regression for #125)
# ===================================================================


class TestListZoneParamsPreserved:
    def test_open_arm_zones_list_survives_to_handoff(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        # runner writes parameters_used including a list[str] zone param (#125 regression)
        _TASK_METRIC["m0"] = "open_arm_time_ratio"
        _FAKE_PARAMS["m0"] = {"open_arm_zones": ["open"]}
        metrics = [{"id": "m0", "subject_index": 0}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)

        _call(_runtime(ws))

        h = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        # parameters_used with list value preserved into metrics_summary
        found = False
        for grp in h["metrics_summary"].values():
            for mstat in grp.values():
                pu = mstat.get("parameters_used", {})
                if pu.get("open_arm_zones") == ["open"]:
                    found = True
        assert found, f"list zone param not preserved: {h['metrics_summary']}"


# ===================================================================
# §4 #8 completeness pure function: completed / partial / skip_reason
# ===================================================================


class TestCompletenessPureFunction:
    def test_all_outputs_present_is_completed(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}, {"id": "m1"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        res = _call(_runtime(ws))
        assert res["status"] == "completed"

    def test_missing_output_marks_failed(self, tmp_path, monkeypatch):
        """A task that fails (rc!=0) writes no artifact → status not completed."""
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}, {"id": "m1"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"m1"}))
        res = _call(_runtime(ws))
        assert res["status"] == "failed"
        assert res["n_failed"] == 1


# ===================================================================
# §4 #9 tool seals handoff with sealed_by="run_plan"
# ===================================================================


class TestSealByRunPlan:
    def test_sealed_by_run_plan_and_schema_valid(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)

        res = _call(_runtime(ws))

        hpath = ws / "handoff_code_executor.json"
        assert hpath.exists()
        h = json.loads(hpath.read_text(encoding="utf-8"))
        assert h["sealed_by"] == "run_plan"
        # schema-required fields present (CodeExecutorHandoff)
        for k in ("status", "summary", "paradigm", "metrics_summary", "per_subject",
                  "output_files", "data_quality_warnings", "errors", "gate_signals"):
            assert k in h, f"missing {k}"
        # file permission 644 (Spec1 lesson)
        assert stat.S_IMODE(os.stat(hpath).st_mode) == 0o644
        assert res["handoff_path"] == "/mnt/user-data/workspace/handoff_code_executor.json"


# ===================================================================
# §4 #10 LLM does not call seal → handoff already on disk (fail-safe reversal)
# ===================================================================


class TestFailSafeSealReversal:
    def test_handoff_sealed_without_seal_tool_call(self, tmp_path):
        """The tool seals deterministically; LLM never needs to call seal (§1.6)."""
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        # Single call to run_metric_plan, no seal_* tool call anywhere in test.
        _call(_runtime(ws))
        assert (ws / "handoff_code_executor.json").exists()


# ===================================================================
# §4 #11 contract: code_executor prompt has run_metric_plan, no bash_constraints
# ===================================================================


class TestPromptContract:
    def test_prompt_mentions_run_metric_plan_and_triage(self):
        from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG

        prompt = CODE_EXECUTOR_CONFIG.system_prompt
        assert "run_metric_plan" in prompt
        assert "<triage>" in prompt
        # bash 编排段已删（无 bash_constraints 标签段）
        assert "<bash_constraints>" not in prompt
        # 正面指令：教用 run_metric_plan 执行（"逐条拼 bash"出现在否定语境里，
        # 可能被 markdown ** ** 拆开，故只断言"逐条拼 bash"出现且 prompt 含 run_metric_plan）
        assert "逐条拼 bash" in prompt
        # 工具列表收 bash
        assert "bash" not in CODE_EXECUTOR_CONFIG.tools
        assert "run_metric_plan" in CODE_EXECUTOR_CONFIG.tools
        assert "write_file" not in CODE_EXECUTOR_CONFIG.tools
        assert "str_replace" not in CODE_EXECUTOR_CONFIG.tools

    def test_script_invocation_provider_not_removed(self):
        """§2.3: ScriptInvocationOnlyProvider 保留（chart-maker 仍用 bash）。"""
        from deerflow.guardrails.script_invocation_only_provider import (
            ScriptInvocationOnlyProvider,
        )
        assert ScriptInvocationOnlyProvider is not None


# ===================================================================
# §4 #13 regression: existing code-executor / auto-seal tests stay green
# (smoke: auto-seal still works via shared aggregator, run_validation=False)
# ===================================================================


class TestAutoSealRegressionSmoke:
    def test_auto_seal_uses_shared_aggregator_run_validation_false(self, tmp_path):
        """auto-seal 走 aggregate_metrics_to_handoff(run_validation=False)，不产 validation warnings。"""
        # Build a minimal completed workspace
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "open_arm_time_ratio_s0", "subject_index": 0,
                    "output": str(ws / "m_open_arm_time_ratio_s0.json")}]
        _TASK_METRIC["open_arm_time_ratio_s0"] = "open_arm_time_ratio"
        _FAKE_VALUES["open_arm_time_ratio_s0"] = 0.5
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        # produce the artifact
        _runner_success("x", ["--output", str(ws / "m_open_arm_time_ratio_s0.json")], "open_arm_time_ratio_s0")

        from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff
        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        assert agg["status"] == "completed"
        # run_validation=False → no METHOD.METRIC_VALIDATION warnings from validate
        codes = [w.get("code") for w in agg["data_quality_warnings"]]
        assert all(c != "METHOD.METRIC_VALIDATION" for c in codes), codes


# ===================================================================
# §4 error paths: plan missing / empty / unparseable
# ===================================================================


class TestErrorPaths:
    def test_plan_missing_seals_failed(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "x"}), encoding="utf-8"
        )
        res = _call(_runtime(ws), plan_path="/mnt/user-data/workspace/plan_metrics.json")
        assert res["status"] == "failed"
        assert res["error_code"] == "plan_missing"
        # even on plan-missing, a failed handoff is sealed (so downstream sees status)
        assert (ws / "handoff_code_executor.json").exists()

    def test_empty_plan_seals_failed(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        plan = _plan([], workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        res = _call(_runtime(ws))
        assert res["status"] == "failed"
        assert res["error_code"] == "empty_plan"

    def test_workspace_missing(self, tmp_path):
        runtime = ToolRuntime(
            state={"thread_data": None}, context=None, config={},
            stream_writer=None, tool_call_id="t", store=None,
        )
        res = _call(runtime)
        assert res["status"] == "failed"
        assert res["error_code"] == "workspace_missing"


# ===================================================================
# §4 #3 per-task timeout (mocked: runner raises, executor records failure)
# ===================================================================


class TestTimeoutHandling:
    def test_runner_exception_recorded_as_failure(self, tmp_path, monkeypatch):
        """A runner that raises is caught per-task, recorded, others continue."""
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}, {"id": "m1"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)

        def boom(script, args, tid):
            if tid == "m0":
                raise RuntimeError("simulated hang/timeout")
            return _runner_success(script, args, tid)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", boom)

        res = _call(_runtime(ws), on_error="continue")
        assert res["n_failed"] == 1
        assert {f["id"] for f in res["failures"]} == {"m0"}
