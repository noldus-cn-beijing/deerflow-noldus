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

        _call(_runtime(ws))  # 副作用：落盘 handoff（下方读文件 + 直调聚合器断言，不用返回值）

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

    def test_aggregator_output_matches_expected_payload(self, tmp_path):
        """§4 #6（强化）：直接钉死聚合器对已知 m_*.json 的**逐字段**输出，而非只测确定性。

        测试 #6 的 spec 本意是"抽出复用无行为变化"。仅断言 direct1==direct2 只证纯函数
        确定性，不证语义正确。这里给定已知 value/group/subject，断言聚合器产出的
        metrics_summary / per_subject / status / 完整性字段逐一符合预期——任何聚合逻辑
        漂移（分组映射、MetricStat 形状、completed 判据）都会在这里响亮失败。
        """
        from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff

        ws = tmp_path / "ws"
        ws.mkdir()
        # 两 subject 同 metric，分属两组；写真实 m_*.json + groups.json。
        metrics = [
            {"id": "open_arm_time_ratio_s0", "subject_index": 0, "output": str(ws / "m_open_arm_time_ratio_s0.json")},
            {"id": "open_arm_time_ratio_s1", "subject_index": 1, "output": str(ws / "m_open_arm_time_ratio_s1.json")},
        ]
        raw_files = ["/mnt/user-data/uploads/ctrl.txt", "/mnt/user-data/uploads/drug.txt"]
        plan = _plan(metrics, workspace=ws, raw_files=raw_files)
        (ws / "m_open_arm_time_ratio_s0.json").write_text(
            json.dumps({"metric": "open_arm_time_ratio", "value": 0.30, "parameters_used": {"open_arm_zones": ["open"]}}),
            encoding="utf-8",
        )
        (ws / "m_open_arm_time_ratio_s1.json").write_text(
            json.dumps({"metric": "open_arm_time_ratio", "value": 0.55, "parameters_used": {"open_arm_zones": ["open"]}}),
            encoding="utf-8",
        )
        (ws / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/ctrl.txt": "Control", "/mnt/user-data/uploads/drug.txt": "Treatment"}),
            encoding="utf-8",
        )

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)

        assert agg["status"] == "completed"
        assert agg["n_total"] == 2
        assert agg["n_present"] == 2
        assert agg["missing_expected"] == []
        # 逐字段 metrics_summary：分组映射 + MetricStat 形状 + list zone 参数透传。
        assert agg["metrics_summary"] == {
            "Control": {"open_arm_time_ratio": {"mean": 0.30, "std": None, "n": 1, "parameters_used": {"open_arm_zones": ["open"]}}},
            "Treatment": {"open_arm_time_ratio": {"mean": 0.55, "std": None, "n": 1, "parameters_used": {"open_arm_zones": ["open"]}}},
        }
        # per_subject 按 subject stem 命名，值逐字段。
        assert agg["per_subject"] == {
            "ctrl": {"open_arm_time_ratio": 0.30},
            "drug": {"open_arm_time_ratio": 0.55},
        }
        # spec 2026-06-30 C1：聚合器额外返回 subject_groups（subject stem → 组），
        # 供指标表导出器复用同一 subject→group 推导（SSOT，不在导出器重推致漂移）。
        assert agg["subject_groups"] == {"ctrl": "Control", "drug": "Treatment"}


def test_aggregate_returns_subject_groups_on_empty_plan(tmp_path):
    """spec 2026-06-30 C1：空 plan / 无 output 早退路径也带 subject_groups={} 键（恒在）。"""
    from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff

    ws = tmp_path / "ws"
    ws.mkdir()
    # 无 metrics[] → 早退 1。
    agg_empty = aggregate_metrics_to_handoff({"paradigm": "epm"}, ws, run_validation=False)
    assert agg_empty["subject_groups"] == {}

    # metrics 有 id 但无 output → 早退 2。
    agg_no_output = aggregate_metrics_to_handoff(
        {"paradigm": "epm", "metrics": [{"id": "x", "subject_index": 0}]}, ws, run_validation=False
    )
    assert agg_no_output["subject_groups"] == {}


# ===================================================================
# spec 2026-06-30 C1 模块1：Step 9.6 确定性导出指标结果表到 outputs/
# ===================================================================


class TestMetricsTableExport:
    """run_metric_plan 完成后确定性写 metrics_table.csv + .json 到 outputs/。

    导出是 best-effort UI 产物：失败不得中断 run（handoff 已 sealed）。
    """

    def test_run_metric_plan_writes_metrics_table_to_outputs(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [
            {"id": "open_arm_time_ratio_s0", "subject_index": 0},
            {"id": "open_arm_time_ratio_s1", "subject_index": 1},
        ]
        _TASK_METRIC.update({"open_arm_time_ratio_s0": "open_arm_time_ratio",
                             "open_arm_time_ratio_s1": "open_arm_time_ratio"})
        _FAKE_VALUES.update({"open_arm_time_ratio_s0": 0.30, "open_arm_time_ratio_s1": 0.55})
        plan = _plan(
            metrics, workspace=ws,
            raw_files=["/mnt/user-data/uploads/ctrl.txt", "/mnt/user-data/uploads/drug.txt"],
        )
        (ws / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/ctrl.txt": "Control",
                        "/mnt/user-data/uploads/drug.txt": "Treatment"}),
            encoding="utf-8",
        )
        _write_plan(ws, plan)

        res = _call(_runtime(ws))

        assert res["status"] == "completed", res
        outputs = ws.parent / "outputs"
        csv_path = outputs / "metrics_table.csv"
        json_path = outputs / "metrics_table.json"
        assert csv_path.is_file() and json_path.is_file()

        # JSON 值匹配 handoff per_subject（SSOT，不双算）。
        h = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        j = json.loads(json_path.read_text(encoding="utf-8"))
        by_subj = {row["subject"]: row for row in j["per_subject"]}
        assert by_subj["ctrl"]["values"]["open_arm_time_ratio"] == h["per_subject"]["ctrl"]["open_arm_time_ratio"]
        assert by_subj["drug"]["values"]["open_arm_time_ratio"] == h["per_subject"]["drug"]["open_arm_time_ratio"]

        # JSON 不含内脏（反 vacuous —— 导出从 agg 读，构造保证）。
        viscera = {"gate_signals", "handoff", "assessment", "statistics", "confidence", "inputs", "sealed_by"}
        assert viscera.isdisjoint(j.keys())

    def test_run_metric_plan_continues_when_export_fails(self, tmp_path, monkeypatch):
        """导出器抛错 → run 仍 completed 且 handoff sealed（best-effort，不中断）。"""
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "open_arm_time_ratio_s0", "subject_index": 0}]
        _TASK_METRIC.update({"open_arm_time_ratio_s0": "open_arm_time_ratio"})
        _FAKE_VALUES.update({"open_arm_time_ratio_s0": 0.30})
        plan = _plan(metrics, workspace=ws)
        _write_plan(ws, plan)

        # 让导出器炸：monkeypatch 模块内的 export_metrics_table 抛错。
        import deerflow.subagents.metrics_table_export as export_mod

        def _boom(**_kw):
            raise RuntimeError("simulated export failure")

        monkeypatch.setattr(export_mod, "export_metrics_table", _boom)

        res = _call(_runtime(ws))

        assert res["status"] == "completed", res  # 不中断
        # handoff 仍 sealed（导出失败前已落盘）。
        assert (ws / "handoff_code_executor.json").is_file()
        # outputs 没有指标表（导出失败了）。
        assert not (ws.parent / "outputs" / "metrics_table.csv").is_file()


# ===================================================================
# Review-fix: env 不污染父进程 + _scoped_path_env 还原
# ===================================================================


class TestEnvNotGloballeyPolluted:
    """run_metric_plan 不得永久 mutate 父进程 os.environ 的 DEERFLOW_PATH_*。

    多线程 Gateway 下，一个 thread 的 run_metric_plan 若全局 mutate DEERFLOW_PATH_*，
    会泄漏成全局污染并发跑的其他 thread（潜在跨线程 workspace 路径错配）。worker env
    走 ProcessPoolExecutor initializer（子进程内），父进程 env 调用前后不变。
    """

    def test_tool_run_does_not_leave_path_env_in_parent(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        # 清掉可能预存的 DEERFLOW_PATH_* 以纯净观察
        for k in [k for k in os.environ if k.startswith("DEERFLOW_PATH_")]:
            monkeypatch.delenv(k, raising=False)
        before = {k: v for k, v in os.environ.items() if k.startswith("DEERFLOW_PATH_")}

        metrics = [{"id": "m0"}]
        plan = _plan(metrics, workspace=ws, raw_files=["/mnt/user-data/uploads/s0.txt"])
        _write_plan(ws, plan)
        _call(_runtime(ws))  # sync runner path（autouse fixture）

        after = {k: v for k, v in os.environ.items() if k.startswith("DEERFLOW_PATH_")}
        assert after == before, f"run_metric_plan 泄漏了 DEERFLOW_PATH_* 到父进程: {set(after) - set(before)}"

    def test_scoped_path_env_restores_on_exit(self, monkeypatch):
        """_scoped_path_env：with 块内可见，退出后还原（新增键删除、原有键复原）。"""
        monkeypatch.delenv("DEERFLOW_PATH_WORKSPACE", raising=False)
        monkeypatch.setenv("DEERFLOW_PATH_EXISTING", "old")

        with _TOOL._scoped_path_env({"DEERFLOW_PATH_WORKSPACE": "/w", "DEERFLOW_PATH_EXISTING": "new"}):
            assert os.environ["DEERFLOW_PATH_WORKSPACE"] == "/w"
            assert os.environ["DEERFLOW_PATH_EXISTING"] == "new"

        # 新增键退出后删除；原有键复原
        assert "DEERFLOW_PATH_WORKSPACE" not in os.environ
        assert os.environ["DEERFLOW_PATH_EXISTING"] == "old"


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


# ===================================================================
# Spec 2026-06-16 statistics 路径列对齐：plan.statistics.parameters 透传成
# run_groupwise_stats 的 --parameters-json argv（harness 侧接线坐实）。
# ===================================================================


class TestStatisticsParametersPropagation:
    """run_metric_plan Step7 把 plan.statistics.parameters 序列化进 stats_argv。"""

    def test_statistics_argv_carries_parameters_json(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}]
        zone_params = {"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}
        statistics = {
            "id": "epm_stats",
            "script": "ethoinsight.scripts.epm.run_groupwise_stats",
            "input": "/mnt/user-data/workspace/handoff_code_executor.json",
            "output": str(ws / "stats.json"),
            "skip_reason": None,
            "parameters": zone_params,
        }
        plan = _plan(
            metrics,
            workspace=ws,
            raw_files=["/mnt/user-data/uploads/s0.txt"],
            groups_file="/mnt/user-data/workspace/groups.json",
            statistics=statistics,
        )
        _write_plan(ws, plan)

        captured: dict = {}

        def capture_runner(script, args, tid):
            if tid == "statistics":
                captured["args"] = list(args)
            return _runner_success(script, args, tid)

        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", capture_runner)
        _call(_runtime(ws), on_error="continue")

        assert "args" in captured, "statistics task 未被捕获执行"
        argv = captured["args"]
        assert "--parameters-json" in argv, f"stats_argv 应含 --parameters-json，实际 {argv}"
        idx = argv.index("--parameters-json")
        payload = json.loads(argv[idx + 1])
        assert payload == zone_params, f"--parameters-json 内容应为 zone 参数，实际 {payload}"

    def test_statistics_argv_empty_parameters_when_none(self, tmp_path, monkeypatch):
        """无 column_aliases → plan.statistics.parameters 缺失 → --parameters-json={}（向后兼容）。"""
        ws = tmp_path / "ws"
        ws.mkdir()
        metrics = [{"id": "m0"}]
        statistics = {
            "id": "epm_stats",
            "script": "ethoinsight.scripts.epm.run_groupwise_stats",
            "input": "/mnt/user-data/workspace/handoff_code_executor.json",
            "output": str(ws / "stats.json"),
            "skip_reason": None,
            # 无 parameters 字段（旧 plan 兼容）
        }
        plan = _plan(
            metrics,
            workspace=ws,
            raw_files=["/mnt/user-data/uploads/s0.txt"],
            groups_file="/mnt/user-data/workspace/groups.json",
            statistics=statistics,
        )
        _write_plan(ws, plan)

        captured: dict = {}

        def capture_runner(script, args, tid):
            if tid == "statistics":
                captured["args"] = list(args)
            return _runner_success(script, args, tid)

        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", capture_runner)
        _call(_runtime(ws), on_error="continue")

        argv = captured["args"]
        idx = argv.index("--parameters-json")
        assert json.loads(argv[idx + 1]) == {}


# ===================================================================
# Spec 2026-06-17 P2 statistics 降级信号：gate_signals.statistics_status
# 三态（ok / crashed / absent_by_design）。修复前 statistics 崩溃静默成空，
# 下游无法区分"崩了"vs"本就无统计"。本组坐实三态可机读 + status 可区分。
# ===================================================================


def _stats_plan(tmp_path, *, skip_reason=None, parameters=None):
    """Build a one-metric plan with a statistics segment for the status tests."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    statistics = {
        "id": "epm_stats",
        "script": "ethoinsight.scripts.epm.run_groupwise_stats",
        "input": "/mnt/user-data/workspace/handoff_code_executor.json",
        "output": str(ws / "stats.json"),
        "skip_reason": skip_reason,
    }
    if parameters is not None:
        statistics["parameters"] = parameters
    plan = _plan(
        [{"id": "m0"}],
        workspace=ws,
        raw_files=["/mnt/user-data/uploads/s0.txt"],
        groups_file="/mnt/user-data/workspace/groups.json",
        statistics=statistics,
    )
    _write_plan(ws, plan)
    return ws


def _read_handoff_gate_signals(ws):
    """Read the sealed handoff_code_executor.json and return its gate_signals dict."""
    handoff_path = ws / "handoff_code_executor.json"
    assert handoff_path.exists(), "handoff 未落盘"
    data = json.loads(handoff_path.read_text(encoding="utf-8"))
    return data["gate_signals"]


class TestStatisticsStatus:
    """gate_signals.statistics_status 三态可机读信号（P2 信号源）。"""

    def test_statistics_ok_when_runner_succeeds(self, tmp_path, monkeypatch):
        """statistics 段无 skip_reason + runner 成功 + 非空产物 → statistics_status=ok。"""
        ws = _stats_plan(tmp_path, skip_reason=None)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success)
        res = _call(_runtime(ws), on_error="continue")

        assert res["gate_signals"]["statistics_status"] == "ok"
        assert res["gate_signals"]["statistics_error"] is None
        # compute 全成功 → status 仍是 completed（ok 不降级）
        assert res["status"] == "completed"
        # handoff 落盘一致
        gf = _read_handoff_gate_signals(ws)
        assert gf["statistics_status"] == "ok"

    def test_statistics_crashed_when_runner_fails(self, tmp_path, monkeypatch):
        """statistics runner rc=1 → statistics_status=crashed + 带 error；status 降为 partial。"""
        ws = _stats_plan(tmp_path, skip_reason=None)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"statistics"}))
        res = _call(_runtime(ws), on_error="continue")

        assert res["gate_signals"]["statistics_status"] == "crashed"
        assert "planned failure" in res["gate_signals"]["statistics_error"]
        # compute 全成功但 statistics 崩 → 降为 partial（不加新枚举，复用三态）
        assert res["status"] == "partial"
        # failures 仍记 statistics 那笔
        assert any(f["id"] == "statistics" for f in res["failures"])

    def test_statistics_absent_by_design_when_skip_reason(self, tmp_path, monkeypatch):
        """statistics 段有 skip_reason（单组/单样本）→ absent_by_design；status 不受影响。"""
        ws = _stats_plan(tmp_path, skip_reason="single sample, n<2")
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success)
        res = _call(_runtime(ws), on_error="continue")

        assert res["gate_signals"]["statistics_status"] == "absent_by_design"
        assert res["gate_signals"]["statistics_error"] is None
        # 合理 skip 不降级（compute 全成 → completed）
        assert res["status"] == "completed"
        # absent_by_design 时 statistics 字段为空 dict（schema 默认值），但 statistics_status 信号
        # 明确标记 absent_by_design——这正是 P2 的核心：用信号区分"空"的成因（崩溃 vs 设计内缺席）。
        handoff = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert handoff["statistics"] == {}

    def test_statistics_crashed_distinct_from_absent(self, tmp_path, monkeypatch):
        """crashed 与 absent_by_design 的 statistics_status 字面量不同（可机读区分）。"""
        # crashed
        ws_crashed = _stats_plan(tmp_path, skip_reason=None)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_fail_on({"statistics"}))
        res_crashed = _call(_runtime(ws_crashed), on_error="continue")

        # absent_by_design（独立 tmp_path）
        ws_absent = _stats_plan(tmp_path / "absent", skip_reason="single sample")
        # absent 时 runner 不被调，override 不影响
        res_absent = _call(_runtime(ws_absent), on_error="continue")

        assert res_crashed["gate_signals"]["statistics_status"] == "crashed"
        assert res_absent["gate_signals"]["statistics_status"] == "absent_by_design"
        # 关键：两者字节不再相同（crashed 带 error，absent 带 skip 语义）
        assert res_crashed["gate_signals"]["statistics_status"] != res_absent["gate_signals"]["statistics_status"]
        assert res_crashed["gate_signals"]["statistics_error"] is not None
        assert res_absent["gate_signals"]["statistics_error"] is None

    def test_statistics_empty_payload_treated_as_crashed(self, tmp_path, monkeypatch):
        """runner rc=0 但写空产物 → crashed（防脚本半成功：rc=0 却没写真东西）。"""
        ws = _stats_plan(tmp_path, skip_reason=None)

        def empty_stats_runner(script, args, tid):
            # statistics task: rc=0 但写空 dict（半成功）
            if tid == "statistics":
                out = None
                for i, a in enumerate(args):
                    if a == "--output" and i + 1 < len(args):
                        out = args[i + 1]
                if out:
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    Path(out).write_text("{}", encoding="utf-8")
                return (tid, 0, "")
            return _runner_success(script, args, tid)

        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", empty_stats_runner)
        res = _call(_runtime(ws), on_error="continue")

        assert res["gate_signals"]["statistics_status"] == "crashed"

    def test_statistics_declared_but_missing_script_is_crashed(self, tmp_path, monkeypatch):
        """statistics 段 skip_reason=None 但缺 script → crashed（残缺段不伪装 absent_by_design）。

        红线一静默降级口子：catalog/resolve 投影出残缺 statistics 段（声明应跑却无 script/output）
        时，必须 crashed 让熔断器接管，而非默认 absent_by_design 走正常 partial。
        """
        ws = tmp_path / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        # 直接构 plan：statistics 段 skip_reason=None 但 script 为空（残缺段）。
        plan = _plan(
            [{"id": "m0"}],
            workspace=ws,
            raw_files=["/mnt/user-data/uploads/s0.txt"],
            statistics={
                "id": "epm_stats",
                "script": "",  # 残缺：声明应跑（skip_reason=None）却无脚本
                "input": "/mnt/user-data/workspace/handoff_code_executor.json",
                "output": str(ws / "stats.json"),
                "skip_reason": None,
            },
        )
        _write_plan(ws, plan)
        monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner_success)
        res = _call(_runtime(ws), on_error="continue")

        assert res["gate_signals"]["statistics_status"] == "crashed"
        assert res["gate_signals"]["statistics_error"] is not None
        # compute 全成功但 statistics 段残缺 → 降为 partial（与 runner 崩溃一致对待）
        assert res["status"] == "partial"
