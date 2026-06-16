"""Spec 2026-06-15 — run_metric_plan Step8 validation env 修复（#5）测试锚点。

根因（已逐层坐实，见 docs/superpowers/specs/2026-06-15-run-metric-plan-step8-validation-env-spec.md）：
  - run_metric_plan_tool.py Step8 ``aggregate_metrics_to_handoff(plan, workspace,
    run_validation=True)`` 在父进程裸跑（没包 ``_scoped_path_env``）。
  - validation 子步 ``validate_plan_results`` 经 ``resolve_sandbox_path`` 读 plan 的
    ``/mnt/user-data/workspace/m_*.json`` 虚拟路径，需 ``DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE``
    env 才能翻成真实路径；env 缺失 → fail-safe 原样返回 /mnt（不存在）→ OSError →
    每个指标记一条 ``result_file_unreadable`` critical → 毒化下游 data-analyst fast-fail。
  - 而 aggregate 主体用 ``workspace.glob`` 读真实路径、不需 env → 数据齐全，于是
    出现「数据在但全标 unreadable」的自相矛盾症状。

修复（治本、单点、与 statistics runner 同款）：Step8 调用包进 ``_scoped_path_env(path_env)``。

锚点设计（直接打 ``aggregate_metrics_to_handoff`` 纯函数，不碰进程池，与既有
test_run_metric_plan.py 同层级）：
  - 锚点 1（red→green）：plan output 用 /mnt 虚拟路径、真实文件在 real_workspace、
    ``DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`` 设为 real_workspace（模拟
    ``_scoped_path_env`` 包住的效果）→ aggregate 后**不应**有 result_file_unreadable。
    修复前：Step8 不包 env → 误报；修复后：包 env → 0 误报。（本测试直接在 env 已设的
    前提下调 aggregate，锁定「validation 子步拿到 env 就读得到真实文件」这一语义。）
  - 锚点 2（守护，证明 env 是 load-bearing）：删 env（模拟修复前 Step8 裸跑）→
    确实误报 result_file_unreadable，证明根因在 env 缺失而非 validate 逻辑。

守 feedback_worktree_shares_main_venv_editable_link：用 importlib 加载 worktree 源，
不依赖 editable link（否则测的是主仓代码、worktree 改动假绿）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

# Load the aggregation module fresh from this worktree's source (守 editable-link 铁律).
_AGG_FILE = Path(__file__).resolve().parents[1] / "packages" / "harness" / "deerflow" / "subagents" / "metric_aggregation.py"


def _load_aggregate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.metric_aggregation_real",
        _AGG_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_AGG = _load_aggregate_module()
aggregate_metrics_to_handoff = _AGG.aggregate_metrics_to_handoff


# Workspace env var consumed by resolve_sandbox_path for /mnt/user-data/workspace/*.
_WS_ENV = "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE"
_VIRTUAL_DIR = "/mnt/user-data/workspace"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metric_file(workspace: Path, fname: str, metric_name: str, value: float) -> None:
    """Write an m_*.json result payload (the shape scripts' save_output_json produces)."""
    (workspace / fname).write_text(
        json.dumps({"metric": metric_name, "value": value}),
        encoding="utf-8",
    )


def _build_plan(workspace: Path) -> dict:
    """Plan whose metrics[].output are /mnt virtual paths; real files live in ``workspace``.

    Two subjects × one ratio metric (open_arm_time_ratio, output_unit=ratio ∈ [0,1]).
    Both subjects' real files are written to ``workspace`` under matching basenames so
    aggregate主体 (workspace.glob) and validation 子步 (resolve_sandbox_path on the
    virtual output) converge on the same real files when the env is set.
    """
    return {
        "paradigm": "epm",
        "ev19_template": "elevated_plus_maze",
        "inputs": {
            "raw_files": [
                "/mnt/user-data/uploads/subject_control.csv",
                "/mnt/user-data/uploads/subject_treatment.csv",
            ],
            "groups_file": f"{_VIRTUAL_DIR}/groups.json",
        },
        "metrics": [
            {
                "id": "open_arm_time_ratio",
                "output": f"{_VIRTUAL_DIR}/m_open_arm_time_ratio__s0.json",
                "output_unit": "ratio",
                "subject_index": 0,
            },
            {
                "id": "open_arm_time_ratio",
                "output": f"{_VIRTUAL_DIR}/m_open_arm_time_ratio__s1.json",
                "output_unit": "ratio",
                "subject_index": 1,
            },
        ],
    }


def _seed_workspace(workspace: Path) -> None:
    """Write the real m_*.json + groups.json the plan expects (values pass ratio [0,1])."""
    _write_metric_file(workspace, "m_open_arm_time_ratio__s0.json", "open_arm_time_ratio", 0.099)
    _write_metric_file(workspace, "m_open_arm_time_ratio__s1.json", "open_arm_time_ratio", 0.198)
    (workspace / "groups.json").write_text(
        json.dumps(
            {
                "/mnt/user-data/uploads/subject_control.csv": "control",
                "/mnt/user-data/uploads/subject_treatment.csv": "treatment",
                "subject_control.csv": "control",
                "subject_treatment.csv": "treatment",
            }
        ),
        encoding="utf-8",
    )


def _unreadable_warnings(agg: dict) -> list[dict]:
    """Pull the result_file_unreadable data_quality_warnings out of an aggregate result."""
    return [w for w in agg.get("data_quality_warnings", []) if "result_file_unreadable" in str(w.get("issue", "")) or "result_file_unreadable" in str(w.get("message", ""))]


# ---------------------------------------------------------------------------
# End-to-end anchor through the real run_metric_plan tool entrypoint
# ---------------------------------------------------------------------------

# Load the run_metric_plan tool module fresh (same strategy as the existing
# test_run_metric_plan.py — bypasses conftest's executor mock, measures this
# worktree's source per the editable-link 假绿 铁律).
_TOOL_FILE = Path(__file__).resolve().parents[1] / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_metric_plan_tool.py"


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


def _runtime_with_workspace(workspace: Path):
    """ToolRuntime whose thread_data maps /mnt/user-data/workspace → real workspace.

    `_build_path_env(thread_data)` reads workspace_path and emits
    DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE=<workspace>, so when Step 8 is wrapped in
    `_scoped_path_env(path_env)` the validation 子步 can resolve /mnt virtual paths.
    """
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(workspace / "_uploads"),
                "outputs_path": str(workspace / "_outputs"),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _runner_writes_to_workspace(real_workspace: Path, metric_for_task: dict[str, str]):
    """Runner that writes the real m_*.json into ``real_workspace`` regardless of the
    virtual --output it receives.

    Production compute scripts run inside the process pool whose workers carry the
    DEERFLOW_PATH_* env (set by _worker_init), so save_output_json(virtual_path)
    resolves to the real workspace. In this synchronous injected runner we don't
    have that env at Step 6 (it's only scoped at Step 7/8), so we map the virtual
    output → real file via basename. This keeps Step 6 deterministically producing
    the real artifacts (the aggregation主体 reads them via workspace.glob) and lets
    the Step 8 validation 子步 be the thing under test.
    """

    def _runner(script: str, args: list[str], task_id: str):
        out = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                out = args[i + 1]
        metric_name = metric_for_task.get(task_id, task_id)
        fname = Path(out).name if out else f"m_{task_id}.json"
        (real_workspace / fname).write_text(json.dumps({"metric": metric_name, "value": 0.198}), encoding="utf-8")
        return (task_id, 0, "")

    return _runner


def test_run_metric_plan_step8_no_false_unreadable(tmp_path, monkeypatch):
    """端到端锚点：plan output 用 /mnt 虚拟路径 + Step6 写真实文件 → run_metric_plan
    落盘的 handoff **不应**有 result_file_unreadable（修复后绿、修复前红）。

    这是 spec §5「回退验证 red→green」的端到端实现：去掉 Step8 的
    ``with _scoped_path_env(path_env):``（恢复裸跑）→ 本测试应红（handoff 出现
    result_file_unreadable）；包回去 → 绿。直接钉死 Step8 必须包 env 这一修复点。
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # plan 用 /mnt 虚拟 output；真实文件由注入 runner 写进 workspace。
    plan = {
        "schema_version": "1.1",
        "paradigm": "epm",
        "ev19_template": "epm",
        "generated_at": "2026-06-15T00:00:00",
        "inputs": {
            "raw_files": ["/mnt/user-data/uploads/s.csv"],
            "groups_file": "/mnt/user-data/workspace/groups.json",
            "columns_file": None,
        },
        "metrics": [
            {
                "id": "open_arm_time_ratio_s0",
                "subject_index": 0,
                "output_unit": "ratio",
                "output": "/mnt/user-data/workspace/m_open_arm_time_ratio_s0.json",
                "script": "ethoinsight.scripts.fake.compute",
                "args": [
                    "--input",
                    "/mnt/user-data/uploads/s.csv",
                    "--output",
                    "/mnt/user-data/workspace/m_open_arm_time_ratio_s0.json",
                ],
            }
        ],
        "statistics": {"skip_reason": "single subject"},
        "skipped": [],
        "notes": [],
    }
    # groups.json 落到真实 workspace（aggregation 主体读它做分组）。
    (workspace / "groups.json").write_text(
        json.dumps({"/mnt/user-data/uploads/s.csv": "control", "s.csv": "control"}),
        encoding="utf-8",
    )
    # plan_metrics.json 落到真实 workspace（Step2 经 replace_virtual_path 读它）。
    (workspace / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
    (workspace / "experiment-context.json").write_text(json.dumps({"analysis_config_id": "test-config-step8"}), encoding="utf-8")

    # 注入同步 runner：把真实文件写进 workspace（Step6 产出）。
    monkeypatch.setattr(
        _TOOL,
        "_TASK_RUNNER_OVERRIDE",
        _runner_writes_to_workspace(workspace, {"open_arm_time_ratio_s0": "open_arm_time_ratio"}),
    )

    runtime = _runtime_with_workspace(workspace)
    result = _TOOL.run_metric_plan_tool.func(runtime, plan_path="/mnt/user-data/workspace/plan_metrics.json")
    # restore
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", None)

    assert result["status"] == "completed", f"run_metric_plan failed: {result}"

    handoff = json.loads((workspace / "handoff_code_executor.json").read_text(encoding="utf-8"))
    unreadable = [w for w in handoff.get("data_quality_warnings", []) if "result_file_unreadable" in str(w.get("issue", "")) or "result_file_unreadable" in str(w.get("message", ""))]
    assert unreadable == [], f"Step8 应包 _scoped_path_env 使 validation 读到真实文件，却误报 result_file_unreadable: {unreadable}"


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def test_aggregate_validation_with_scoped_env_no_false_unreadable(tmp_path, monkeypatch):
    """锚点 1：env 设好（模拟 Step8 包 _scoped_path_env）→ 不应误报 result_file_unreadable。

    修复前 Step8 在父进程裸跑、不包 env → resolve_sandbox_path 读 /mnt 失败 → 误报；
    修复后 Step8 包 ``_scoped_path_env(path_env)`` → resolve 拿到 env → 读到真实文件。
    本测试直接设该 env（等价于 _scoped_path_env 块内可见的 env），锁定「validation
    子步拿到 env 就读得到真实文件、无误报」这一语义——修复的价值正是让 Step8 处于
    此 env scope 内。
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _seed_workspace(workspace)
    plan = _build_plan(workspace)

    # 模拟 _scoped_path_env(path_env) 块内的可见 env：把 /mnt/user-data/workspace 翻向真实 workspace。
    monkeypatch.setenv(_WS_ENV, str(workspace))

    agg = aggregate_metrics_to_handoff(plan, workspace, run_validation=True)

    unreadable = _unreadable_warnings(agg)
    assert unreadable == [], f"env 已设却误报 result_file_unreadable: {unreadable}"
    # 主体也应成功读到真实文件（数据齐全）——坐实「数据在但误标 unreadable」的不对称症状根源。
    assert agg["status"] == "completed", f"expected completed, got {agg['status']}: {agg.get('errors')}"
    assert agg["n_present"] == 2


def test_aggregate_validation_without_env_does_report_unreadable(tmp_path, monkeypatch):
    """锚点 2（守护）：env 缺失（模拟修复前 Step8 裸跑）→ 确实误报 result_file_unreadable。

    证明根因在 env 缺失、而非 validate 逻辑本身——validate 给它 env 就能 resolve。
    这条与锚点 1 共同钉死：有 env→无误报、无 env→误报，``_scoped_path_env`` 是 load-bearing。
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _seed_workspace(workspace)
    plan = _build_plan(workspace)

    # 修复前 Step8 的真实状态：父进程没包 _scoped_path_env，env 不在。
    monkeypatch.delenv(_WS_ENV, raising=False)

    agg = aggregate_metrics_to_handoff(plan, workspace, run_validation=True)

    unreadable = _unreadable_warnings(agg)
    assert unreadable, "无 env 时应误报 result_file_unreadable（证明 env 是修复关键）"
    # 主体仍成功（glob 真实路径不依赖 env）——复现「数据齐全却全标 unreadable」的矛盾症状。
    assert agg["status"] == "completed"
    assert agg["n_present"] == 2
