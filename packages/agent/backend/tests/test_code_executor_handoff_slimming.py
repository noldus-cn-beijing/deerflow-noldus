"""Spec 2026-06-18 — code-executor handoff 瘦身测试（红→绿坐实，TDD）。

四个结构性改动：
1. task_context 从 ethoinsight 4 个 handoff schema 移除（死重量，下游不消费），
   seal 仅在 schema 仍声明该字段时注入（向前兼容通用 schema）。
2. statistics.outlier_diagnostics 拆到旁路 handoff_code_executor_outliers.json，
   主 handoff 留 outlier_diagnostics_ref + outlier_diagnostics_count。
3. output_files 完整路径列表拆到旁路 handoff_code_executor_outputs.json，
   主 handoff 留 output_files_ref + output_files_count，output_files 清空为 {}。
4. data-analyst / chart-maker / report-writer SKILL 删除禁用 bash 的 bundle 指引。

红线坐实：28-subject × 5-metric + 65 outlier 场景下，改前主 handoff > 50K（被
sandbox read_file 50K 截断线斩掉尾部 gate_signals / data_quality_warnings），
改后 < 50K 且尾部字段在首 50K 内。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

# Load the tool module fresh (mirrors test_run_metric_plan.py loading strategy —
# bypasses conftest sys.modules manipulation of deerflow.subagents.executor).
_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_metric_plan_tool.py"
)


def _load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.tools.builtins.run_metric_plan_tool_slim",
        _TOOL_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TOOL = _load_tool_module()

# Seal helper + schemas (imported normally — these don't close the executor cycle).
from deerflow.subagents.handoff_schemas import (  # noqa: E402
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
    ReportWriterHandoff,
)
from deerflow.tools.builtins.seal_handoff_tools import (  # noqa: E402
    _seal_handoff_to_workspace,
)

# 4 个 ethoinsight handoff 类，spec §3.2 要求全部移除 task_context。
_ETHOINSIGHT_HANDOFFS = [
    CodeExecutorHandoff,
    ChartMakerHandoff,
    DataAnalystHandoff,
    ReportWriterHandoff,
]

# sandbox read_file 全局截断线（config.yaml: read_file_output_max_chars）。
# 瘦身目标 = 主 handoff 在此线之下，尾部 fast-fail 字段单次可达。
READ_FILE_MAX_CHARS = 50000


# ---------------------------------------------------------------------------
# Fixtures + helpers（复用 test_run_metric_plan.py 的加载/调用模式）
# ---------------------------------------------------------------------------


def _runtime(workspace: Path) -> ToolRuntime:
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
    for m in metrics:
        if "output" not in m:
            m["output"] = str(workspace / f"m_{m['id']}.json")
        if "args" not in m:
            m["args"] = [
                "--input",
                (raw_files or ["/mnt/user-data/uploads/fake.txt"])[m.get("subject_index", 0)],
                "--output",
                m["output"],
            ]
        m.setdefault("script", "ethoinsight.scripts.fake.compute")
        m.setdefault("subject_index", 0)
    return {
        "schema_version": "1.1",
        "paradigm": "epm",
        "ev19_template": "epm",
        "generated_at": "2026-06-18T00:00:00",
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


def _write_plan(workspace: Path, plan: dict) -> None:
    (workspace / "plan_metrics.json").write_text(
        json.dumps(plan, ensure_ascii=False), encoding="utf-8"
    )
    (workspace / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "slim-test-config"}), encoding="utf-8"
    )


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "slim-test-config"}), encoding="utf-8"
    )
    return ws


def _call(runtime, **kwargs):
    return _TOOL.run_metric_plan_tool.func(runtime, **kwargs)


# ---------------------------------------------------------------------------
# 28-subject × 5-metric 场景构造（坐实改前 >50K 红线，验证改后 <50K 绿）
# ---------------------------------------------------------------------------

N_SUBJECTS = 28
N_METRICS = 5  # open_arm_time_ratio / total_distance / entries_open / time_closed / latency_open
_METRIC_NAMES = [
    "open_arm_time_ratio",
    "total_distance",
    "entries_open_arm",
    "time_in_closed_arm",
    "latency_to_enter_open",
]


def _big_metrics(workspace: Path) -> list[dict]:
    """140 metric rows = 28 subjects × 5 metrics → 140 m_*.json 产物路径。

    每条带 output_unit=ratio（catalog.resolve 在生产会填），value=0.5 ∈ [0,1] 合法，
    让 validate_catalog 不产 METRIC_VALIDATION 警告（避免测试场景的 data_quality_warnings
    失真膨胀，干扰主 handoff 体积断言——那是瘦身目标字段之外的噪声）。
    """
    metrics: list[dict] = []
    for s in range(N_SUBJECTS):
        for mi, name in enumerate(_METRIC_NAMES):
            metrics.append(
                {
                    "id": f"{name}_s{s}",
                    "subject_index": s,
                    "output": str(workspace / f"m_{name}_s{s}.json"),
                    "output_unit": "ratio",
                    "args": [
                        "--input",
                        f"/mnt/user-data/uploads/s{s}.txt",
                        "--output",
                        str(workspace / f"m_{name}_s{s}.json"),
                    ],
                    "script": "ethoinsight.scripts.epm.compute",
                }
            )
    return metrics


def _big_statistics_payload(outlier_count: int = 65) -> dict:
    """Statistics 脚本输出，含 outlier_count 条 outlier_diagnostics（每条 ~430 字符）。

    真实 dogfood 28-subject 场景产 65 条；此处构造等量体积的离群诊断数组。
    """
    outliers = [
        {
            "subject": f"subject #{i}",
            "metric": _METRIC_NAMES[i % N_METRICS],
            "group": "treatment" if i % 2 else "control",
            "value": 1234.5 + i,
            "group_mean": 500.0,
            "group_std": 80.0,
            "deviation": f"{(1234.5 + i - 500.0) / 80.0:.2f} SD",
            "rule": ">=1.5 SD",
            "counterfactual": (
                f"排除 subject #{i} 后 {_METRIC_NAMES[i % N_METRICS]} 组 mean 500.0→"
                f"{500.0 - i * 0.3:.2f}, std 80.0→{80.0 - i * 0.1:.2f}, "
                f"效应量 d 0.85→{0.85 - i * 0.005:.3f}（方向不变，结论稳健）"
            ),
        }
        for i in range(outlier_count)
    ]
    return {
        "test_used": "Mann-Whitney U (Shapiro-Wilk 拒绝正态)",
        "comparisons": [
            {
                "metric": name,
                "control_median": 0.30,
                "treatment_median": 0.15,
                "test_statistic": 42.0,
                "p_value": 0.03,
                "effect_size": 0.72,
                "n_control": 14,
                "n_treatment": 14,
            }
            for name in _METRIC_NAMES
        ],
        "summary": {"n_groups": 2, "n_per_group": 14, "normality": "rejected"},
        "outlier_diagnostics": outliers,
    }


def _big_runner(value_map: dict[str, float], stats_payload: dict):
    """Sync runner：metrics 写 m_*.json，statistics 写 stats output。

    替代 ProcessPoolExecutor fork/pickle；与 test_run_metric_plan.py 同模式。
    """

    def _r(script: str, args: list[str], task_id: str):
        out = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                out = args[i + 1]
        if task_id == "statistics":
            if out:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(stats_payload, ensure_ascii=False), encoding="utf-8")
            return (task_id, 0, "")
        # metric task
        payload = {
            "metric": _METRIC_NAMES[0] if not task_id else task_id.rsplit("_s", 1)[0],
            "value": value_map.get(task_id, 0.5),
            "parameters_used": {"velocity_threshold": 30.0},
        }
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(payload), encoding="utf-8")
        return (task_id, 0, "")

    return _r


def _build_big_handoff(tmp_path: Path, monkeypatch, outlier_count: int = 65) -> Path:
    """跑 run_metric_plan 产出大 handoff，返回 workspace。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    metrics = _big_metrics(ws)
    stats_payload = _big_statistics_payload(outlier_count=outlier_count)
    statistics = {
        "id": "epm_stats",
        "script": "ethoinsight.scripts.epm.run_groupwise_stats",
        "input": "/mnt/user-data/workspace/handoff_code_executor.json",
        "output": str(ws / "stats.json"),
        "skip_reason": None,
    }
    plan = _plan(
        metrics,
        workspace=ws,
        raw_files=[f"/mnt/user-data/uploads/s{s}.txt" for s in range(N_SUBJECTS)],
        groups_file="/mnt/user-data/workspace/groups.json",
        statistics=statistics,
    )
    _write_plan(ws, plan)

    value_map = {m["id"]: 0.1 + (i % 10) * 0.05 for i, m in enumerate(metrics)}
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _big_runner(value_map, stats_payload))
    _call(_runtime(ws), on_error="continue")
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", None)
    return ws


# ===================================================================
# §4.1 红线：主 handoff 体积 + 尾部字段可达性
# ===================================================================


class TestMainHandoffSizeAndTailReachability:
    def test_main_handoff_under_read_limit_with_28_subjects(self, tmp_path, monkeypatch):
        """28-subject × 5-metric + 65 outlier → 主 handoff < 50K（改前 ~85K）。"""
        ws = _build_big_handoff(tmp_path, monkeypatch, outlier_count=65)
        main = (ws / "handoff_code_executor.json").read_text(encoding="utf-8")
        assert len(main) < READ_FILE_MAX_CHARS, (
            f"主 handoff {len(main)} 字符 ≥ {READ_FILE_MAX_CHARS} 截断线，"
            f"data-analyst 单次 read_file 读不全。"
        )

    def test_gate_signals_and_dq_warnings_in_first_50k(self, tmp_path, monkeypatch):
        """gate_signals / data_quality_warnings 必须出现在首 50K 截断窗口内。"""
        ws = _build_big_handoff(tmp_path, monkeypatch, outlier_count=65)
        main = (ws / "handoff_code_executor.json").read_text(encoding="utf-8")
        window = main[:READ_FILE_MAX_CHARS]
        assert '"gate_signals"' in window, "gate_signals 被截断线斩在外面（data-analyst fast-fail 读不到）"
        assert '"data_quality_warnings"' in window, "data_quality_warnings 被截断线斩在外面"


# ===================================================================
# §4.2 旁路文件契约
# ===================================================================


class TestSidecarContracts:
    def test_outliers_sidecar_written_and_referenced(self, tmp_path, monkeypatch):
        """outlier_diagnostics 拆旁路：文件存在、内容完整、主 handoff 引用且不再内嵌。"""
        ws = _build_big_handoff(tmp_path, monkeypatch, outlier_count=65)
        sidecar = ws / "handoff_code_executor_outliers.json"
        assert sidecar.exists(), "旁路 outlier 文件未写"
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert len(sidecar_data) == 65

        main = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert main["outlier_diagnostics_count"] == 65
        ref = main["outlier_diagnostics_ref"]
        assert ref is not None
        assert ref.startswith("/mnt/user-data/workspace/"), f"ref 必须是虚拟路径，实际 {ref}"
        # 主 handoff 的 statistics 内不再内嵌完整 outlier_diagnostics
        stats = main.get("statistics", {})
        embedded = stats.get("outlier_diagnostics")
        assert not embedded, f"主 handoff 仍内嵌 outlier_diagnostics（{len(embedded)} 条），未拆旁路"

    def test_outputs_sidecar_written_and_referenced(self, tmp_path, monkeypatch):
        """output_files 140 条路径拆旁路，主 handoff 留 count + ref，output_files 清空。"""
        ws = _build_big_handoff(tmp_path, monkeypatch, outlier_count=65)
        sidecar = ws / "handoff_code_executor_outputs.json"
        assert sidecar.exists(), "旁路 outputs 文件未写"
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        # 140 条 metric 产物 + plan_metrics + groups（聚合器产物集）
        metrics_paths = sidecar_data.get("metrics", [])
        assert len(metrics_paths) == 140, f"期望 140 条产物路径，实际 {len(metrics_paths)}"

        main = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert main["output_files_count"] == 140
        ref = main["output_files_ref"]
        assert ref is not None and ref.startswith("/mnt/user-data/workspace/")
        # 主 handoff 的 output_files 清空（spec §3.2 默认：清空）
        assert main["output_files"] in ({}, None), (
            f"主 handoff output_files 应清空，实际 {main['output_files']}"
        )

    def test_no_outliers_no_sidecar(self, tmp_path, monkeypatch):
        """outlier_diagnostics 为空时不写旁路文件，count=0，ref=None。"""
        ws = _build_big_handoff(tmp_path, monkeypatch, outlier_count=0)
        sidecar = ws / "handoff_code_executor_outliers.json"
        assert not sidecar.exists(), "无离群时不应写旁路文件"
        main = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert main["outlier_diagnostics_count"] == 0
        assert main["outlier_diagnostics_ref"] is None


# ===================================================================
# §4.3 task_context 移除 + 不破坏读取方
# ===================================================================


class TestTaskContextRemoval:
    @pytest.mark.parametrize("model_cls", _ETHOINSIGHT_HANDOFFS)
    def test_task_context_not_injected_for_ethoinsight_handoffs(self, model_cls, tmp_path):
        """4 个 ethoinsight handoff seal 后产出 JSON 不含 task_context 键。"""
        ws = _make_workspace(tmp_path)
        # 各 schema 的 completed 校验要求不同核心产物，分别给最小合法 payload。
        payload = {
            CodeExecutorHandoff: {
                "status": "completed",
                "summary": "slim test",
                "paradigm": "epm",
                "metrics_summary": {"control": {"m": {"mean": 1.0}}},
            },
            ChartMakerHandoff: {
                "status": "completed",
                "summary": "slim test",
                "paradigm": "epm",
                "chart_files": ["/mnt/user-data/outputs/x.png"],
            },
            DataAnalystHandoff: {
                "status": "completed",
                "summary": "slim test",
                "paradigm": "epm",
                "key_findings": ["finding one"],
            },
            ReportWriterHandoff: {
                "status": "completed",
                "summary": "slim test",
                "paradigm": "epm",
                "report_path": "/mnt/user-data/outputs/report.md",
                "sections_written": ["summary"],
            },
        }[model_cls]
        _seal_handoff_to_workspace(model_cls, _filename_for(model_cls), payload, ws)
        raw = (ws / _filename_for(model_cls)).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "task_context" not in data, (
            f"{model_cls.__name__} 产出仍含 task_context（死重量未移除）"
        )

    def test_task_context_still_injected_for_schemas_that_declare_it(self, tmp_path):
        """通用 schema 仍声明 task_context 时 seal 仍注入（条件化未误伤通用路径）。"""
        from pydantic import BaseModel, ConfigDict

        from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace

        class GenericHandoffWithTaskContext(BaseModel):
            model_config = ConfigDict(extra="allow")
            status: str = "completed"
            summary: str = "generic"
            paradigm: str = "x"
            analysis_config_id: str = "cfg"
            task_context: dict | None = None

        ws = _make_workspace(tmp_path)
        _seal_handoff_to_workspace(
            GenericHandoffWithTaskContext,
            "generic_handoff.json",
            {"status": "completed", "summary": "generic", "paradigm": "x"},
            ws,
        )
        data = json.loads((ws / "generic_handoff.json").read_text(encoding="utf-8"))
        assert "task_context" in data, "声明 task_context 的通用 schema 应仍被注入"

    def test_old_handoff_with_task_context_still_parses(self):
        """旧 handoff（带 task_context）实例化新 CodeExecutorHandoff 不抛（向前兼容）。"""
        old = {
            "status": "completed",
            "summary": "legacy",
            "paradigm": "epm",
            "analysis_config_id": "legacy-cfg",
            "metrics_summary": {"control": {"m": {"mean": 1.0}}},
            "task_context": {
                "file_changes": ["/mnt/user-data/workspace/m_x.json"],
                "verify_commands": ["ls /mnt/user-data/workspace/m_x.json"],
                "failed_paths": [],
                "pending_items": [],
            },
        }
        # extra="allow" 吞掉多余字段，不应抛。
        obj = CodeExecutorHandoff(**old)
        assert obj.status == "completed"


def _filename_for(model_cls) -> str:
    """Map handoff class → its conventional seal filename."""
    return {
        CodeExecutorHandoff: "handoff_code_executor.json",
        ChartMakerHandoff: "handoff_chart_maker.json",
        DataAnalystHandoff: "handoff_data_analyst.json",
        ReportWriterHandoff: "handoff_report_writer.json",
    }[model_cls]


# ===================================================================
# §4.4 SKILL 契约（静态断言，防回归）
# ===================================================================


class TestSubagentSkillNoBashBundle:
    """三个禁用 bash 的 subagent，SKILL 不得含用 bash 拼 bundle 的指引。"""

    @pytest.mark.parametrize(
        "module_name,config_attr",
        [
            ("data_analyst", "DATA_ANALYST_CONFIG"),
            ("chart_maker", "CHART_MAKER_CONFIG"),
            ("report_writer", "REPORT_WRITER_CONFIG"),
        ],
    )
    def test_no_bash_bundle_for_banned_bash(self, module_name, config_attr):
        import importlib

        mod = importlib.import_module(f"deerflow.subagents.builtins.{module_name}")
        config = getattr(mod, config_attr)
        disallowed = getattr(config, "disallowed_tools", []) or []
        system_prompt = getattr(config, "system_prompt", "") or ""
        if "bash" in disallowed:
            # 用 bash 拼 bundle 的契约断裂串（spec §3.4/§3.5）。
            forbidden_markers = [
                "context_bundle.txt",  # da_context_bundle / cm_context_bundle / rw_context_bundle
                "bash cat /mnt/user-data/workspace/handoff",
            ]
            for marker in forbidden_markers:
                assert marker not in system_prompt, (
                    f"{module_name} 禁用了 bash 但 SKILL 仍含用 bash 拼 bundle 的指引 "
                    f"（{marker!r}）—— 契约断裂，指引从未能执行。"
                )
