"""outlier 旁路 sidecar 主题解析 + OutlierFinding 对齐（spec 2026-06-18 §4.1/§4.2）。

data-analyst 曾在 thinking 里把 `subject #i` 逐条映射成真名 + 把数值 deviation 翻译成
定性串——撑爆 50K 撞 900s 超时。本批坐实下沉链路完整：statistics runner 产出的 outlier
（subject 已是文件名 stem、deviation 已是定性串、含 OutlierFinding 全部 5 字段）经
run_metric_plan_tool 旁路落盘后**原样透传**，data-analyst 可直接引用，无需任何 thinking 内
的映射/翻译/重组。

镜像 test_code_executor_handoff_slimming.py 的 importlib 加载策略（绕开 conftest 的 executor
sys.modules mock）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "run_metric_plan_tool.py"
)


def _load_tool_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.tools.builtins.run_metric_plan_tool_outlier_resolution",
        _TOOL_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TOOL = _load_tool_module()

from deerflow.subagents.handoff_schemas import OutlierFinding  # noqa: E402


def _runtime(workspace: Path) -> ToolRuntime:
    return ToolRuntime(
        state={"thread_data": {"workspace_path": str(workspace)}},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _call(runtime, **kwargs):
    return _TOOL.run_metric_plan_tool.func(runtime, **kwargs)


def _stem_outlier_statistics_payload() -> dict:
    """模拟真实 statistics runner 产出：outlier subject 已是文件名 stem + 定性 deviation。

    真实链路：epm/run_groupwise_stats.py → build_subject_label_map → compare_groups(
    subject_label_map=...) → compute_outlier_diagnostics 产出含 stem subject + 合成 deviation
    的条目。本 fixture 直接构造等价成品，验证 run_metric_plan_tool 旁路落盘原样透传。
    """
    return {
        "test_used": "Mann-Whitney U",
        "comparisons": [
            {
                "metric": "open_arm_time_ratio",
                "control_median": 0.10,
                "treatment_median": 0.30,
                "p_value": 0.03,
                "effect_size": 0.7,
            }
        ],
        "summary": {"n_groups": 2, "n_per_group": 4},
        "outlier_diagnostics": [
            {
                # subject 已是真实 stem（非 "subject #i"）——统计层预填
                "subject": "Raw data-EPM-Xuhui-Trial 3",
                "metric": "open_arm_time_ratio",
                "group": "control",
                "value": 1.0,
                # 定性 deviation 已合成（非裸数值）——统计层 _format_outlier_deviation 产出
                "deviation": "2.0x group median; 2.2 SD above mean",
                "deviation_sd": 2.2,
                "deviation_median_ratio": 2.0,
                "group_mean": 0.25,
                "group_std": 0.34,
                "group_median": 0.12,
                "loo_mean": 0.0001,
                "loo_std": 0.0001,
                # counterfactual 串内 subject 已是真名
                "counterfactual": "control mean 0.2530 → 0.0001 (std 0.3356 → 0.0001) if Raw data-EPM-Xuhui-Trial 3 excluded",
            },
            {
                "subject": "Raw data-EPM-Xuhui-Trial 7",
                "metric": "total_entry_count",
                "group": "treatment",
                "value": 1.0,
                "deviation": "extreme deviation; 1.8 SD below mean",
                "deviation_sd": 1.8,
                "deviation_median_ratio": 1000000000.0,
                "group_mean": 8.0,
                "group_std": 3.5,
                "group_median": 8.0,
                "loo_mean": 8.7,
                "loo_std": 3.0,
                "counterfactual": "treatment mean 8.0 → 8.7 if Raw data-EPM-Xuhui-Trial 7 excluded",
            },
        ],
    }


def _build_workspace_with_stem_outliers(tmp_path: Path, monkeypatch) -> Path:
    """跑 run_metric_plan 产出含 stem subject outlier 的 handoff + 旁路文件。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"analysis_config_id": "stem-test"}), encoding="utf-8"
    )

    metrics = [
        {
            "id": "open_arm_time_ratio_s0",
            "subject_index": 0,
            "output": str(ws / "m_open_s0.json"),
            "output_unit": "ratio",
            "args": ["--input", "/mnt/user-data/uploads/s0.txt", "--output", str(ws / "m_open_s0.json")],
            "script": "ethoinsight.scripts.epm.compute",
        }
    ]
    stats_payload = _stem_outlier_statistics_payload()
    statistics = {
        "id": "epm_stats",
        "script": "ethoinsight.scripts.epm.run_groupwise_stats",
        "input": "/mnt/user-data/workspace/handoff_code_executor.json",
        "output": str(ws / "stats.json"),
        "skip_reason": None,
    }
    plan = {
        "schema_version": "1.1",
        "paradigm": "epm",
        "ev19_template": "epm",
        "generated_at": "2026-06-18T00:00:00",
        "inputs": {
            "raw_files": ["/mnt/user-data/uploads/s0.txt"],
            "groups_file": "/mnt/user-data/workspace/groups.json",
            "columns_file": None,
        },
        "metrics": metrics,
        "statistics": statistics,
        "skipped": [],
        "notes": [],
    }
    (ws / "plan_metrics.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    def _runner(script, args, task_id):
        out = None
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                out = args[i + 1]
        if task_id == "statistics":
            if out:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(stats_payload, ensure_ascii=False), encoding="utf-8")
            return (task_id, 0, "")
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps({"metric": "open_arm_time_ratio", "value": 0.5, "output_unit": "ratio"}),
                encoding="utf-8",
            )
        return (task_id, 0, "")

    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", _runner)
    _call(_runtime(ws), on_error="continue")
    monkeypatch.setattr(_TOOL, "_TASK_RUNNER_OVERRIDE", None)
    return ws


class TestOutlierSidecarSubjectResolution:
    """旁路 sidecar 的 subject 已是真名 + deviation 是定性串 + OutlierFinding 字段齐备（§4.1/§4.2）。"""

    def test_sidecar_subject_is_real_stem_not_index(self, tmp_path, monkeypatch):
        """sidecar 每条 subject 不含 'subject #'、含真实 stem 标识（§4.1）。"""
        ws = _build_workspace_with_stem_outliers(tmp_path, monkeypatch)
        sidecar = ws / "handoff_code_executor_outliers.json"
        assert sidecar.exists()
        entries = json.loads(sidecar.read_text(encoding="utf-8"))
        assert len(entries) == 2
        for e in entries:
            assert "subject #" not in e["subject"], f"subject 仍是组内 index: {e['subject']!r}"
            assert "Trial" in e["subject"], f"subject 缺真实 stem 标识: {e['subject']!r}"

    def test_sidecar_counterfactual_carries_real_subject(self, tmp_path, monkeypatch):
        """counterfactual 串里的 subject 同步更新为真名（data-analyst 原样引用不重映射）。"""
        ws = _build_workspace_with_stem_outliers(tmp_path, monkeypatch)
        entries = json.loads((ws / "handoff_code_executor_outliers.json").read_text(encoding="utf-8"))
        for e in entries:
            assert "subject #" not in e["counterfactual"]
            assert e["subject"] in e["counterfactual"], "counterfactual 应含该条 subject 真名"

    def test_sidecar_has_qualitative_deviation_string(self, tmp_path, monkeypatch):
        """每条 deviation 是定性串（非裸数值），data-analyst 引用无需在 thinking 里翻译（§3.2）。"""
        ws = _build_workspace_with_stem_outliers(tmp_path, monkeypatch)
        entries = json.loads((ws / "handoff_code_executor_outliers.json").read_text(encoding="utf-8"))
        for e in entries:
            assert isinstance(e["deviation"], str)
            assert len(e["deviation"]) > 0
            # 定性串含 'group median' 或 'extreme deviation' 之类的措辞，非纯数字
            assert any(kw in e["deviation"] for kw in ("group median", "extreme deviation", "SD"))

    def test_sidecar_entry_constructs_outlier_finding(self, tmp_path, monkeypatch):
        """每条可直接构造 OutlierFinding（§4.2）——data-analyst 无需逐字段重组。"""
        ws = _build_workspace_with_stem_outliers(tmp_path, monkeypatch)
        entries = json.loads((ws / "handoff_code_executor_outliers.json").read_text(encoding="utf-8"))
        for e in entries:
            # OutlierFinding: subject / metric / value / deviation / counterfactual
            projection = {k: e[k] for k in ("subject", "metric", "value", "deviation", "counterfactual")}
            finding = OutlierFinding(**projection)  # 不抛即字段齐备、类型正确
            assert finding.subject == e["subject"]
            assert finding.deviation == e["deviation"]

    def test_main_handoff_references_sidecar(self, tmp_path, monkeypatch):
        """主 handoff 留 count + 虚拟路径 ref，statistics 不内嵌 outlier 数组。"""
        ws = _build_workspace_with_stem_outliers(tmp_path, monkeypatch)
        main = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert main["outlier_diagnostics_count"] == 2
        ref = main["outlier_diagnostics_ref"]
        assert ref is not None and ref.startswith("/mnt/user-data/workspace/")
        # 主 handoff statistics 不再内嵌完整 outlier_diagnostics
        stats = main.get("statistics", {})
        assert not stats.get("outlier_diagnostics"), "主 handoff 仍内嵌 outlier_diagnostics"
