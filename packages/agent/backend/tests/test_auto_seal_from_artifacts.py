"""Spec C — Integration tests for _attempt_auto_seal_from_artifacts.

Tests the deterministic auto-seal fallback: when seal-resume fails but the
subagent has produced output files (report.md / plot_*.png), the harness
constructs a valid handoff JSON from those artifacts without relying on LLM.

Because conftest.py pre-mocks deerflow.subagents.executor, we load the real
module via importlib (same pattern as test_seal_resume / test_executor_handoff_emission).
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the real executor module (same pattern as test_executor_handoff_emission)
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
        "deerflow.subagents.executor_real_auto_seal",
        _EXECUTOR_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _REAL_EXECUTOR = module
    return module


@pytest.fixture(autouse=True)
def _load_module():
    _get_real_executor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_thread(tmp_path: Path):
    """Create thread directory structure: user-data/{workspace,outputs}.

    Returns (workspace_path_str, outputs_path).
    """
    user_data = tmp_path / "user-data"
    ws = user_data / "workspace"
    out = user_data / "outputs"
    ws.mkdir(parents=True)
    out.mkdir(parents=True)
    return str(ws), out


def _auto_seal(subagent_name: str, workspace_path: str | None) -> bool:
    """Shortcut to call _attempt_auto_seal_from_artifacts on the real module."""
    mod = _get_real_executor()
    return mod._attempt_auto_seal_from_artifacts(subagent_name, workspace_path)


# ===================================================================
# Test: report-writer auto-seal (core path)
# ===================================================================

class TestAutoSealReportWriter:
    """report-writer 有产出 (outputs/report.md) 无 handoff → auto-seal 成功。"""

    def test_report_exists_no_handoff_auto_seals(self, tmp_path: Path):
        """report.md 存在、无 handoff → auto-seal 成功，产物过内容校验。"""
        mod = _get_real_executor()
        ws, out = _mk_thread(tmp_path)
        # 写 report.md（模拟 subagent 已完成产出）
        (out / "report.md").write_text(
            "# 实验概况\n\n内容...\n\n## 结果\n\n统计分析结果...\n\n## 讨论\n\n讨论内容...\n",
            encoding="utf-8",
        )
        # experiment-context.json（_seal_handoff_to_workspace 需要 analysis_config_id）
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config-123"}),
            encoding="utf-8",
        )

        ok = _auto_seal("report-writer", ws)
        assert ok is True

        # handoff 文件已写入
        h = Path(ws) / "handoff_report_writer.json"
        assert h.exists()
        data = json.loads(h.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["report_path"] == "/mnt/user-data/outputs/report.md"
        assert len(data["sections_written"]) >= 2  # 从 markdown 标题解析
        assert "实验概况" in data["sections_written"]
        assert "结果" in data["sections_written"]
        assert any("auto-seal" in e for e in data["errors"])  # auto-seal 标记
        # analysis_config_id 已注入（由 _seal_handoff_to_workspace）
        assert data["analysis_config_id"] == "test-config-123"
        # 文件权限 644（Spec1 教训）
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

        # 集成断言：auto-seal 产物能过 _validate_handoff_emitted（含内容非空检查）
        validation_error = mod._validate_handoff_emitted("report-writer", ws)
        assert validation_error is None, f"auto-sealed handoff should validate, got: {validation_error}"

    def test_no_report_does_not_auto_seal(self, tmp_path: Path):
        """outputs/ 空（无 report.md）→ 不兜底，返回 False。"""
        ws, _out = _mk_thread(tmp_path)  # outputs 空
        ok = _auto_seal("report-writer", ws)
        assert ok is False  # 没产出 → 不兜底 → 外层走 FAILED

    def test_existing_handoff_not_overwritten(self, tmp_path: Path):
        """workspace 已有非空 handoff → 不覆盖，返回 False。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        # 已有 handoff（模拟 seal-resume 成功后又被 auto-seal 检查到——不该到这，保险）
        (Path(ws) / "handoff_report_writer.json").write_text(
            '{"status":"completed","report_path":"/x","sections_written":["orig"]}',
            encoding="utf-8",
        )
        ok = _auto_seal("report-writer", ws)
        assert ok is False  # 已有 handoff → 不覆盖

    def test_empty_report_does_not_auto_seal(self, tmp_path: Path):
        """report.md 为空文件 → 不兜底（无实质产出）。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("", encoding="utf-8")  # 空文件
        ok = _auto_seal("report-writer", ws)
        assert ok is False

    def test_sections_parsed_from_headings(self, tmp_path: Path):
        """验证从 markdown 标题解析 sections_written 的准确性。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text(
            "# 方法\n方法内容\n## 被试\n被试信息\n### 分组\n分组信息\n# 结果\n结果内容\n",
            encoding="utf-8",
        )
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )

        ok = _auto_seal("report-writer", ws)
        assert ok is True
        data = json.loads((Path(ws) / "handoff_report_writer.json").read_text(encoding="utf-8"))
        # 各级标题都被提取（去除 # 前缀和空格）
        assert "方法" in data["sections_written"]
        assert "被试" in data["sections_written"]
        assert "分组" in data["sections_written"]
        assert "结果" in data["sections_written"]
        assert len(data["sections_written"]) == 4


# ===================================================================
# Test: chart-maker auto-seal
# ===================================================================

class TestAutoSealChartMaker:
    """chart-maker 有产出 (outputs/plot_*.png) 无 handoff → auto-seal 成功。"""

    def test_charts_exist_no_handoff_auto_seals(self, tmp_path: Path):
        """plot_*.png 存在 → auto-seal 成功，chart_files 带正确虚拟路径前缀。"""
        mod = _get_real_executor()
        ws, out = _mk_thread(tmp_path)
        # 模拟 chart-maker 已产出图表
        (out / "plot_trajectory.png").write_text("fake png", encoding="utf-8")
        (out / "plot_heatmap.png").write_text("fake png", encoding="utf-8")
        # code-executor handoff（提供 paradigm）
        (Path(ws) / "handoff_code_executor.json").write_text(
            json.dumps({"paradigm": "epm", "metrics_summary": {"epm": {}}}),
            encoding="utf-8",
        )
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config"}),
            encoding="utf-8",
        )

        ok = _auto_seal("chart-maker", ws)
        assert ok is True

        h = Path(ws) / "handoff_chart_maker.json"
        assert h.exists()
        data = json.loads(h.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["paradigm"] == "epm"
        assert "auto-seal" in data["summary"]
        # chart_files 有正确的 /mnt/user-data/outputs/ 前缀（schema 校验要求）
        assert len(data["chart_files"]) == 2
        for cf in data["chart_files"]:
            assert cf.startswith("/mnt/user-data/outputs/")
        chart_names = [Path(cf).name for cf in data["chart_files"]]
        assert "plot_trajectory.png" in chart_names
        assert "plot_heatmap.png" in chart_names
        # 文件权限 644
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

        # 集成断言：能过 _validate_handoff_emitted
        validation_error = mod._validate_handoff_emitted("chart-maker", ws)
        assert validation_error is None, f"auto-sealed chart-maker handoff should validate, got: {validation_error}"

    def test_no_charts_does_not_auto_seal(self, tmp_path: Path):
        """outputs/ 无 plot_*.png → 不兜底。"""
        ws, _out = _mk_thread(tmp_path)  # outputs 空
        ok = _auto_seal("chart-maker", ws)
        assert ok is False

    def test_paradigm_fallback_empty_string(self, tmp_path: Path):
        """无 code_executor handoff → paradigm 回退空字符串。"""
        ws, out = _mk_thread(tmp_path)
        (out / "plot_test.png").write_text("fake", encoding="utf-8")
        (Path(ws) / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )
        # 无 handoff_code_executor.json

        ok = _auto_seal("chart-maker", ws)
        assert ok is True
        data = json.loads((Path(ws) / "handoff_chart_maker.json").read_text(encoding="utf-8"))
        assert data["paradigm"] == ""  # 回退空字符串（schema 允许）


# ===================================================================
# Test: code-executor auto-seal (Spec A)
# ===================================================================

class TestAutoSealCodeExecutorCompleted:
    """code-executor 有完整 plan + 全部 m_*.json → auto-seal completed。"""

    def test_full_reconstruction_auto_seals_completed(self, tmp_path: Path):
        """3 指标 × 2 subject = 6 期望，全在 → status=completed, sealed_by=framework_rebuild。"""
        mod = _get_real_executor()
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        # 写 plan_metrics.json（模拟 lead 已经生成好的施工单）
        plan = {
            "paradigm": "epm",
            "ev19_template": "epm_v1",
            "inputs": {
                "raw_files": [
                    "/mnt/user-data/uploads/subject1.xlsx",
                    "/mnt/user-data/uploads/subject2.xlsx",
                ],
                "groups_file": "/mnt/user-data/workspace/groups.json",
            },
            "metrics": [
                {
                    "id": "open_arm_time_ratio",
                    "script": "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
                    "output": "/mnt/user-data/workspace/m_open_arm_time_ratio_s0.json",
                    "subject_index": 0,
                    "display_name_zh": "开臂时间比",
                },
                {
                    "id": "open_arm_time_ratio",
                    "script": "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
                    "output": "/mnt/user-data/workspace/m_open_arm_time_ratio_s1.json",
                    "subject_index": 1,
                    "display_name_zh": "开臂时间比",
                },
                {
                    "id": "open_arm_entry_ratio",
                    "script": "ethoinsight.scripts.epm.compute_open_arm_entry_ratio",
                    "output": "/mnt/user-data/workspace/m_open_arm_entry_ratio_s0.json",
                    "subject_index": 0,
                    "display_name_zh": "开臂进入比",
                },
                {
                    "id": "open_arm_entry_ratio",
                    "script": "ethoinsight.scripts.epm.compute_open_arm_entry_ratio",
                    "output": "/mnt/user-data/workspace/m_open_arm_entry_ratio_s1.json",
                    "subject_index": 1,
                    "display_name_zh": "开臂进入比",
                },
                {
                    "id": "total_distance",
                    "script": "ethoinsight.scripts.epm.compute_total_distance",
                    "output": "/mnt/user-data/workspace/m_total_distance_s0.json",
                    "subject_index": 0,
                    "display_name_zh": "总路程",
                },
                {
                    "id": "total_distance",
                    "script": "ethoinsight.scripts.epm.compute_total_distance",
                    "output": "/mnt/user-data/workspace/m_total_distance_s1.json",
                    "subject_index": 1,
                    "display_name_zh": "总路程",
                },
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        # 写 groups.json
        groups = {
            "/mnt/user-data/uploads/subject1.xlsx": "Control",
            "/mnt/user-data/uploads/subject2.xlsx": "Treatment",
        }
        (ws_path / "groups.json").write_text(json.dumps(groups), encoding="utf-8")

        # 写所有 6 个 m_*.json
        metric_files = {
            "m_open_arm_time_ratio_s0.json": {"metric": "open_arm_time_ratio", "value": 0.35, "parameters_used": {}},
            "m_open_arm_time_ratio_s1.json": {"metric": "open_arm_time_ratio", "value": 0.42, "parameters_used": {}},
            "m_open_arm_entry_ratio_s0.json": {"metric": "open_arm_entry_ratio", "value": 0.40, "parameters_used": {}},
            "m_open_arm_entry_ratio_s1.json": {"metric": "open_arm_entry_ratio", "value": 0.38, "parameters_used": {}},
            "m_total_distance_s0.json": {"metric": "total_distance", "value": 1500.0, "parameters_used": {}},
            "m_total_distance_s1.json": {"metric": "total_distance", "value": 1200.0, "parameters_used": {}},
        }
        for fname, data in metric_files.items():
            (ws_path / fname).write_text(json.dumps(data), encoding="utf-8")

        # experiment-context.json（_seal 需要 analysis_config_id）
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test-config-epm"}),
            encoding="utf-8",
        )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        h = ws_path / "handoff_code_executor.json"
        assert h.exists()
        data = json.loads(h.read_text(encoding="utf-8"))

        # 状态
        assert data["status"] == "completed"
        assert data["sealed_by"] == "framework_rebuild"
        assert data["paradigm"] == "epm"
        assert data["ev19_template"] == "epm_v1"
        assert "auto-seal" in data["summary"]
        assert "6/6" in data["summary"]

        # metrics_summary（规范字段名）
        assert "metrics_summary" in data
        ms = data["metrics_summary"]
        assert "Control" in ms
        assert "Treatment" in ms
        assert "open_arm_time_ratio" in ms["Control"]
        assert ms["Control"]["open_arm_time_ratio"]["mean"] == 0.35
        assert ms["Treatment"]["open_arm_time_ratio"]["mean"] == 0.42

        # per_subject
        assert "per_subject" in data
        ps = data["per_subject"]
        assert "subject1" in ps
        assert "subject2" in ps
        assert ps["subject1"]["open_arm_time_ratio"] == 0.35
        assert ps["subject2"]["total_distance"] == 1200.0

        # errors 为空（全量产出）
        assert data["errors"] == []

        # inputs 透传
        assert data["inputs"]["raw_files"] == plan["inputs"]["raw_files"]
        assert data["inputs"]["groups"] == groups

        # 文件权限 644
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

        # 集成断言：能过 _validate_handoff_emitted（含内容非空检查，走 metrics_summary）
        validation_error = mod._validate_handoff_emitted("code-executor", str(ws_path))
        assert validation_error is None, f"auto-sealed code-executor handoff should validate, got: {validation_error}"

    def test_empty_plan_metrics_does_not_auto_seal(self, tmp_path: Path):
        """plan_metrics.json 存在但 metrics 数组为空 → 不兜底。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)
        (ws_path / "plan_metrics.json").write_text(
            json.dumps({"paradigm": "epm", "metrics": []}),
            encoding="utf-8",
        )
        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is False

    def test_no_expected_outputs_in_plan_does_not_auto_seal(self, tmp_path: Path):
        """plan 的 metrics 条目都没有 output 字段 → 无法枚举期望集 → 不兜底。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)
        (ws_path / "plan_metrics.json").write_text(
            json.dumps({
                "paradigm": "epm",
                "metrics": [{"id": "x", "script": "y", "subject_index": 0}],
            }),
            encoding="utf-8",
        )
        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is False


class TestAutoSealCodeExecutorPartial:
    """完整性判据：有缺失 m_*.json → status=partial。"""

    def test_missing_one_metric_marks_partial(self, tmp_path: Path):
        """6 期望，5 实际 → status=partial，errors 含缺失项。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        plan = {
            "paradigm": "oft",
            "inputs": {
                "raw_files": ["/mnt/user-data/uploads/s1.xlsx", "/mnt/user-data/uploads/s2.xlsx"],
            },
            "metrics": [
                {"id": "center_distance", "script": "...", "output": "/mnt/user-data/workspace/m_center_distance_s0.json", "subject_index": 0},
                {"id": "center_distance", "script": "...", "output": "/mnt/user-data/workspace/m_center_distance_s1.json", "subject_index": 1},
                {"id": "distance_ratio", "script": "...", "output": "/mnt/user-data/workspace/m_distance_ratio_s0.json", "subject_index": 0},
                {"id": "distance_ratio", "script": "...", "output": "/mnt/user-data/workspace/m_distance_ratio_s1.json", "subject_index": 1},
                {"id": "total_distance", "script": "...", "output": "/mnt/user-data/workspace/m_total_distance_s0.json", "subject_index": 0},
                {"id": "total_distance", "script": "...", "output": "/mnt/user-data/workspace/m_total_distance_s1.json", "subject_index": 1},
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
        (ws_path / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/s1.xlsx": "A", "/mnt/user-data/uploads/s2.xlsx": "B"}),
            encoding="utf-8",
        )
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )

        # 只写 5 个（缺 m_total_distance_s1.json）
        for fname in [
            "m_center_distance_s0.json", "m_center_distance_s1.json",
            "m_distance_ratio_s0.json", "m_distance_ratio_s1.json",
            "m_total_distance_s0.json",
        ]:
            (ws_path / fname).write_text(
                json.dumps({"metric": fname.replace(".json", ""), "value": 100.0}),
                encoding="utf-8",
            )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        h = ws_path / "handoff_code_executor.json"
        data = json.loads(h.read_text(encoding="utf-8"))

        # 完整性判据核心断言
        assert data["status"] == "partial", (
            f"Expected partial but got {data['status']}: missing file should prevent completed"
        )
        assert "5/6" in data["summary"]

        # errors 含缺失项
        assert len(data["errors"]) >= 1
        missing_errors = [e for e in data["errors"] if "total_distance_s1" in e]
        assert len(missing_errors) >= 1, f"errors should mention missing file, got: {data['errors']}"

        # data_quality_warnings 含 AUTO_SEAL_INCOMPLETE
        assert len(data["data_quality_warnings"]) >= 1
        incomplete_warnings = [
            w for w in data["data_quality_warnings"]
            if w.get("code") == "METHOD.AUTO_SEAL_INCOMPLETE"
        ]
        assert len(incomplete_warnings) == 1
        assert incomplete_warnings[0]["severity"] == "critical"
        assert incomplete_warnings[0]["blocks_downstream"] is True

    def test_missing_does_not_block_other_reconstruction(self, tmp_path: Path):
        """即使有缺失，存在的 m_*.json 仍被正确重建进 metrics_summary。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        plan = {
            "paradigm": "oft",
            "inputs": {"raw_files": ["/mnt/user-data/uploads/s1.xlsx"]},
            "metrics": [
                {"id": "center_distance", "script": "...", "output": "/mnt/user-data/workspace/m_center_distance_s0.json", "subject_index": 0},
                {"id": "distance_ratio", "script": "...", "output": "/mnt/user-data/workspace/m_distance_ratio_s0.json", "subject_index": 0},
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
        (ws_path / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/s1.xlsx": "Control"}),
            encoding="utf-8",
        )
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )

        # 只写 1 个（缺 m_distance_ratio_s0.json）
        (ws_path / "m_center_distance_s0.json").write_text(
            json.dumps({"metric": "center_distance", "value": 42.5, "parameters_used": {"center_zone": "in_zone"}}),
            encoding="utf-8",
        )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        data = json.loads((ws_path / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert data["status"] == "partial"
        # 存在的指标仍被重建
        assert "center_distance" in data["metrics_summary"].get("Control", {})
        assert data["metrics_summary"]["Control"]["center_distance"]["mean"] == 42.5
        assert data["metrics_summary"]["Control"]["center_distance"]["parameters_used"] == {"center_zone": "in_zone"}


class TestAutoSealCodeExecutorFieldNameCanonical:
    """Spec A.2.4: 重建产出使用规范字段名 metrics_summary，绕开字段三分裂故障类。"""

    def test_reconstructed_handoff_uses_canonical_metrics_summary(self, tmp_path: Path):
        """auto-seal 产出的 handoff 使用 metrics_summary（规范名），过内容校验。"""
        mod = _get_real_executor()
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        plan = {
            "paradigm": "fst",
            "inputs": {"raw_files": ["/mnt/user-data/uploads/s1.xlsx"]},
            "metrics": [
                {"id": "immobility_time", "script": "...", "output": "/mnt/user-data/workspace/m_immobility_time_s0.json", "subject_index": 0},
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
        (ws_path / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/s1.xlsx": "Treatment"}),
            encoding="utf-8",
        )
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )
        (ws_path / "m_immobility_time_s0.json").write_text(
            json.dumps({"metric": "immobility_time", "value": 120.0, "parameters_used": {"velocity_threshold": 30}}),
            encoding="utf-8",
        )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        data = json.loads((ws_path / "handoff_code_executor.json").read_text(encoding="utf-8"))

        # 规范字段名（非 metrics / metrics_results）
        assert "metrics_summary" in data, "auto-seal MUST use canonical field name 'metrics_summary'"
        assert data["metrics_summary"] != {}, "metrics_summary should be non-empty"

        # 过 _check_code_executor_content（规范字段非空校验）
        check_result = mod._check_code_executor_content(data)
        assert check_result is None, f"canonical field should pass content check, got: {check_result}"

        # 集成：过完整 _validate_handoff_emitted
        validation_error = mod._validate_handoff_emitted("code-executor", str(ws_path))
        assert validation_error is None, f"should validate, got: {validation_error}"

    def test_historical_metrics_field_not_present_in_reconstructed(self, tmp_path: Path):
        """重建产物不含 metrics / metrics_results 历史字段（只含规范 metrics_summary）。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        plan = {
            "paradigm": "fst",
            "inputs": {"raw_files": ["/mnt/user-data/uploads/s1.xlsx"]},
            "metrics": [
                {"id": "immobility_time", "script": "...", "output": "/mnt/user-data/workspace/m_immobility_time_s0.json", "subject_index": 0},
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
        (ws_path / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/s1.xlsx": "Treatment"}),
            encoding="utf-8",
        )
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )
        (ws_path / "m_immobility_time_s0.json").write_text(
            json.dumps({"metric": "immobility_time", "value": 120.0}),
            encoding="utf-8",
        )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        data = json.loads((ws_path / "handoff_code_executor.json").read_text(encoding="utf-8"))
        # 不含历史字段
        assert "metrics" not in data, "reconstructed handoff should NOT contain legacy 'metrics' field"
        assert "metrics_results" not in data, "reconstructed handoff should NOT contain legacy 'metrics_results' field"


class TestAutoSealCodeExecutorSealedBy:
    """sealed_by 来源标记：auto-seal 产物必须标 framework_rebuild。"""

    def test_sealed_by_is_framework_rebuild(self, tmp_path: Path):
        """auto-seal 产出的 handoff 标记 sealed_by=framework_rebuild。"""
        ws, _out = _mk_thread(tmp_path)
        ws_path = Path(ws)

        plan = {
            "paradigm": "epm",
            "inputs": {"raw_files": ["/mnt/user-data/uploads/s1.xlsx"]},
            "metrics": [
                {"id": "oat", "script": "...", "output": "/mnt/user-data/workspace/m_oat_s0.json", "subject_index": 0},
            ],
        }
        (ws_path / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")
        (ws_path / "groups.json").write_text(
            json.dumps({"/mnt/user-data/uploads/s1.xlsx": "Control"}),
            encoding="utf-8",
        )
        (ws_path / "experiment-context.json").write_text(
            json.dumps({"analysis_config_id": "test"}),
            encoding="utf-8",
        )
        (ws_path / "m_oat_s0.json").write_text(
            json.dumps({"metric": "oat", "value": 0.35}),
            encoding="utf-8",
        )

        ok = _auto_seal("code-executor", str(ws_path))
        assert ok is True

        data = json.loads((ws_path / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert data.get("sealed_by") == "framework_rebuild", (
            f"auto-sealed handoff MUST have sealed_by='framework_rebuild', got {data.get('sealed_by')}"
        )

    def test_sealed_by_defaults_to_model_in_schema(self, tmp_path: Path):
        """验证 schema 默认值为 'model'（正常 seal 路径的产物）。"""
        # 直接构造一个正常 seal 的 handoff（不经过 auto-seal）
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        h = CodeExecutorHandoff(
            status="completed",
            summary="test",
            paradigm="epm",
            analysis_config_id="test123",
            metrics_summary={"Control": {"oat": {"mean": 0.35, "std": None, "n": 1}}},
        )
        assert h.sealed_by == "model", "default sealed_by should be 'model'"

class TestAutoSealScope:
    """data-analyst 永不 auto-seal（判读是认知产物）。code-executor 在无 plan 时也不 auto-seal。"""

    def test_data_analyst_never_auto_sealed(self, tmp_path: Path):
        """data-analyst 即便有报告文件也绝不 auto-seal（判读是认知产物）。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        ok = _auto_seal("data-analyst", ws)
        assert ok is False

    def test_code_executor_without_plan_does_not_auto_seal(self, tmp_path: Path):
        """code-executor 无 plan_metrics.json → 不兜底（无法对账）。"""
        ws, _out = _mk_thread(tmp_path)
        # 有 m_*.json 但无 plan → 无法枚举期望集 → 不兜底
        (Path(ws) / "m_test.json").write_text(
            json.dumps({"metric": "test", "value": 1.0}),
            encoding="utf-8",
        )
        ok = _auto_seal("code-executor", ws)
        assert ok is False

    def test_unknown_subagent_never_auto_sealed(self, tmp_path: Path):
        """未在白名单的 subagent → 不兜底。"""
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X\n", encoding="utf-8")
        ok = _auto_seal("general-purpose", ws)
        assert ok is False

    def test_no_workspace_fails_safe(self, tmp_path: Path):
        """workspace_path=None → 不抛异常，返回 False。"""
        assert _auto_seal("report-writer", None) is False


# ===================================================================
# Test: robustness — never raise
# ===================================================================

class TestAutoSealRobustness:
    """_attempt_auto_seal_from_artifacts 绝不抛异常（调用点在 executor try/except 内）。"""

    def test_never_raises_on_garbage(self):
        """垃圾路径 → 返回 False，不抛。"""
        try:
            result = _auto_seal("report-writer", "/nonexistent/\x00bad")
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_on_empty_string_workspace(self):
        """空字符串 workspace → 返回 False，不抛。"""
        try:
            result = _auto_seal("report-writer", "")
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_on_non_existent_outputs(self, tmp_path: Path):
        """workspace 存在但 outputs 目录不存在 → 返回 False，不抛。"""
        ws_path = tmp_path / "workspace_only"
        ws_path.mkdir(parents=True)
        # 没有 outputs 目录
        try:
            result = _auto_seal("report-writer", str(ws_path))
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")

    def test_never_raises_when_report_unreadable(self, tmp_path: Path):
        """report.md 存在但不可读 → 返回 False，不抛。"""
        ws, out = _mk_thread(tmp_path)
        report = out / "report.md"
        report.write_text("# X\n", encoding="utf-8")
        # 模拟不可读：删除文件后指向不存在的路径
        # 实际上我们测试 outputs 目录存在但 glob 返回空
        # 更实际的场景：workspace 路径下有 outputs 但 outputs 不是目录
        bad_tmp = tmp_path / "bad_thread"
        bad_tmp.mkdir()
        bad_ws = bad_tmp / "workspace"
        bad_ws.mkdir(parents=True)
        # outputs 是文件而非目录
        (bad_tmp / "outputs").write_text("not a dir")
        try:
            result = _auto_seal("report-writer", str(bad_ws))
            assert result is False
        except Exception as e:
            pytest.fail(f"must not raise, got {e}")


# ===================================================================
# Test: auto-seal 不破坏 _AUTO_SEALABLE 白名单完整性
# ===================================================================

class TestAutoSealableMap:
    """验证 _AUTO_SEALABLE 包含 report-writer、chart-maker、code-executor（Spec A），不含 data-analyst。"""

    def test_auto_sealable_contains_mechanical_and_code_executor(self):
        mod = _get_real_executor()
        sealable = mod._AUTO_SEALABLE
        assert "report-writer" in sealable
        assert "chart-maker" in sealable
        assert "code-executor" in sealable, "Spec A: code-executor is now auto-sealable (metrics from m_*.json)"
        assert "data-analyst" not in sealable, "data-analyst handoff is cognitive product"
        assert "general-purpose" not in sealable
        assert len(sealable) == 3  # report-writer + chart-maker + code-executor
