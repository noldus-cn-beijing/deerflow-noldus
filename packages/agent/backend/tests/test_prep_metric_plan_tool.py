"""Tests for prep_metric_plan_tool."""

import json

import pytest
from langchain.tools import ToolRuntime

from deerflow.tools.builtins.prep_metric_plan_tool import (
    _ERROR_HINTS,
    prep_metric_plan_tool,
)


def _runtime_with_paths(workspace, uploads) -> ToolRuntime:
    """Build a real ToolRuntime with thread_data state (matches set_experiment_paradigm test style)."""
    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(uploads),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _runtime_without_workspace() -> ToolRuntime:
    return ToolRuntime(
        state={"thread_data": None},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _write_ethovision_file(path: str, columns: list[str]):
    """Write a UTF-16 LE EthoVision trajectory file with full metadata header.

    parse_header 期望: BOM + line-count + metadata kv 段 + column-header 行 + units 行 + data。
    """
    header_lines = 36
    lines: list[str] = []
    # Line 1: header line count
    lines.append(f'"Number of header lines:";"{header_lines}"')
    # Lines 2..34: metadata key-value pairs
    metadata = [
        ("Experiment", "Mock EPM"),
        ("Trial name", "Trial 1"),
        ("Subject", "Subject 1"),
        ("Start time", "2026-01-01 00:00:00"),
        ("Trial duration", "300"),
        ("Arena name", "Arena 1"),
        ("Number of Subjects", "1"),
    ]
    for k, v in metadata:
        lines.append(f'"{k}";"{v}"')
    # Pad metadata to header_lines - 2 lines
    while len(lines) < header_lines - 2:
        lines.append('""')
    # column-header line
    lines.append('"' + '";"'.join(columns) + '"')
    # units line
    lines.append('"' + '";"'.join(["s"] * len(columns)) + '"')
    # 1 data row
    lines.append(";".join(["-1.0"] * len(columns)))
    content = "\r\n".join(lines) + "\r\n"
    # BOM + UTF-16 LE
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")  # UTF-16 LE BOM
        f.write(content.encode("utf-16-le"))


EPM_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "in zone Open arms 1 / Center-point",
    "in zone Open arms 2 / Center-point",
    "in zone Closed arms 1 / Center-point",
    "in zone Closed arms 2 / Center-point",
    "in zone Center-point / Center-point",
]


class TestPrepMetricPlanToolOk:
    def test_normal_path_with_epm_data(self, tmp_path):
        """正常路径: mock EthoVision EPM 数据 → status=ok, metric_count > 0。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test_epm.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["paradigm"] == "epm"
        assert result["plan_summary"]["metric_count"] > 0
        # plan_path 真实存在
        plan_path = workspace / "plan_metrics.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert "metrics" in plan_data


class TestPrepMetricPlanToolErrors:
    def test_workspace_missing(self):
        """thread_data 为 None → error_code=workspace_missing, hint 含 'bug'。"""
        runtime = _runtime_without_workspace()
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/x.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "workspace_missing"
        assert "bug" in result["hint"].lower()

    def test_file_not_found(self, tmp_path):
        """传不存在的路径 → error_code=file_not_found, hint 含 ask_clarification。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/nonexistent.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "file_not_found"
        assert "ask_clarification" in result["hint"].lower()

    def test_unknown_paradigm(self, tmp_path):
        """传 paradigm='invalid' → error_code=unknown_paradigm。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test.txt",
            "paradigm": "invalid_paradigm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "unknown_paradigm"

    def test_columns_missing(self, tmp_path):
        """mock 数据缺 in_zone_open_arms_* 列, paradigm=epm → error_code=columns_missing/empty_plan。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        # 只有基础列，没有 in zone Open arms
        minimal_columns = [
            "Trial time",
            "Recording time",
            "X center",
            "Y center",
        ]
        data_file = uploads / "minimal.txt"
        _write_ethovision_file(str(data_file), minimal_columns)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/minimal.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] in {"columns_missing", "empty_plan"}

    def test_plan_file_written_on_success(self, tmp_path):
        """status=ok 后 plan_path 真实存在 + JSON 可读。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test2.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test2.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_path = workspace / "plan_metrics.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert isinstance(plan_data, dict)
        assert "metrics" in plan_data
        assert len(plan_data["metrics"]) == result["plan_summary"]["metric_count"]


class TestPrepMetricPlanToolW20:
    """W20: 输出文件名 metric_plan.json → plan_metrics.json，import resolve_metrics + plan_metrics_to_dict。"""

    def test_writes_plan_metrics_json_not_metric_plan_json(self, tmp_path):
        """W20: 输出文件名从 metric_plan.json 改为 plan_metrics.json。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "w20_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/w20_epm.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        # plan_path 返回值中包含新文件名
        assert result["plan_path"] == "/mnt/user-data/workspace/plan_metrics.json"
        # 新文件存在
        assert (workspace / "plan_metrics.json").exists()
        # 旧文件不存在
        assert not (workspace / "metric_plan.json").exists()

    def test_plan_metrics_json_has_no_charts_field(self, tmp_path):
        """W20: plan_metrics 输出 JSON 不含 charts 字段。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "w20_epm2.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/w20_epm2.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        assert plan_file.exists()
        payload = json.loads(plan_file.read_text())
        assert "metrics" in payload
        assert "charts" not in payload

    def test_plan_metrics_json_has_metrics_field(self, tmp_path):
        """W20: plan_metrics.json 含 metrics 字段且 metric_count > 0。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "w20_epm3.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/w20_epm3.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        assert isinstance(payload["metrics"], list)
        assert len(payload["metrics"]) == result["plan_summary"]["metric_count"]
        assert len(payload["metrics"]) > 0


class TestPrepMetricPlanToolVirtualPathLeakage:
    """2026-05-20: plan_metrics.json 不能泄漏宿主机绝对路径。

    根因:之前 prep_metric_plan_tool 把 real_file_path (宿主机绝对路径)
    传给 resolve_metrics 作 raw_files,导致 plan.inputs.raw_files 和
    plan.metrics[*].input 都是宿主机路径。subagent read 后照抄进 --input,
    宿主机路径在 sandbox 内不可达 → 失败 → 重试用虚拟路径 → 成功。
    浪费一个 tool call。
    """

    def test_inputs_raw_files_uses_virtual_path(self, tmp_path):
        """plan_metrics.json 的 inputs.raw_files[*] 必须是 /mnt/user-data/uploads/<file>。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "vp_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/vp_epm.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        assert payload["inputs"]["raw_files"] == ["/mnt/user-data/uploads/vp_epm.txt"]
        # 绝对不能含宿主机路径片段
        for rf in payload["inputs"]["raw_files"]:
            assert str(uploads) not in rf, f"raw_files leaked host path: {rf}"

    def test_metrics_input_uses_virtual_path(self, tmp_path):
        """plan_metrics.json 的 metrics[*].input 必须是虚拟路径。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "vp_epm2.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/vp_epm2.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        assert len(payload["metrics"]) > 0
        for metric in payload["metrics"]:
            assert metric["input"] == "/mnt/user-data/uploads/vp_epm2.txt", (
                f"metric {metric['id']} input is not virtual path: {metric['input']}"
            )
            assert str(uploads) not in metric["input"]

    def test_filename_with_spaces_preserved_in_virtual_path(self, tmp_path):
        """EthoVision 常见文件名含连续空格,虚拟路径要原样保留。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        # 模拟 5 连续空格的 EthoVision 文件名
        filename = "轨迹-EPM XT190-Trial     1-Arena 1.txt"
        data_file = uploads / filename
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        virtual_path = f"/mnt/user-data/uploads/{filename}"
        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": virtual_path,
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        assert payload["inputs"]["raw_files"] == [virtual_path]
        for metric in payload["metrics"]:
            assert metric["input"] == virtual_path
