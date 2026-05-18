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
        plan_path = workspace / "metric_plan.json"
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
        plan_path = workspace / "metric_plan.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert isinstance(plan_data, dict)
        assert "metrics" in plan_data
        assert len(plan_data["metrics"]) == result["plan_summary"]["metric_count"]
