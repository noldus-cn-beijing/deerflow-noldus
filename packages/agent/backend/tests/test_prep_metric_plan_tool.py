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
            "uploaded_files": ["/mnt/user-data/uploads/test_epm.txt"],
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

    def test_plan_metrics_json_contains_interpretation_fields(self, tmp_path):
        """W27 e2e: tool 写出的 plan_metrics.json 必须含 catalog 5 个判读 / 展示字段。

        这是 catalog → resolve_metrics → plan_metrics_to_dict → JSON 全链路防线。
        任何环节漏字段(dataclass / 透传 / 序列化),这里 fail。
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/test_epm.txt"],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_path = workspace / "plan_metrics.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert plan_data["metrics"], "plan_metrics.json metrics 为空"

        expected_fields = {
            "unit_zh",
            "one_liner",
            "output_unit",
            "direction_for_anxiety",
            "statistical_default",
        }
        for m in plan_data["metrics"]:
            missing = expected_fields - set(m.keys())
            assert not missing, f"metric {m['id']}: plan_metrics.json 缺字段 {missing}"
            # 类型契约(JSON 反序列化后)
            assert isinstance(m["unit_zh"], str)
            assert isinstance(m["one_liner"], str)
            assert isinstance(m["output_unit"], str)
            assert m["direction_for_anxiety"] in (None, "lower_is_anxious", "higher_is_anxious")
            assert m["statistical_default"] in ("groupwise_compare", "paired_compare")


class TestPrepMetricPlanToolErrors:
    def test_workspace_missing(self):
        """thread_data 为 None → error_code=workspace_missing, hint 含 'bug'。"""
        runtime = _runtime_without_workspace()
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/x.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/nonexistent.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/test.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/minimal.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/test2.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/w20_epm.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/w20_epm2.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/w20_epm3.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/vp_epm.txt"],
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
            "uploaded_files": ["/mnt/user-data/uploads/vp_epm2.txt"],
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
            "uploaded_files": [virtual_path],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        assert payload["inputs"]["raw_files"] == [virtual_path]
        for metric in payload["metrics"]:
            assert metric["input"] == virtual_path


class TestPrepMetricPlanToolMultipleFiles:
    """防回归 (2026-05-20 FST E2E): 用户传 2 个 Arena 数据,Arena 2 在 plan 这一层
    被丢失只生成 Arena 1 的 PlanMetric。修复后 prep_metric_plan 接受
    uploaded_files: list[str],为每个文件 × 每个指标各生成一个 PlanMetric。

    覆盖:
    - 单元素 list 仍走老路径,output 文件名兼容 m_<id>.json (无 _s 后缀)
    - 多元素 list 展开 N×M 个 PlanMetric,output 带 _s<idx> 后缀防覆盖
    - inputs.raw_files 保留全部文件路径
    - subject_index 字段 0..N-1
    - failed_file 字段定位失败文件
    - 空 list 报 no_files_provided
    """

    def test_two_files_expand_to_metrics_times_files(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        f1 = uploads / "arena1.txt"
        f2 = uploads / "arena2.txt"
        _write_ethovision_file(str(f1), EPM_COLUMNS)
        _write_ethovision_file(str(f2), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": [
                "/mnt/user-data/uploads/arena1.txt",
                "/mnt/user-data/uploads/arena2.txt",
            ],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["subject_count"] == 2
        unique_metric_count = result["plan_summary"]["metric_count"]
        assert unique_metric_count > 0

        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        # raw_files 完整保留两个虚拟路径
        assert payload["inputs"]["raw_files"] == [
            "/mnt/user-data/uploads/arena1.txt",
            "/mnt/user-data/uploads/arena2.txt",
        ]
        # metrics 数 = 唯一指标 × 文件数
        assert len(payload["metrics"]) == unique_metric_count * 2
        # 每个指标各出现 2 次 (每个 subject 一次),且 subject_index 是 0/1
        from collections import Counter
        per_id_count = Counter(m["id"] for m in payload["metrics"])
        for mid, cnt in per_id_count.items():
            assert cnt == 2, f"metric {mid} 出现 {cnt} 次,应为 2"
        indices = sorted({m["subject_index"] for m in payload["metrics"]})
        assert indices == [0, 1]
        # output 带 _s 后缀防覆盖
        outputs = [m["output"] for m in payload["metrics"]]
        s0 = [o for o in outputs if "_s0" in o]
        s1 = [o for o in outputs if "_s1" in o]
        assert len(s0) == unique_metric_count
        assert len(s1) == unique_metric_count
        # 每个指标的 input 都对应它声明的 subject
        for m in payload["metrics"]:
            expected_input = (
                "/mnt/user-data/uploads/arena1.txt"
                if m["subject_index"] == 0
                else "/mnt/user-data/uploads/arena2.txt"
            )
            assert m["input"] == expected_input

    def test_single_file_keeps_legacy_output_name_without_suffix(self, tmp_path):
        """单文件场景 output 仍是 m_<id>.json 无 _s 后缀,保持向后兼容。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "single.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": ["/mnt/user-data/uploads/single.txt"],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["subject_count"] == 1
        plan_file = workspace / "plan_metrics.json"
        payload = json.loads(plan_file.read_text())
        for m in payload["metrics"]:
            assert "_s" not in m["output"], (
                f"单文件 output 不应带 _s 后缀: {m['output']}"
            )
            assert m["subject_index"] == 0

    def test_empty_uploaded_files_returns_error(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": [],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "no_files_provided"
        assert "no_files_provided" in _ERROR_HINTS

    def test_second_file_missing_returns_failed_file(self, tmp_path):
        """第 2 个文件不存在 → 报 file_not_found + failed_file 字段指向 arena2。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        f1 = uploads / "arena1.txt"
        _write_ethovision_file(str(f1), EPM_COLUMNS)
        # arena2 不写入

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_files": [
                "/mnt/user-data/uploads/arena1.txt",
                "/mnt/user-data/uploads/arena2.txt",
            ],
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "file_not_found"
        assert result["failed_file"] == "/mnt/user-data/uploads/arena2.txt"
