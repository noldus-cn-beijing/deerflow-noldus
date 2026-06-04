"""Unit tests for seal_handoff_tools.py (Sprint 0)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deerflow.subagents.handoff_schemas import (
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
)
from deerflow.tools.builtins.seal_handoff_tools import _build_task_context, _seal_handoff


def _make_runtime(workspace_dir: str) -> MagicMock:
    """Create a mock Runtime with workspace_path in thread_data."""
    runtime = MagicMock()
    runtime.state = {
        "thread_data": {
            "workspace_path": workspace_dir,
        },
    }
    return runtime


def _make_workspace(tmp_path: Path) -> Path:
    """Create a workspace directory with an experiment-context.json."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"paradigm": "fst", "analysis_config_id": "test-config-id"}),
        encoding="utf-8",
    )
    return ws


class TestSealCodeExecutorHappyPath:
    def test_seal_writes_file_and_manifest(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        result = _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {
                "status": "completed",
                "summary": "all good",
                "paradigm": "fst",
                "metrics_summary": {"control": {"mean_nnd": {"mean": 42.5}}},
            },
            runtime,
        )

        assert result.startswith("OK: sealed handoff_code_executor.json")

        handoff_path = ws / "handoff_code_executor.json"
        assert handoff_path.exists()
        data = json.loads(handoff_path.read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["paradigm"] == "fst"
        assert data["analysis_config_id"] == "test-config-id"

        manifest_path = ws / ".lineage" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "handoff_code_executor.json" in manifest
        entry = manifest["handoff_code_executor.json"]
        assert entry["sha256"]
        assert entry["analysis_config_id"] == "test-config-id"
        assert "timestamp" in entry


class TestSealChartMakerPathValidation:
    def test_bad_chart_path_raises(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        with pytest.raises(ValueError, match="schema validation failed"):
            _seal_handoff(
                ChartMakerHandoff,
                "handoff_chart_maker.json",
                {
                    "paradigm": "fst",
                    "summary": "bad paths",
                    "chart_files": ["/mnt/user-data/workspace/x.png"],
                },
                runtime,
            )


class TestSealDataAnalystMinimumFields:
    def test_only_status_required(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        result = _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {"status": "completed"},
            runtime,
        )
        assert result.startswith("OK: sealed handoff_data_analyst.json")

        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["key_findings"] == []


class TestSealAtomicWriteNoPartial:
    def test_tmp_not_left_on_rename_failure(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        payload = {
            "status": "completed",
            "summary": "test",
            "paradigm": "fst",
        }

        # Pre-create the target file as a directory to make os.rename fail
        (ws / "handoff_code_executor.json").mkdir()

        with pytest.raises(OSError):
            _seal_handoff(CodeExecutorHandoff, "handoff_code_executor.json", payload, runtime)

        assert (ws / "handoff_code_executor.json.tmp").exists()
        assert (ws / "handoff_code_executor.json").is_dir()


class TestManifestSha256:
    def test_manifest_includes_sha256_and_config_id(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {"status": "completed", "summary": "s", "paradigm": "fst"},
            runtime,
        )

        manifest = json.loads((ws / ".lineage" / "manifest.json").read_text(encoding="utf-8"))
        entry = manifest["handoff_code_executor.json"]
        assert len(entry["sha256"]) == 64
        assert entry["analysis_config_id"] == "test-config-id"


class TestSealPendingConfigId:
    def test_no_context_file_uses_pending(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {"status": "completed", "summary": "s", "paradigm": "fst"},
            runtime,
        )

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        assert data["analysis_config_id"] == "PENDING_SPRINT_4.5"


class TestManifestConcurrent:
    def test_two_seals_both_in_manifest(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {"status": "completed", "summary": "s", "paradigm": "fst"},
            runtime,
        )
        _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {"status": "completed"},
            runtime,
        )

        manifest = json.loads((ws / ".lineage" / "manifest.json").read_text(encoding="utf-8"))
        assert "handoff_code_executor.json" in manifest
        assert "handoff_data_analyst.json" in manifest


# ============================================================================
# _build_task_context 纯函数单元测试 (spec §四.5)
# ============================================================================


class TestBuildTaskContextFileChanges:
    """output_files → file_changes + verify_commands."""

    def test_extracts_paths_from_output_files(self):
        payload = {
            "output_files": {
                "metrics": "/mnt/user-data/outputs/metrics.json",
                "charts": "/mnt/user-data/outputs/chart.png",
            },
        }
        tc = _build_task_context(payload)
        assert tc["file_changes"] == [
            "/mnt/user-data/outputs/metrics.json",
            "/mnt/user-data/outputs/chart.png",
        ]

    def test_handles_list_values_in_output_files(self):
        payload = {
            "output_files": {
                "charts": [
                    "/mnt/user-data/outputs/heatmap.png",
                    "/mnt/user-data/outputs/boxplot.png",
                ],
            },
        }
        tc = _build_task_context(payload)
        assert tc["file_changes"] == [
            "/mnt/user-data/outputs/heatmap.png",
            "/mnt/user-data/outputs/boxplot.png",
        ]

    def test_json_files_get_json_tool_verify_command(self):
        payload = {
            "output_files": {
                "stats": "/mnt/user-data/outputs/statistics.json",
            },
        }
        tc = _build_task_context(payload)
        assert tc["verify_commands"] == ["python -m json.tool /mnt/user-data/outputs/statistics.json > /dev/null"]

    def test_non_json_files_get_ls_verify_command(self):
        payload = {
            "output_files": {
                "chart": "/mnt/user-data/outputs/plot.png",
            },
        }
        tc = _build_task_context(payload)
        assert tc["verify_commands"] == ["ls /mnt/user-data/outputs/plot.png"]

    def test_empty_output_files_yields_empty_lists(self):
        tc = _build_task_context({})
        assert tc["file_changes"] == []
        assert tc["verify_commands"] == []


class TestBuildTaskContextFailedPaths:
    """errors → failed_paths."""

    def test_extracts_string_errors(self):
        payload = {
            "errors": [
                "group 'control' metric 'distance_moved': n=2 — underpowered",
                "Shapiro-Wilk test failed: too few samples",
            ],
        }
        tc = _build_task_context(payload)
        assert tc["failed_paths"] == payload["errors"]

    def test_filters_non_string_errors(self):
        payload = {"errors": ["valid error", 42, None, {"msg": "dict error"}]}
        tc = _build_task_context(payload)
        assert tc["failed_paths"] == ["valid error"]

    def test_empty_errors_yields_empty_list(self):
        tc = _build_task_context({"errors": []})
        assert tc["failed_paths"] == []


class TestBuildTaskContextPendingItems:
    """status=partial → pending_items 诚实留空（真实 partial 为统计跳过，errors 恒空）。"""

    def test_partial_status_yields_empty_pending_items(self):
        payload = {
            "status": "partial",
            "errors": ["metric X not computed", "metric Y missing data"],
        }
        tc = _build_task_context(payload)
        # 真实 partial 时 errors 恒空（partial 实为统计跳过非脚本失败），pending_items 诚实留空
        assert tc["pending_items"] == []

    def test_completed_status_yields_empty_pending(self):
        payload = {
            "status": "completed",
            "errors": ["some warning"],
        }
        tc = _build_task_context(payload)
        assert tc["failed_paths"] == ["some warning"]  # errors still → failed_paths
        assert tc["pending_items"] == []  # but NOT pending (status != partial)

    def test_failed_status_yields_empty_pending(self):
        payload = {
            "status": "failed",
            "errors": ["catastrophic failure"],
        }
        tc = _build_task_context(payload)
        assert tc["pending_items"] == []


class TestBuildTaskContextDefensive:
    """异常输入不抛、返回部分结果."""

    def test_empty_payload_returns_four_empty_lists(self):
        tc = _build_task_context({})
        assert tc == {
            "file_changes": [],
            "verify_commands": [],
            "failed_paths": [],
            "pending_items": [],
        }

    def test_none_output_files_does_not_crash(self):
        tc = _build_task_context({"output_files": None})
        assert tc["file_changes"] == []
        assert tc["verify_commands"] == []

    def test_none_errors_does_not_crash(self):
        tc = _build_task_context({"errors": None})
        assert tc["failed_paths"] == []
        assert tc["pending_items"] == []

    def test_none_status_does_not_crash(self):
        tc = _build_task_context({"status": None, "errors": ["e1"]})
        assert tc["pending_items"] == []


class TestSealIntegrationTaskContext:
    """seal 端集成测试：seal 后落盘 JSON 含 task_context 且值正确."""

    def test_seal_injects_task_context(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {
                "status": "completed",
                "summary": "all metrics computed",
                "paradigm": "fst",
                "output_files": {
                    "metrics": "/mnt/user-data/outputs/metrics.json",
                    "chart": "/mnt/user-data/outputs/heatmap.png",
                },
                "errors": ["group B n=1 skipped"],
            },
            runtime,
        )

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        tc = data["task_context"]
        assert tc is not None
        assert tc["file_changes"] == [
            "/mnt/user-data/outputs/metrics.json",
            "/mnt/user-data/outputs/heatmap.png",
        ]
        assert "python -m json.tool" in tc["verify_commands"][0]
        assert "ls" in tc["verify_commands"][1]
        assert tc["failed_paths"] == ["group B n=1 skipped"]
        assert tc["pending_items"] == []  # status=completed → 空

    def test_seal_partial_status_pending_items_empty(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {
                "status": "partial",
                "summary": "partial run",
                "paradigm": "fst",
                "output_files": {
                    "metrics": "/mnt/user-data/outputs/metrics.json",
                },
                "errors": ["metric X not computed - too few frames"],
            },
            runtime,
        )

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        tc = data["task_context"]
        # 真实 partial 时 pending_items 诚实留空（partial 为统计跳过，errors 恒空）
        assert tc["pending_items"] == []

    def test_seal_forward_compat_explicit_task_context(self, tmp_path):
        """显式提供 task_context 时不被覆盖（payload.setdefault 语义）."""
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {
                "status": "completed",
                "summary": "test",
                "paradigm": "fst",
                "task_context": {"file_changes": ["/custom/path.txt"], "verify_commands": [], "failed_paths": [], "pending_items": []},
            },
            runtime,
        )

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        tc = data["task_context"]
        # setdefault 语义：已显式提供 → 不覆盖
        assert tc["file_changes"] == ["/custom/path.txt"]
