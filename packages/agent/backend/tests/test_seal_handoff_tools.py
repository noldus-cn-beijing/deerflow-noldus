"""Unit tests for seal_handoff_tools.py (Sprint 0)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deerflow.subagents.handoff_schemas import (
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
)
from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff


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


class TestSealLegacyWarningFallback:
    """Sprint 0/1 transition: auto-inject LEGACY.UNCATEGORIZED for warnings without code."""

    def test_auto_injects_legacy_code(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        result = _seal_handoff(
            CodeExecutorHandoff,
            "handoff_code_executor.json",
            {
                "status": "completed",
                "summary": "s",
                "paradigm": "fst",
                "data_quality_warnings": [
                    {"severity": "critical", "metric": "all", "message": "sample too small", "code": "LEGACY.UNCATEGORIZED", "evidence": {}, "blocks_downstream": True},
                ],
            },
            runtime,
        )
        assert result.startswith("OK:")

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        w = data["data_quality_warnings"][0]
        assert w["code"] == "LEGACY.UNCATEGORIZED"
        assert w["evidence"] == {}
        assert w["blocks_downstream"] is True


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
