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
            {"status": "completed", "key_findings": ["finding"]},
            runtime,
        )
        assert result.startswith("OK: sealed handoff_data_analyst.json")

        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["key_findings"] == ["finding"]


class TestSealAtomicWriteNoPartial:
    def test_tmp_not_left_on_rename_failure(self, tmp_path):
        ws = _make_workspace(tmp_path)
        runtime = _make_runtime(str(ws))

        payload = {
            "status": "completed",
            "summary": "test",
            "paradigm": "fst",
            "metrics_summary": {"g": {"m": {"mean": 1.0}}},
        }
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
            {"status": "completed", "summary": "s", "paradigm": "fst", "metrics_summary": {"g": {"m": {"mean": 1.0}}}},
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
            {"status": "completed", "summary": "s", "paradigm": "fst", "metrics_summary": {"g": {"m": {"mean": 1.0}}}},
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
            {"status": "completed", "summary": "s", "paradigm": "fst", "metrics_summary": {"g": {"m": {"mean": 1.0}}}},
            runtime,
        )
        _seal_handoff(
            DataAnalystHandoff,
            "handoff_data_analyst.json",
            {"status": "completed", "key_findings": ["f"]},
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
                "metrics_summary": {"g": {"m": {"mean": 1.0}}},
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
                "metrics_summary": {"g": {"m": {"mean": 1.0}}},
                "task_context": {"file_changes": ["/custom/path.txt"], "verify_commands": [], "failed_paths": [], "pending_items": []},
            },
            runtime,
        )

        data = json.loads((ws / "handoff_code_executor.json").read_text(encoding="utf-8"))
        tc = data["task_context"]
        # setdefault 语义：已显式提供 → 不覆盖
        assert tc["file_changes"] == ["/custom/path.txt"]


# ---------------------------------------------------------------------------
# _normalize_report_image_paths — server-side path normalisation (2026-06-04)
# Locks the three LLM-written variants + idempotency + skip-if-missing.
# ---------------------------------------------------------------------------
from pathlib import Path
import tempfile

from deerflow.tools.builtins.seal_handoff_tools import _normalize_report_image_paths


class TestNormalizeReportImagePaths:
    """Server-side normalization of markdown image paths before sealing.

    Artifacts API (resolve_thread_virtual_path) requires mnt/user-data/outputs/…
    (no leading slash). LLMs write three broken variants; all must become canonical.
    """

    def _run(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)
        _normalize_report_image_paths(path)
        result = path.read_text(encoding="utf-8")
        path.unlink(missing_ok=True)
        return result

    def test_relative_outputs_prefix_fixed(self):
        md = "![Figure 1](outputs/boxplot.png)"
        assert self._run(md) == "![Figure 1](mnt/user-data/outputs/boxplot.png)"

    def test_absolute_virtual_leading_slash_fixed(self):
        md = "![Figure 1](/mnt/user-data/outputs/plot_s0.png)"
        assert self._run(md) == "![Figure 1](mnt/user-data/outputs/plot_s0.png)"

    def test_correct_form_unchanged(self):
        md = "![Figure 1](mnt/user-data/outputs/plot_s0.png)"
        assert self._run(md) == md  # idempotent

    def test_multiple_images_all_fixed(self):
        md = (
            "![A](outputs/plot_s0.png)\n"
            "![B](/mnt/user-data/outputs/plot_s1.png)\n"
            "![C](mnt/user-data/outputs/plot_s2.png)\n"
        )
        result = self._run(md)
        assert result.count("mnt/user-data/outputs/") == 3
        assert "outputs/plot_s0" not in result.split("mnt/user-data/outputs/")[0]
        assert "(/mnt/" not in result

    def test_non_image_links_untouched(self):
        md = "[link](outputs/data.csv) and ![img](outputs/chart.png)"
        result = self._run(md)
        assert "(outputs/data.csv)" in result  # non-image link preserved
        assert "(mnt/user-data/outputs/chart.png)" in result

    def test_missing_file_silently_skipped(self):
        # Must not raise
        _normalize_report_image_paths(Path("/tmp/nonexistent_report_xyz.md"))


# ---------------------------------------------------------------------------
# _resolve_report_image_placeholders — chart image placeholder resolution (2026-06-05)
# Layer 1 of the two-layer defence: resolves {{img:<basename>}} placeholders
# to canonical virtual paths from handoff_chart_maker.json.chart_files.
# ---------------------------------------------------------------------------
from deerflow.tools.builtins.seal_handoff_tools import (
    _load_chart_files_map,
    _resolve_report_image_placeholders,
)


class TestResolveReportImagePlaceholders:
    """Server-side resolution of {{img:<basename>}} placeholders before sealing.

    Chart-maker writes handoff_chart_maker.json with chart_files; LLM writes
    {{img:<basename>}} in report.md; seal_report_writer_handoff resolves them.
    """

    def _make_chart_handoff(self, workspace: Path, chart_files: list[str]) -> None:
        """Write a handoff_chart_maker.json to workspace."""
        data = {
            "status": "completed",
            "paradigm": "epm",
            "summary": "charts done",
            "chart_files": chart_files,
        }
        (workspace / "handoff_chart_maker.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    def _write_report(self, outputs_dir: Path, content: str) -> Path:
        report = outputs_dir / "report.md"
        report.write_text(content, encoding="utf-8")
        return report

    def test_resolves_valid_basename(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_trajectory_s0.png"])
        report = self._write_report(out, "![Figure 1]({{img:plot_trajectory_s0.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert result == "![Figure 1](mnt/user-data/outputs/plot_trajectory_s0.png)"

    def test_resolves_multiple_placeholders(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, [
            "/mnt/user-data/outputs/plot_s0.png",
            "/mnt/user-data/outputs/box_open_arm.png",
        ])
        report = self._write_report(
            out,
            "![A]({{img:plot_s0.png}})\n![B]({{img:box_open_arm.png}})",
        )

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "mnt/user-data/outputs/plot_s0.png" in result
        assert "mnt/user-data/outputs/box_open_arm.png" in result
        assert "{{img:" not in result

    def test_unmatched_basename_stub(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/real_chart.png"])
        report = self._write_report(out, "![X]({{img:invented_name.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "图表 'invented_name.png' 未找到" in result
        assert "可用: real_chart.png" in result

    def test_missing_handoff_file_noop(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        # No handoff_chart_maker.json
        report = self._write_report(out, "![X]({{img:anything.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "{{img:anything.png}}" in result  # preserved as-is

    def test_empty_chart_files_noop(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, [])
        report = self._write_report(out, "![X]({{img:anything.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "{{img:anything.png}}" in result  # preserved as-is

    def test_missing_report_file_noop(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/x.png"])
        # Don't create report.md — must not raise
        _resolve_report_image_placeholders(Path("/tmp/no_such_report.md"), ws)

    def test_basename_exact_match_only(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_trajectory_s0.png"])
        report = self._write_report(out, "![X]({{img:trajectory.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        # "trajectory" ≠ "plot_trajectory_s0" — partial match should NOT resolve
        assert "图表 'trajectory.png' 未找到" in result

    def test_idempotent_on_already_resolved(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot.png"])
        # Already has correct path, no placeholder
        report = self._write_report(
            out, "![X](mnt/user-data/outputs/plot.png)"
        )

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert result == "![X](mnt/user-data/outputs/plot.png)"

    def test_leading_slash_stripped(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot.png"])
        report = self._write_report(out, "![X]({{img:plot.png}})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        # Value must NOT start with / (artifacts API requires no leading slash)
        assert "](/mnt/" not in result
        assert "](mnt/user-data/outputs/plot.png)" in result

    def test_whitespace_in_placeholder(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_X.png"])
        report = self._write_report(out, "![X]({{img: plot_X.png }})")

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "mnt/user-data/outputs/plot_X.png" in result

    def test_mixed_placeholders_and_literal(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_s0.png"])
        report = self._write_report(
            out,
            "![A]({{img:plot_s0.png}})\n"
            "![B](outputs/legacy_chart.png)\n"
            "![C](/mnt/user-data/outputs/legacy_abs.png)",
        )

        _resolve_report_image_placeholders(report, ws)

        # Layer 1 resolves placeholder
        result = report.read_text(encoding="utf-8")
        assert "mnt/user-data/outputs/plot_s0.png" in result
        # Layer 1 does NOT touch literal paths (Layer 2 handles those)
        assert "outputs/legacy_chart.png" in result
        assert "/mnt/user-data/outputs/legacy_abs.png" in result

    def test_corrupt_handoff_json(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        (ws / "handoff_chart_maker.json").write_text("not valid json {{{")
        report = self._write_report(out, "![X]({{img:plot.png}})")

        # Must not raise
        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert "{{img:plot.png}}" in result  # preserved as-is

    def test_no_placeholders_in_content(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot.png"])
        original = "# Report\n\nSome text without any placeholders.\n"
        report = self._write_report(out, original)

        _resolve_report_image_placeholders(report, ws)

        result = report.read_text(encoding="utf-8")
        assert result == original  # no unintended side-effects


class TestLoadChartFilesMap:
    """Unit tests for _load_chart_files_map helper."""

    def test_returns_basename_to_path_map(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "handoff_chart_maker.json").write_text(json.dumps({
            "chart_files": [
                "/mnt/user-data/outputs/plot_s0.png",
                "/mnt/user-data/outputs/box_open_arm.png",
            ],
        }))

        result = _load_chart_files_map(ws)

        assert result == {
            "plot_s0.png": "mnt/user-data/outputs/plot_s0.png",
            "box_open_arm.png": "mnt/user-data/outputs/box_open_arm.png",
        }

    def test_missing_file_returns_empty(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        assert _load_chart_files_map(ws) == {}

    def test_empty_chart_files_returns_empty(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "handoff_chart_maker.json").write_text(json.dumps({"chart_files": []}))
        assert _load_chart_files_map(ws) == {}

    def test_chart_files_not_list_returns_empty(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "handoff_chart_maker.json").write_text(json.dumps({"chart_files": "not a list"}))
        assert _load_chart_files_map(ws) == {}

    def test_non_string_entries_filtered(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "handoff_chart_maker.json").write_text(json.dumps({
            "chart_files": ["/mnt/user-data/outputs/real.png", 42, None],
        }))

        result = _load_chart_files_map(ws)

        assert result == {"real.png": "mnt/user-data/outputs/real.png"}
