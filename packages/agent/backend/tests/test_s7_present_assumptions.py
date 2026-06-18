"""Unit tests for Sprint 7: present_assumptions_tool."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from deerflow.tools.builtins.present_assumptions import (
    _build_assumptions_markdown,
    present_assumptions_tool,
)


def _make_runtime(workspace_dir: str) -> MagicMock:
    """Create a mock Runtime with workspace_path in thread_data."""
    runtime = MagicMock()
    runtime.state = {
        "thread_data": {
            "workspace_path": workspace_dir,
        },
    }
    return runtime


def _make_workspace(
    tmp_path: Path,
    *,
    config_id: str = "abc1234567890abcd",
    paradigm: str = "epm",
    overrides: dict | None = None,
    gates: list[str] | None = None,
    params_in_use: dict | None = None,
    quality_warnings: list[dict] | None = None,
    audit_findings: list[dict] | None = None,
) -> Path:
    """Create a workspace with configurable experiment-context, plan, and handoff."""
    ws = tmp_path / "workspace"
    ws.mkdir()

    ctx = {
        "paradigm": paradigm,
        "analysis_config_id": config_id,
        "parameter_overrides": overrides or {},
        "gate_completed": gates or ["gate1_paradigm"],
    }
    (ws / "experiment-context.json").write_text(json.dumps(ctx, ensure_ascii=False), encoding="utf-8")

    if params_in_use is not None:
        plan = {"parameters_in_use": params_in_use}
        (ws / "plan_metrics.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    if quality_warnings is not None or audit_findings is not None:
        da = {
            "status": "completed",
            "quality_warnings": quality_warnings or [],
            "parameter_audit_findings": audit_findings or [],
        }
        (ws / "handoff_data_analyst.json").write_text(json.dumps(da, ensure_ascii=False), encoding="utf-8")

    return ws


# ============================================================================
# Test _build_assumptions_markdown (pure function, no IO)
# ============================================================================


class TestBuildAssumptionsMarkdown:
    def test_empty_when_all_defaults(self):
        """All-default run (no overrides, no warnings, no audit) → empty string.

        A simple analysis must NOT render a redundant panel (brief §7:
        "简单分析 lead 不调"). Only overrides/warnings/audit count as content.
        """
        ctx = {"analysis_config_id": "abc", "parameter_overrides": {}, "gate_completed": ["gate1_paradigm"]}
        result = _build_assumptions_markdown(ctx, None, None)
        assert result == ""

    def test_empty_when_all_none(self):
        """All inputs None → empty string."""
        result = _build_assumptions_markdown(None, None, None)
        assert result == ""

    def test_config_id_in_summary(self):
        ctx = {"analysis_config_id": "deadbeef12345678", "parameter_overrides": {"threshold": 0.75}}
        result = _build_assumptions_markdown(ctx, None, None)
        assert "config_id=deadbeef12345678" in result

    def test_overrides_rendered(self):
        ctx = {"analysis_config_id": "x", "parameter_overrides": {"immobility_threshold": 0.5, "n_per_group": 10}}
        result = _build_assumptions_markdown(ctx, None, None)
        assert "`immobility_threshold`: `0.5`" in result
        assert "`n_per_group`: `10`" in result

    def test_parameters_in_use_rendered(self):
        plan = {"parameters_in_use": {"alpha": 0.05, "test": "mann-whitney"}}
        result = _build_assumptions_markdown(None, plan, None)
        assert "运行时参数" in result
        assert "`alpha`: `0.05`" in result

    def test_quality_warnings_critical_rendered(self):
        da = {
            "quality_warnings": [
                {"severity": "critical", "code": "LOW_N", "message": "n=1 per group", "blocks_downstream": True},
                {"severity": "warning", "code": "MISSING_DATA", "message": "some missing"},
            ],
        }
        result = _build_assumptions_markdown(None, None, da)
        assert "critical warnings: **1** 条" in result
        assert "blocks_downstream: **1** 条" in result
        assert "`LOW_N`: n=1 per group" in result

    def test_audit_findings_rendered(self):
        da = {
            "parameter_audit_findings": [
                {"parameter": "immobility_threshold", "severity": "critical"},
                {"parameter": "alpha", "severity": "warning"},
            ],
        }
        result = _build_assumptions_markdown(None, None, da)
        assert "findings: **2** 条" in result
        assert "critical: **1** 条" in result
        assert "`immobility_threshold` [critical]" in result

    def test_audit_findings_capped_at_5(self):
        findings = [{"parameter": f"param_{i}", "severity": "warning"} for i in range(8)]
        da = {"parameter_audit_findings": findings}
        result = _build_assumptions_markdown(None, None, da)
        assert "... and 3 more" in result

    def test_empty_audit_findings_not_rendered(self):
        """parameter_audit_findings == []（data-analyst 2026-06-18 起恒传）→ 不渲染参数审计段。

        spec §4.5：data-analyst 不再产出参数审计，handoff 里该字段恒为 []。即使同时有
        quality_warnings（渲染数据质量段），也不得误渲染空的参数审计段。这是 present_assumptions
        对空数组 graceful 的显式坐实（test_empty_when_all_defaults 传 da=None 不够精确）。
        """
        da = {
            "quality_warnings": [
                {"severity": "critical", "code": "LOW_N", "message": "n=1", "blocks_downstream": True},
            ],
            "parameter_audit_findings": [],  # data-analyst 现恒传空数组
        }
        result = _build_assumptions_markdown(None, None, da)
        # 数据质量段仍渲染（quality_warnings 非空）
        assert "数据质量" in result
        # 参数审计段不得渲染（空数组 = 无内容）
        assert "参数审计" not in result

    def test_details_html_structure(self):
        ctx = {"analysis_config_id": "abc123", "parameter_overrides": {"x": 1}}
        result = _build_assumptions_markdown(ctx, None, None)
        assert result.startswith("<details>")
        assert result.strip().endswith("</details>")


class TestPresentAssumptionsTool:
    """Integration tests calling the tool with file IO."""

    def test_tool_reads_workspace_files(self, tmp_path):
        ws = _make_workspace(
            tmp_path,
            config_id="tool-test-id",
            overrides={"threshold": 0.5},
            quality_warnings=[{"severity": "critical", "code": "LOW_N", "message": "n too low", "blocks_downstream": True}],
        )
        runtime = _make_runtime(str(ws))
        result = present_assumptions_tool.func(workspace_dir="/mnt/user-data/workspace/", runtime=runtime)
        assert "config_id=tool-test-id" in result
        assert "`threshold`: `0.5`" in result
        assert "LOW_N" in result

    def test_tool_returns_empty_for_clean_analysis(self, tmp_path):
        ws = _make_workspace(tmp_path, config_id="clean", overrides={})
        # No plan, no da handoff, no overrides → nothing worth surfacing.
        runtime = _make_runtime(str(ws))
        result = present_assumptions_tool.func(workspace_dir="/mnt/user-data/workspace/", runtime=runtime)
        assert result == ""

    def test_tool_handles_missing_workspace(self, tmp_path):
        empty_ws = tmp_path / "nonexistent"
        empty_ws.mkdir()
        runtime = _make_runtime(str(empty_ws))
        result = present_assumptions_tool.func(workspace_dir="/mnt/user-data/workspace/", runtime=runtime)
        assert result == ""

    def test_tool_reads_plan_metrics(self, tmp_path):
        ws = _make_workspace(
            tmp_path,
            config_id="plan-test",
            params_in_use={"test": "shapiro-wilk", "alpha": 0.05},
        )
        runtime = _make_runtime(str(ws))
        result = present_assumptions_tool.func(workspace_dir="/mnt/user-data/workspace/", runtime=runtime)
        assert "运行时参数" in result
        assert "`test`: `shapiro-wilk`" in result

    def test_tool_reads_audit_findings(self, tmp_path):
        ws = _make_workspace(
            tmp_path,
            config_id="audit-test",
            audit_findings=[
                {"parameter": "velocity_threshold", "severity": "critical"},
            ],
        )
        runtime = _make_runtime(str(ws))
        result = present_assumptions_tool.func(workspace_dir="/mnt/user-data/workspace/", runtime=runtime)
        assert "参数审计" in result
        assert "`velocity_threshold` [critical]" in result
