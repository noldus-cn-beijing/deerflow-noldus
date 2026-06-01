"""Tests for Sprint 4.5: analysis_config_id + parameter_overrides.

TDD tests for:
1. compute_analysis_config_id: deterministic hash of (catalog_default + overrides)
2. set_experiment_paradigm with parameter_overrides
3. analysis_config_id in handoff schemas
"""

import json
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime

from deerflow.agents.middlewares.experiment_context import (
    compute_analysis_config_id,
    set_experiment_paradigm_tool,
)


def _runtime_with_workspace(workspace: Path) -> ToolRuntime:
    """Create a minimal ToolRuntime with workspace_path injected into state."""
    return ToolRuntime(
        state={"thread_data": {"workspace_path": str(workspace)}},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


# ---------------------------------------------------------------------------
# 1. compute_analysis_config_id pure-function tests
# ---------------------------------------------------------------------------


class TestComputeAnalysisConfigId:
    """Test the deterministic analysis_config_id hash function."""

    def test_same_input_same_id(self):
        """Deterministic: identical inputs always produce the same id."""
        id1 = compute_analysis_config_id({"paradigm": "epm"}, {"threshold": 0.5})
        id2 = compute_analysis_config_id({"paradigm": "epm"}, {"threshold": 0.5})
        assert id1 == id2

    def test_different_input_different_id(self):
        """Different overrides produce different ids."""
        id1 = compute_analysis_config_id({"paradigm": "epm"}, {"threshold": 0.5})
        id2 = compute_analysis_config_id({"paradigm": "epm"}, {"threshold": 0.8})
        assert id1 != id2

    def test_different_defaults_different_id(self):
        """Different catalog defaults produce different ids."""
        id1 = compute_analysis_config_id({"paradigm": "epm"}, {})
        id2 = compute_analysis_config_id({"paradigm": "fst"}, {})
        assert id1 != id2

    def test_empty_overrides_still_produces_id(self):
        """Empty overrides is a valid input; still produces a deterministic id."""
        config_id = compute_analysis_config_id({"paradigm": "epm"}, {})
        assert config_id
        assert isinstance(config_id, str)
        assert len(config_id) == 16  # 16 hex chars (64 bits)

    def test_id_is_hex_string(self):
        """The id must be a 16-char hex string."""
        config_id = compute_analysis_config_id({"paradigm": "epm"}, {"n_per_group": 8})
        assert len(config_id) == 16
        # Must be valid hex
        int(config_id, 16)

    def test_key_order_irrelevant(self):
        """Override key ordering should not affect the hash."""
        id1 = compute_analysis_config_id({"paradigm": "epm"}, {"a": 1, "b": 2})
        id2 = compute_analysis_config_id({"paradigm": "epm"}, {"b": 2, "a": 1})
        assert id1 == id2

    def test_default_key_order_irrelevant(self):
        """Default key ordering should not affect the hash."""
        id1 = compute_analysis_config_id({"a": 1, "b": 2}, {})
        id2 = compute_analysis_config_id({"b": 2, "a": 1}, {})
        assert id1 == id2


# ---------------------------------------------------------------------------
# 2. set_experiment_paradigm_tool with parameter_overrides
# ---------------------------------------------------------------------------


class TestSetExperimentParadigmOverrides:
    """Test that set_experiment_paradigm stores overrides and computes config id."""

    def test_gate1_stores_overrides_and_config_id(self, tmp_path):
        """Gate 1 with parameter_overrides stores them + computes analysis_config_id."""
        runtime = _runtime_with_workspace(tmp_path)
        result_str = set_experiment_paradigm_tool.invoke(
            {
                "paradigm": "epm",
                "paradigm_cn": "高架十字迷宫",
                "category": "anxiety",
                "subject": "rodent",
                "ev19_template": "PlusMaze-AllZones",
                "parameter_overrides": {"immobility_threshold": 0.5},
                "runtime": runtime,
            },
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"

        # Verify experiment-context.json has overrides + config id
        ctx = json.loads((tmp_path / "experiment-context.json").read_text())
        assert "parameter_overrides" in ctx
        assert ctx["parameter_overrides"] == {"immobility_threshold": 0.5}
        assert "analysis_config_id" in ctx
        assert len(ctx["analysis_config_id"]) == 16

    def test_gate1_without_overrides_stores_empty_and_config_id(self, tmp_path):
        """Gate 1 without overrides stores empty dict + still computes config id."""
        runtime = _runtime_with_workspace(tmp_path)
        result_str = set_experiment_paradigm_tool.invoke(
            {
                "paradigm": "epm",
                "paradigm_cn": "高架十字迷宫",
                "category": "anxiety",
                "subject": "rodent",
                "ev19_template": "PlusMaze-AllZones",
                "runtime": runtime,
            },
        )
        result = json.loads(result_str)
        assert result["status"] == "ok"

        ctx = json.loads((tmp_path / "experiment-context.json").read_text())
        assert ctx["parameter_overrides"] == {}
        assert len(ctx["analysis_config_id"]) == 16

    def test_config_id_deterministic_across_calls(self, tmp_path):
        """Same paradigm + same overrides → same analysis_config_id."""
        runtime = _runtime_with_workspace(tmp_path)

        # First call
        set_experiment_paradigm_tool.invoke(
            {
                "paradigm": "epm",
                "paradigm_cn": "高架十字迷宫",
                "category": "anxiety",
                "subject": "rodent",
                "ev19_template": "PlusMaze-AllZones",
                "parameter_overrides": {"threshold": 0.5},
                "runtime": runtime,
            },
        )
        ctx1 = json.loads((tmp_path / "experiment-context.json").read_text())

        # Remove file and call again with same params
        (tmp_path / "experiment-context.json").unlink()
        set_experiment_paradigm_tool.invoke(
            {
                "paradigm": "epm",
                "paradigm_cn": "高架十字迷宫",
                "category": "anxiety",
                "subject": "rodent",
                "ev19_template": "PlusMaze-AllZones",
                "parameter_overrides": {"threshold": 0.5},
                "runtime": runtime,
            },
        )
        ctx2 = json.loads((tmp_path / "experiment-context.json").read_text())

        assert ctx1["analysis_config_id"] == ctx2["analysis_config_id"]

    def test_gate2_preserves_overrides_and_config_id(self, tmp_path):
        """Gate 2 (acknowledge_quality) preserves overrides and config_id from Gate 1."""
        runtime = _runtime_with_workspace(tmp_path)

        # Gate 1 with overrides
        set_experiment_paradigm_tool.invoke(
            {
                "paradigm": "epm",
                "paradigm_cn": "高架十字迷宫",
                "category": "anxiety",
                "subject": "rodent",
                "ev19_template": "PlusMaze-AllZones",
                "parameter_overrides": {"threshold": 0.5},
                "runtime": runtime,
            },
        )
        ctx1 = json.loads((tmp_path / "experiment-context.json").read_text())
        original_config_id = ctx1["analysis_config_id"]

        # Gate 2
        set_experiment_paradigm_tool.invoke(
            {
                "acknowledge_quality": True,
                "runtime": runtime,
            },
        )
        ctx2 = json.loads((tmp_path / "experiment-context.json").read_text())

        assert ctx2["parameter_overrides"] == {"threshold": 0.5}
        assert ctx2["analysis_config_id"] == original_config_id
        assert "gate2_quality_acknowledged" in ctx2["gate_completed"]


# ---------------------------------------------------------------------------
# 3. analysis_config_id in handoff schemas
# ---------------------------------------------------------------------------


class TestHandoffSchemaConfigId:
    """Test that handoff schemas include analysis_config_id field."""

    def test_code_executor_handoff_has_config_id(self):
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        h = CodeExecutorHandoff(status="completed", summary="ok", paradigm="epm", analysis_config_id="abcd1234efgh5678")
        assert h.analysis_config_id == "abcd1234efgh5678"

    def test_data_analyst_handoff_has_config_id(self):
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        h = DataAnalystHandoff(status="completed", key_findings=["test"], analysis_config_id="abcd1234efgh5678")
        assert h.analysis_config_id == "abcd1234efgh5678"

    def test_report_writer_handoff_has_config_id(self):
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        h = ReportWriterHandoff(status="completed", report_path="/path", analysis_config_id="abcd1234efgh5678")
        assert h.analysis_config_id == "abcd1234efgh5678"

    def test_chart_maker_handoff_has_config_id(self):
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        h = ChartMakerHandoff(paradigm="epm", summary="ok", analysis_config_id="abcd1234efgh5678")
        assert h.analysis_config_id == "abcd1234efgh5678"

    def test_config_id_defaults_to_pending(self):
        """analysis_config_id defaults to 'PENDING' when not explicitly provided."""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        h = CodeExecutorHandoff(status="completed", summary="ok", paradigm="epm")
        assert h.analysis_config_id == "PENDING"

    def test_old_handoff_without_config_id_still_validates(self):
        """Handoff JSON without analysis_config_id still validates (default='PENDING')."""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        raw = {"status": "completed", "summary": "ok", "paradigm": "epm", "metrics_summary": {}}
        h = CodeExecutorHandoff.model_validate(raw)
        assert h.analysis_config_id == "PENDING"


# ---------------------------------------------------------------------------
# 4. prep_metric_plan integration: reads config_id from experiment-context.json
# ---------------------------------------------------------------------------


class TestPrepMetricPlanConfigId:
    """Test that prep_metric_plan reads analysis_config_id from experiment-context.json."""

    def test_plan_includes_config_id_from_context(self, tmp_path):
        """When experiment-context.json has analysis_config_id, plan includes it."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Write experiment-context.json with a known config id
        ctx = {
            "paradigm": "epm",
            "analysis_config_id": "abcdef1234567890",
            "parameter_overrides": {"threshold": 0.5},
        }
        (workspace / "experiment-context.json").write_text(json.dumps(ctx))

        # Create a minimal EthoVision file for EPM
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        data_file = uploads / "test_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        from langchain.tools import ToolRuntime

        runtime = ToolRuntime(
            state={
                "thread_data": {
                    "workspace_path": str(workspace),
                    "uploads_path": str(uploads),
                },
            },
            context=None,
            config={},
            stream_writer=None,
            tool_call_id="test-id",
            store=None,
        )

        from deerflow.tools.builtins.prep_metric_plan_tool import prep_metric_plan_tool

        result = prep_metric_plan_tool.invoke(
            {
                "uploaded_files": ["/mnt/user-data/uploads/test_epm.txt"],
                "paradigm": "epm",
                "runtime": runtime,
            },
        )

        assert result["status"] == "ok"
        assert result["analysis_config_id"] == "abcdef1234567890"

        # Check plan_metrics.json also contains it
        plan_data = json.loads((workspace / "plan_metrics.json").read_text())
        assert plan_data["analysis_config_id"] == "abcdef1234567890"
        assert plan_data["parameter_overrides"] == {"threshold": 0.5}

    def test_plan_works_without_context(self, tmp_path):
        """When experiment-context.json is missing, config_id defaults to PENDING."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        uploads = tmp_path / "uploads"
        uploads.mkdir()
        data_file = uploads / "test_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        from langchain.tools import ToolRuntime

        runtime = ToolRuntime(
            state={
                "thread_data": {
                    "workspace_path": str(workspace),
                    "uploads_path": str(uploads),
                },
            },
            context=None,
            config={},
            stream_writer=None,
            tool_call_id="test-id",
            store=None,
        )

        from deerflow.tools.builtins.prep_metric_plan_tool import prep_metric_plan_tool

        result = prep_metric_plan_tool.invoke(
            {
                "uploaded_files": ["/mnt/user-data/uploads/test_epm.txt"],
                "paradigm": "epm",
                "runtime": runtime,
            },
        )

        assert result["status"] == "ok"
        assert result["analysis_config_id"] == "PENDING"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# EthoVision columns for EPM (must match catalog expectations)
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


def _write_ethovision_file(path: str, columns: list[str]) -> None:
    """Write a UTF-16 LE EthoVision trajectory file with full metadata header.

    Reuses the same format as test_prep_metric_plan_tool.
    """
    header_lines = 36
    lines: list[str] = []
    # Line 1: header line count
    lines.append(f'"Number of header lines:";"{header_lines}"')
    # Lines 2..N: metadata key-value pairs
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
