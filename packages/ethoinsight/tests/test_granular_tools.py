"""Tests for the 5 granular analysis tools."""

from __future__ import annotations

import glob
import json
import os
import pickle
from pathlib import Path
from langchain_core.runnables import RunnableConfig

import pytest
from langchain.tools import ToolRuntime

from ethoinsight.templates.tool import (
    parse_trajectories_tool,
    compute_metrics_tool,
    run_statistics_tool,
    generate_charts_tool,
    assess_and_handoff_tool,
)


@pytest.fixture
def mock_runtime():
    """ToolRuntime with None thread_data (paths are physical already)."""
    return ToolRuntime(
        state={"thread_data": None},
        context=None,
        config=RunnableConfig(),
        stream_writer=None,
        tool_call_id=None,
        store=None,
    )


@pytest.fixture
def demo_file_pattern() -> str:
    """Pattern pointing at shoaling demo data."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    pattern = str(root / "demo-data" / "*.txt")
    if not glob.glob(pattern):
        pytest.skip(f"No demo data at {pattern}")
    return pattern


class TestParseTrajectories:
    def test_happy_path(self, mock_runtime, demo_file_pattern, tmp_path):
        workspace = str(tmp_path)
        result = parse_trajectories_tool.invoke({
            "runtime": mock_runtime,
            "file_pattern": demo_file_pattern,
            "workspace_dir": workspace,
        })
        payload = json.loads(result)
        assert payload["status"] == "completed"
        assert payload["n_files"] >= 1
        assert os.path.exists(payload["pkl_path"])
        summary_path = os.path.join(workspace, "parsed_summary.json")
        assert os.path.exists(summary_path)

    def test_no_matching_files(self, mock_runtime, tmp_path):
        result = parse_trajectories_tool.invoke({
            "runtime": mock_runtime,
            "file_pattern": str(tmp_path / "nonexistent_*.txt"),
            "workspace_dir": str(tmp_path),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "No files matched" in payload["error"]


class TestComputeMetrics:
    def test_missing_dependency(self, mock_runtime, tmp_path):
        result = compute_metrics_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": '{"control":["Subject 1"]}',
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "missing dependency" in payload["error"]

    def test_invalid_groups_json(self, mock_runtime, tmp_path):
        (tmp_path / "parsed.pkl").write_bytes(pickle.dumps({"summary": {"total_files": 1}}))
        result = compute_metrics_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": "not a json",
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "Invalid groups JSON" in payload["error"]

    def test_happy_path_shoaling(self, mock_runtime, demo_file_pattern, tmp_path):
        workspace = str(tmp_path / "ws")
        output = str(tmp_path / "out")
        os.makedirs(workspace, exist_ok=True)

        parse_trajectories_tool.invoke({
            "runtime": mock_runtime,
            "file_pattern": demo_file_pattern,
            "workspace_dir": workspace,
        })

        subjects = json.loads(open(os.path.join(workspace, "parsed_summary.json")).read())["subjects"]
        if len(subjects) < 4:
            pytest.skip("Need ≥4 subjects in demo data")
        groups = json.dumps({
            "control": subjects[:2],
            "treatment": subjects[2:4],
        })

        result = compute_metrics_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": groups,
            "workspace_dir": workspace,
            "output_dir": output,
        })
        payload = json.loads(result)
        assert payload["status"] == "completed"
        assert payload["paradigm"] == "shoaling"
        assert len(payload["computed_metrics"]) >= 1
        assert os.path.exists(payload["pkl_path"])


class TestRunStatistics:
    def test_missing_dependency(self, mock_runtime, tmp_path):
        result = run_statistics_tool.invoke({
            "runtime": mock_runtime,
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "missing dependency" in payload["error"]

    def test_single_group_fails(self, mock_runtime, tmp_path):
        m_result = {
            "paradigm": "shoaling",
            "per_subject": {"s1": {"m": 1.0}},
            "group_summary": {"control": {"m": {"mean": 1.0, "std": 0.0, "n": 1, "values": [1.0]}}},
            "metadata": {"computed_metrics": ["m"]},
        }
        (tmp_path / "metrics.pkl").write_bytes(pickle.dumps(m_result))
        result = run_statistics_tool.invoke({
            "runtime": mock_runtime,
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "at least 2 groups" in payload["error"]


class TestGenerateCharts:
    def test_missing_dependency(self, mock_runtime, tmp_path):
        result = generate_charts_tool.invoke({
            "runtime": mock_runtime,
            "chart_types": "box_plot",
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert "missing dependency" in payload["error"]


class TestAssessAndHandoff:
    def test_missing_dependency(self, mock_runtime, tmp_path):
        result = assess_and_handoff_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": '{"control":["s1"]}',
            "workspace_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "handoff_path": str(tmp_path / "handoff.json"),
        })
        payload = json.loads(result)
        assert payload["status"] == "failed"


class TestEndToEndShoaling:
    """Full pipeline on demo shoaling data."""

    def test_full_pipeline_writes_valid_handoff(self, mock_runtime, demo_file_pattern, tmp_path):
        workspace = str(tmp_path / "ws")
        output = str(tmp_path / "out")
        os.makedirs(workspace, exist_ok=True)

        # Step 1
        parse_result = json.loads(parse_trajectories_tool.invoke({
            "runtime": mock_runtime,
            "file_pattern": demo_file_pattern,
            "workspace_dir": workspace,
        }))
        assert parse_result["status"] == "completed"
        subjects = parse_result["subjects"]
        if len(subjects) < 4:
            pytest.skip("Need ≥4 subjects")
        groups_str = json.dumps({"control": subjects[:2], "treatment": subjects[2:4]})

        # Step 2
        compute_result = json.loads(compute_metrics_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": groups_str,
            "workspace_dir": workspace,
            "output_dir": output,
        }))
        assert compute_result["status"] == "completed"

        # Step 3
        stats_result = json.loads(run_statistics_tool.invoke({
            "runtime": mock_runtime,
            "workspace_dir": workspace,
            "output_dir": output,
        }))
        assert stats_result["status"] == "completed"

        # Step 4
        charts_result = json.loads(generate_charts_tool.invoke({
            "runtime": mock_runtime,
            "chart_types": "box_plot",
            "workspace_dir": workspace,
            "output_dir": output,
        }))
        assert charts_result["status"] == "completed"

        # Step 5
        handoff_result = json.loads(assess_and_handoff_tool.invoke({
            "runtime": mock_runtime,
            "paradigm": "shoaling",
            "groups": groups_str,
            "workspace_dir": workspace,
            "output_dir": output,
            "handoff_path": str(tmp_path / "handoff.json"),
        }))
        assert handoff_result["status"] == "completed"

        # Verify handoff schema
        handoff = json.loads(Path(handoff_result["handoff_path"]).read_text())
        assert handoff["status"] == "completed"
        assert "metrics_summary" in handoff
        assert "statistics" in handoff
        assert "assessment" in handoff
        assert "output_files" in handoff
        assert "metadata" in handoff
        assert handoff["metadata"]["paradigm"] == "shoaling"
        assert handoff["metadata"]["n_files"] >= 2
