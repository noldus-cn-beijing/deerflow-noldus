"""Unit test: assess_and_handoff_tool must write per_subject into handoff JSON.

Complements the end-to-end test in test_granular_tools.py which requires demo
data (and is skipped on many dev machines). This test constructs the required
workspace files directly so it runs everywhere.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig
from langchain.tools import ToolRuntime

from ethoinsight.templates.tool import assess_and_handoff_tool


@pytest.fixture
def mock_runtime():
    return ToolRuntime(
        state={"thread_data": None},
        context=None,
        config=RunnableConfig(),
        stream_writer=None,
        tool_call_id=None,
        store=None,
    )


def test_handoff_includes_per_subject(mock_runtime, tmp_path: Path):
    workspace = tmp_path / "ws"
    output = tmp_path / "out"
    workspace.mkdir()
    output.mkdir()

    # Fabricate metrics.pkl with a per_subject block mirroring ethoinsight shape
    m_result = {
        "paradigm": "shoaling",
        "per_subject": {
            "Subject 1": {"distance_moved": 24942.3, "mean_nnd": 36.09},
            "Subject 2": {"distance_moved": 23715.4, "mean_nnd": 39.86},
            "Subject 3": {"distance_moved": 12518.2, "mean_nnd": 70.02},
        },
        "group_summary": {
            "control": {
                "distance_moved": {"mean": 24328.9, "std": 867.5, "n": 2, "values": [24942.3, 23715.4]},
                "mean_nnd": {"mean": 37.97, "std": 2.67, "n": 2, "values": [36.09, 39.86]},
            },
            "treatment": {
                "distance_moved": {"mean": 12518.2, "std": 0.0, "n": 1, "values": [12518.2]},
                "mean_nnd": {"mean": 70.02, "std": 0.0, "n": 1, "values": [70.02]},
            },
        },
        "group_level_metrics": {},
        "data_quality_warnings": [],
        "metadata": {"paradigm": "shoaling"},
    }
    with (workspace / "metrics.pkl").open("wb") as f:
        pickle.dump(m_result, f)

    # Minimal stats so the code path writes the handoff
    (workspace / "statistics.json").write_text(
        json.dumps({"comparisons": [], "summary": {}, "alpha": 0.05, "correction": "bonferroni"})
    )
    (workspace / "charts.json").write_text(json.dumps({"chart_paths": []}))
    (workspace / "parsed_summary.json").write_text(json.dumps({"n_files": 3}))

    handoff_path = tmp_path / "handoff.json"
    result = assess_and_handoff_tool.invoke({
        "runtime": mock_runtime,
        "paradigm": "shoaling",
        "groups": json.dumps({"control": ["Subject 1", "Subject 2"], "treatment": ["Subject 3"]}),
        "workspace_dir": str(workspace),
        "output_dir": str(output),
        "handoff_path": str(handoff_path),
    })
    payload = json.loads(result)
    assert payload["status"] == "completed", payload

    handoff = json.loads(handoff_path.read_text())
    assert "per_subject" in handoff, "handoff JSON missing per_subject field"
    assert "Subject 3" in handoff["per_subject"]
    subject_3 = handoff["per_subject"]["Subject 3"]
    # Raw value must survive round-trip (used by data-analyst for outlier detection)
    assert subject_3["mean_nnd"] == pytest.approx(70.02, rel=1e-3)
