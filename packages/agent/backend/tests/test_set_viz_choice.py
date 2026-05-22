"""Tests for set_viz_choice tool — gate3 viz acknowledgement."""

import json
from pathlib import Path

from langchain.tools import ToolRuntime

from deerflow.agents.middlewares.experiment_context import set_viz_choice_tool


def _runtime_with_workspace(workspace: Path) -> ToolRuntime:
    return ToolRuntime(
        state={"thread_data": {"workspace_path": str(workspace)}},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def test_set_viz_choice_no(tmp_path):
    ctx_path = tmp_path / "experiment-context.json"
    ctx_path.write_text(json.dumps({"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}))

    result = set_viz_choice_tool.invoke({"choice": "no", "workspace_dir": str(tmp_path)})
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["viz_choice"] == "no"

    saved = json.loads(ctx_path.read_text())
    assert "gate3_viz_acknowledged" in saved["gate_completed"]
    assert saved["viz_choice"] == "no"
    assert "viz_acknowledged_at" in saved


def test_set_viz_choice_yes_through_runtime(tmp_path):
    """Thread_data workspace_path should take precedence over default workspace_dir."""
    ctx_path = tmp_path / "experiment-context.json"
    ctx_path.write_text(json.dumps({"paradigm": "open_field", "gate_completed": ["gate1_paradigm"]}))

    runtime = _runtime_with_workspace(tmp_path)
    result = set_viz_choice_tool.invoke({
        "choice": "yes",
        "workspace_dir": "/wrong/path/",
        "runtime": runtime,
    })
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["viz_choice"] == "yes"

    saved = json.loads(ctx_path.read_text())
    assert "gate3_viz_acknowledged" in saved["gate_completed"]
    assert saved["viz_choice"] == "yes"


def test_set_viz_choice_missing_context(tmp_path):
    """If experiment-context.json is missing, return error."""
    result = set_viz_choice_tool.invoke({"choice": "yes", "workspace_dir": str(tmp_path)})
    data = json.loads(result)
    assert data["status"] == "error"
    assert "missing" in data["message"].lower()


def test_set_viz_choice_preserves_existing_fields(tmp_path):
    """Writing gate3 must not drop existing paradigm / gate_completed fields."""
    existing = {
        "paradigm": "shoaling",
        "paradigm_cn": "群聚行为",
        "category": "anxiety",
        "subject": "fish",
        "ev19_template": "Shoaling-AllZones",
        "gate_completed": ["gate1_paradigm", "gate2_quality_acknowledged"],
        "paradigm_confirmed_at": "2026-05-22T10:00:00+00:00",
    }
    ctx_path = tmp_path / "experiment-context.json"
    ctx_path.write_text(json.dumps(existing))

    result = set_viz_choice_tool.invoke({"choice": "yes", "workspace_dir": str(tmp_path)})
    data = json.loads(result)
    assert data["status"] == "ok"

    saved = json.loads(ctx_path.read_text())
    assert saved["paradigm"] == "shoaling"
    assert saved["paradigm_cn"] == "群聚行为"
    assert saved["ev19_template"] == "Shoaling-AllZones"
    assert "gate1_paradigm" in saved["gate_completed"]
    assert "gate2_quality_acknowledged" in saved["gate_completed"]
    assert "gate3_viz_acknowledged" in saved["gate_completed"]
    assert saved["viz_choice"] == "yes"


def test_set_viz_choice_idempotent(tmp_path):
    """Calling set_viz_choice twice should not duplicate gate3 in gate_completed."""
    ctx_path = tmp_path / "experiment-context.json"
    ctx_path.write_text(json.dumps({"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}))

    # First call
    set_viz_choice_tool.invoke({"choice": "yes", "workspace_dir": str(tmp_path)})
    # Second call (same choice)
    result = set_viz_choice_tool.invoke({"choice": "yes", "workspace_dir": str(tmp_path)})
    data = json.loads(result)
    assert data["status"] == "ok"

    saved = json.loads(ctx_path.read_text())
    assert saved["gate_completed"].count("gate3_viz_acknowledged") == 1


def test_set_viz_choice_corrupt_context_is_none(tmp_path):
    """Corrupt JSON in context file still treated as None — returns error."""
    ctx_path = tmp_path / "experiment-context.json"
    ctx_path.write_text("{not valid json")

    result = set_viz_choice_tool.invoke({"choice": "yes", "workspace_dir": str(tmp_path)})
    data = json.loads(result)
    assert data["status"] == "error"
    assert "missing" in data["message"].lower()
