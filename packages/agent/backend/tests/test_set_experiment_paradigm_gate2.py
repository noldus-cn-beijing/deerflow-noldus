"""Tests for set_experiment_paradigm Gate 2 quality acknowledgement."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from deerflow.agents.middlewares.experiment_context import set_experiment_paradigm_tool

# LangChain @tool decorator wraps functions as StructuredTool objects.
# Access the underlying function via .func for direct testing.
_sep = set_experiment_paradigm_tool.func


def _make_runtime(workspace_path):
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": workspace_path}}
    return runtime


def test_gate1_normal_creation(tmp_path):
    """Gate 1 mode: creates context with gate_completed=["gate1_paradigm"]."""
    result = json.loads(
        _sep(
            paradigm="forced_swim",
            paradigm_cn="强迫游泳实验",
            category="anxiety",
            subject="rodent",
            ev19_template="PorsoltCylinder-NoZones",
            workspace_dir=str(tmp_path),
            runtime=_make_runtime(str(tmp_path)),
        )
    )
    assert result["status"] == "ok"
    assert result["paradigm"] == "forced_swim"

    ctx_path = tmp_path / "experiment-context.json"
    ctx = json.loads(ctx_path.read_text())
    assert ctx["gate_completed"] == ["gate1_paradigm"]


def test_gate2_acknowledge_appends_to_existing(tmp_path):
    """Gate 2 mode: appends gate2_quality_acknowledged to existing context."""
    # Setup: create Gate 1 context first
    _sep(
        paradigm="forced_swim",
        paradigm_cn="强迫游泳实验",
        category="anxiety",
        subject="rodent",
        ev19_template="PorsoltCylinder-NoZones",
        workspace_dir=str(tmp_path),
        runtime=_make_runtime(str(tmp_path)),
    )

    # Act: acknowledge quality
    result = json.loads(
        _sep(
            acknowledge_quality=True,
            workspace_dir=str(tmp_path),
            runtime=_make_runtime(str(tmp_path)),
        )
    )
    assert result["status"] == "ok"
    assert "gate2_quality_acknowledged" in result["gate_completed"]

    ctx_path = tmp_path / "experiment-context.json"
    ctx = json.loads(ctx_path.read_text())
    assert "gate1_paradigm" in ctx["gate_completed"]
    assert "gate2_quality_acknowledged" in ctx["gate_completed"]
    assert ctx["paradigm"] == "forced_swim"  # preserved


def test_gate2_acknowledge_without_gate1_returns_error(tmp_path):
    """Gate 2 acknowledge before Gate 1 should return error."""
    result = json.loads(
        _sep(
            acknowledge_quality=True,
            workspace_dir=str(tmp_path),
            runtime=_make_runtime(str(tmp_path)),
        )
    )
    assert result["status"] == "error"
    assert "Cannot acknowledge quality before Gate 1" in result["message"]


def test_gate2_duplicate_acknowledge_does_not_duplicate(tmp_path):
    """Repeated Gate 2 acknowledge should not duplicate the entry."""
    # Setup
    _sep(
        paradigm="forced_swim",
        paradigm_cn="强迫游泳实验",
        category="anxiety",
        subject="rodent",
        ev19_template="PorsoltCylinder-NoZones",
        workspace_dir=str(tmp_path),
        runtime=_make_runtime(str(tmp_path)),
    )

    # Acknowledge twice
    _sep(acknowledge_quality=True, workspace_dir=str(tmp_path), runtime=_make_runtime(str(tmp_path)))
    _sep(acknowledge_quality=True, workspace_dir=str(tmp_path), runtime=_make_runtime(str(tmp_path)))

    ctx_path = tmp_path / "experiment-context.json"
    ctx = json.loads(ctx_path.read_text())
    # gate2_quality_acknowledged should appear exactly once
    assert ctx["gate_completed"].count("gate2_quality_acknowledged") == 1


def test_gate1_preserves_existing_gate2_acknowledge(tmp_path):
    """Gate 1 re-call should preserve existing gate2_quality_acknowledged."""
    # Setup: Gate 1 + Gate 2
    _sep(
        paradigm="forced_swim",
        paradigm_cn="强迫游泳实验",
        category="anxiety",
        subject="rodent",
        ev19_template="PorsoltCylinder-NoZones",
        workspace_dir=str(tmp_path),
        runtime=_make_runtime(str(tmp_path)),
    )
    _sep(acknowledge_quality=True, workspace_dir=str(tmp_path), runtime=_make_runtime(str(tmp_path)))

    # Act: change paradigm (user changes mind)
    _sep(
        paradigm="epm",
        paradigm_cn="高架十字迷宫",
        category="anxiety",
        subject="rodent",
        ev19_template="PlusMaze-AllZones",
        workspace_dir=str(tmp_path),
        runtime=_make_runtime(str(tmp_path)),
    )

    ctx_path = tmp_path / "experiment-context.json"
    ctx = json.loads(ctx_path.read_text())
    assert ctx["paradigm"] == "epm"
    assert "gate1_paradigm" in ctx["gate_completed"]
    assert "gate2_quality_acknowledged" in ctx["gate_completed"]


def test_gate1_missing_required_fields_returns_error(tmp_path):
    """Gate 1 with missing required fields should return error."""
    result = json.loads(
        _sep(
            paradigm="forced_swim",
            paradigm_cn=None,
            category=None,
            subject=None,
            ev19_template=None,
            workspace_dir=str(tmp_path),
            runtime=_make_runtime(str(tmp_path)),
        )
    )
    assert result["status"] == "error"
    assert "Missing required fields" in result["message"]
