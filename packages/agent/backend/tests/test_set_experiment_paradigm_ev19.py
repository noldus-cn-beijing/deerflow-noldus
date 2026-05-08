"""Tests for set_experiment_paradigm tool with ev19_template parameter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime

from deerflow.agents.middlewares.experiment_context import set_experiment_paradigm_tool


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


def test_valid_ev19_template_writes_context_with_field(tmp_path):
    """Valid ev19_template is written to experiment-context.json."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PlusMaze-AllZones",
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["ev19_template"] == "PlusMaze-AllZones"

    written = json.loads((tmp_path / "experiment-context.json").read_text(encoding="utf-8"))
    assert written["paradigm"] == "epm"
    assert written["ev19_template"] == "PlusMaze-AllZones"


def test_invalid_ev19_template_returns_error_with_candidates(tmp_path):
    """Unknown ev19_template returns error + suggested close matches."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PlusMze-AllZones",  # typo
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert "candidates" in parsed
    assert any("PlusMaze" in c for c in parsed["candidates"])

    # Context file MUST NOT be written on error
    assert not (tmp_path / "experiment-context.json").exists()


def test_paradigm_template_mismatch_writes_warning_but_proceeds(tmp_path):
    """If ev19_template is valid but not in the recommended list for paradigm, a warning is included but the write succeeds."""
    runtime = _runtime_with_workspace(tmp_path)

    result = set_experiment_paradigm_tool.invoke({
        "paradigm": "epm",
        "paradigm_cn": "高架十字迷宫",
        "category": "anxiety",
        "subject": "rodent",
        "ev19_template": "PorsoltCylinder-AllZones",  # 抑郁范式模板用在 EPM 上
        "runtime": runtime,
    })

    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "warning" in parsed

    # Context file IS written
    written = json.loads((tmp_path / "experiment-context.json").read_text(encoding="utf-8"))
    assert written["ev19_template"] == "PorsoltCylinder-AllZones"
