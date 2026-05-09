"""Soft gate tests — analysis entrypoints fail-fast when ev19_template is missing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_context(workspace: Path, *, with_ev19: bool):
    ctx = {"paradigm": "shoaling", "category": "zebrafish", "subject": "fish"}
    if with_ev19:
        ctx["ev19_template"] = "OpenFieldCircle-NoZones-Fish"
    (workspace / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")


def test_require_ev19_template_returns_none_when_set(tmp_path):
    """Returns None when ev19_template is present."""
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=True)
    assert require_ev19_template(str(tmp_path)) is None


def test_require_ev19_template_returns_error_when_missing(tmp_path):
    """Returns error dict when ev19_template is missing."""
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=False)
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "ev19_template" in result["reason"]
    assert "remediation" in result


def test_require_ev19_template_returns_error_when_no_file(tmp_path):
    """Returns error dict when experiment-context.json doesn't exist."""
    from ethoinsight.templates._gate import require_ev19_template

    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "remediation" in result


def test_require_ev19_template_returns_error_when_json_invalid(tmp_path):
    """Returns error dict when experiment-context.json is malformed."""
    from ethoinsight.templates._gate import require_ev19_template

    (tmp_path / "experiment-context.json").write_text("{not valid json", encoding="utf-8")
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
