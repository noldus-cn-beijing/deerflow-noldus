"""Soft gate tests — analysis entrypoints fail-fast when ev19_template is missing."""

import json
from pathlib import Path

import pytest


def _write_context(workspace: Path, *, with_ev19: bool, paradigm: str = "epm"):
    ctx = {"paradigm": paradigm, "category": "anxiety", "subject": "rodent"}
    if with_ev19:
        ctx["ev19_template"] = "PlusMaze-AllZones"
    (workspace / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")


def test_gate_passes_when_ev19_template_set(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=True)
    result = require_ev19_template(str(tmp_path))
    assert result is None


def test_gate_returns_error_when_ev19_template_missing(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=False)
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "ev19_template" in result["reason"]


def test_gate_returns_error_when_context_file_missing(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"
    assert "experiment-context.json" in result["reason"]


def test_gate_returns_error_on_malformed_context(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    (tmp_path / "experiment-context.json").write_text("not valid json {{{", encoding="utf-8")
    result = require_ev19_template(str(tmp_path))
    assert result is not None
    assert result["status"] == "error"


def test_gate_handles_trailing_slash(tmp_path):
    from ethoinsight.templates._gate import require_ev19_template

    _write_context(tmp_path, with_ev19=True)
    result = require_ev19_template(str(tmp_path) + "/")
    assert result is None
