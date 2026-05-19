"""W4: catalog CLI --mode metrics / charts。"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.fixture
def workspace(tmp_path):
    cols_file = tmp_path / "columns.json"
    cols_file.write_text(json.dumps({"columns": [
        "Trial time", "X center", "Y center",
        "in_zone_open_arms_center", "in_zone_closed_arms_center",
    ]}), encoding="utf-8")
    raws_file = tmp_path / "raws.json"
    raws_file.write_text(json.dumps([str(tmp_path / "raw1.txt")]), encoding="utf-8")
    return tmp_path, cols_file, raws_file


def test_mode_metrics_writes_plan_metrics_json(workspace):
    ws, cols, raws = workspace
    out = ws / "plan_metrics.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--mode", "metrics",
         "--paradigm", "epm",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "charts" not in payload


def test_mode_charts_writes_plan_charts_json(workspace):
    ws, cols, raws = workspace
    out = ws / "plan_charts.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--mode", "charts",
         "--paradigm", "epm",
         "--user-intent", "再画几个图",
         "--total-subjects", "1", "--n-per-group", "1", "--n-groups", "1",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "charts" in payload
    assert "charts_fallback_available" in payload
    assert payload["user_intent"] == "再画几个图"
    assert "metrics" not in payload


def test_default_mode_is_metrics_for_backward_compat(workspace):
    ws, cols, raws = workspace
    out = ws / "plan_legacy.json"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve",
         "--paradigm", "epm",
         "--columns-file", str(cols),
         "--raw-files-json", str(raws),
         "--workspace-dir", str(ws),
         "--output", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in payload
