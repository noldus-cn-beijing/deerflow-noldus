"""Tests for ethoinsight.parse.dump_headers CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "ethoinsight.parse.dump_headers", *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_dump_headers_writes_expected_keys(tmp_path):
    """需要一份真 EthoVision 文件 fixture；如不存在，标 skip。"""
    fixtures = _fixtures_dir()
    candidates = list(fixtures.glob("*.txt")) if fixtures.exists() else []
    if not candidates:
        pytest.skip("No EthoVision fixture file under tests/fixtures/")
    raw = candidates[0]

    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli(["--input", str(raw), "--output", str(out)])
    assert rc == 0, f"stderr: {stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "file" in data
    assert "columns" in data
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) > 0


def test_dump_headers_file_not_found(tmp_path):
    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli(
        [
            "--input",
            "/nonexistent/path/raw.txt",
            "--output",
            str(out),
        ]
    )
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "file.not_found"


def test_dump_headers_format_unrecognized(tmp_path):
    """非 EthoVision 文件应返回 format.unrecognized."""
    bad = tmp_path / "not_ethovision.txt"
    bad.write_text("this is not an EthoVision file", encoding="utf-8")
    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli(["--input", str(bad), "--output", str(out)])
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "format.unrecognized"
