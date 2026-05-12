"""Tests for ethoinsight.scripts._common.*."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, check=False,
    )


class TestComputeDistanceMoved:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts._common.compute_distance_moved",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "distance_moved"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeVelocityStats:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts._common.compute_velocity_stats",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "velocity_stats"
        # value is a dict with mean/std/max/min/median
        v = payload["value"]
        assert v is None or set(v.keys()) >= {"mean", "std", "max", "min", "median"}


class TestPlotTrajectory:
    def test_single_input_produces_png(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "trajectory.png"
        result = _run_script(
            "ethoinsight.scripts._common.plot_trajectory",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # PNG 不应为空

    def test_multiple_inputs_produces_png(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))
        out_path = tmp_path / "trajectory_all.png"
        result = _run_script(
            "ethoinsight.scripts._common.plot_trajectory",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
