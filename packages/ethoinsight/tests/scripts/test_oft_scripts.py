"""Subprocess-level tests for ethoinsight.scripts.oft.*."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run `python -m <module> <args>` and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestComputeCenterTimeRatio:
    def test_happy_path_writes_json_and_emits_result(
        self, oft_trajectory_file: Path, tmp_path: Path
    ):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.oft.compute_center_time_ratio",
            ["--input", str(oft_trajectory_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "center_time_ratio"
        assert isinstance(payload["value"], float)
        assert 0.0 <= payload["value"] <= 1.0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] ") :])
        assert result_payload["metric"] == "center_time_ratio"

    def test_missing_input_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.oft.compute_center_time_ratio",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_file_without_center_columns_returns_value_none(
        self, tmp_path: Path, make_oft_df
    ):
        from tests.scripts.conftest import _df_to_ethovision_file

        # Build a df with coordinates but no center zone columns
        df = make_oft_df()
        df = df.drop(columns=[c for c in df.columns if "zone" in c.lower()])
        path = tmp_path / "no_center.txt"
        _df_to_ethovision_file(df, path)

        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.oft.compute_center_time_ratio",
            ["--input", str(path), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "center_time_ratio"
        assert payload["value"] is None


class TestComputeThigmotaxisIndex:
    def test_happy_path(self, oft_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.oft.compute_thigmotaxis_index",
            ["--input", str(oft_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "thigmotaxis_index"
        assert isinstance(payload["value"], float)
        assert 0.0 <= payload["value"] <= 1.0


class TestComputeCenterDistanceRatio:
    def test_happy_path(self, oft_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.oft.compute_center_distance_ratio",
            ["--input", str(oft_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "center_distance_ratio"
        # ratio is None or float in [0, 1]
        assert payload["value"] is None or (
            isinstance(payload["value"], float) and 0.0 <= payload["value"] <= 1.0
        )


class TestComputeCenterEntryCount:
    def test_happy_path(self, oft_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.oft.compute_center_entry_count",
            ["--input", str(oft_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "center_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestPlotBoxCenter:
    def test_plot_with_groups(self, oft_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in oft_trajectory_files]))

        # First 3 = control, last 3 = treatment
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Subject 1", "Subject 2", "Subject 3"],
                    "treatment": ["Subject 4", "Subject 5", "Subject 6"],
                }
            )
        )

        out_path = tmp_path / "box.png"
        result = _run_script(
            "ethoinsight.scripts.oft.plot_box_center",
            [
                "--inputs",
                str(inputs_file),
                "--groups",
                str(groups_file),
                "--output",
                str(out_path),
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000


class TestRunGroupwiseStats:
    def test_stats_with_two_groups(
        self, oft_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in oft_trajectory_files]))

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Subject 1", "Subject 2", "Subject 3"],
                    "treatment": ["Subject 4", "Subject 5", "Subject 6"],
                }
            )
        )

        out_path = tmp_path / "stats.json"
        result = _run_script(
            "ethoinsight.scripts.oft.run_groupwise_stats",
            [
                "--inputs",
                str(inputs_file),
                "--groups",
                str(groups_file),
                "--output",
                str(out_path),
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert "comparisons" in payload
        assert "alpha" in payload
