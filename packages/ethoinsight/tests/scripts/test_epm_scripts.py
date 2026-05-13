"""Subprocess-level tests for ethoinsight.scripts.epm.*."""

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


class TestComputeOpenArmTimeRatio:
    def test_happy_path_writes_json_and_emits_result(
        self, epm_trajectory_file: Path, tmp_path: Path
    ):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert isinstance(payload["value"], float)
        assert 0.0 <= payload["value"] <= 1.0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] ") :])
        assert result_payload["metric"] == "open_arm_time_ratio"

    def test_missing_input_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_file_without_open_arm_columns_returns_value_none(
        self, tmp_path: Path, make_epm_df
    ):
        from tests.scripts.conftest import _df_to_ethovision_file

        # Build a df without any open-arm zone columns
        df = make_epm_df(open_arm_cols=["in_zone_closed_arm_1"])  # only closed arm
        df = df.drop(columns=["in_zone_closed_arm_1"])  # remove all zone cols
        df["x_center"] = df["x_center"]  # keep position cols
        path = tmp_path / "no_open_arm.txt"
        _df_to_ethovision_file(df, path)

        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(path), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert payload["value"] is None


class TestComputeOpenArmEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestComputeOpenArmEntryRatio:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_ratio"
        # ratio is None or float in [0, 1]
        assert payload["value"] is None or (
            isinstance(payload["value"], float) and 0.0 <= payload["value"] <= 1.0
        )


class TestComputeOpenArmTime:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeTotalEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_total_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "total_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestPlotBoxOpenArm:
    def test_plot_with_groups(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

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
            "ethoinsight.scripts.epm.plot_box_open_arm",
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
        self, epm_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

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
            "ethoinsight.scripts.epm.run_groupwise_stats",
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
