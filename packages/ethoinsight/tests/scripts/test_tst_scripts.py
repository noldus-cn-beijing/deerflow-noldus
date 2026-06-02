"""Subprocess-level tests for ethoinsight.scripts.tst.*."""

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


class TestComputeImmobilityTime:
    def test_happy_path_writes_json_and_emits_result(
        self, tst_trajectory_file: Path, tmp_path: Path
    ):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.tst.compute_immobility_time",
            ["--input", str(tst_trajectory_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "immobility_time"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] ") :])
        assert result_payload["metric"] == "immobility_time"

    def test_missing_input_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.tst.compute_immobility_time",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_file_without_activity_column_uses_velocity_fallback(
        self, tmp_path: Path, make_tst_df
    ):
        from tests.scripts.conftest import _df_to_ethovision_file

        # Build a df without Activity_State — velocity fallback should work
        df = make_tst_df()
        df = df.drop(columns=["Activity_State"])
        path = tmp_path / "no_activity.txt"
        _df_to_ethovision_file(df, path)

        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.tst.compute_immobility_time",
            ["--input", str(path), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "immobility_time"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeImmobilityLatency:
    def test_happy_path(self, tst_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.tst.compute_immobility_latency",
            ["--input", str(tst_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "immobility_latency"
        # With alternating 20/20 pattern, the first frame is immobile (0),
        # so latency = first trial_time value = 0.0
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeImmobilityBoutCount:
    def test_happy_path(self, tst_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.tst.compute_immobility_bout_count",
            ["--input", str(tst_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "immobility_bout_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestPlotBoxImmobility:
    def test_plot_with_groups(self, tst_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in tst_trajectory_files]))

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
            "ethoinsight.scripts.tst.plot_box_immobility",
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
        self, tst_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in tst_trajectory_files]))

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
            "ethoinsight.scripts.tst.run_groupwise_stats",
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
