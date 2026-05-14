"""Subprocess-level tests for ethoinsight.scripts.shoaling.*."""

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


def _build_inputs_json(paths: list[Path], target: Path) -> Path:
    """Write an inputs.json file listing trajectory file paths."""
    data = [str(p) for p in paths]
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


class TestComputeInterIndividualDistance:
    def test_happy_path_writes_json_and_emits_result(
        self, shoaling_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.shoaling.compute_inter_individual_distance",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "inter_individual_distance"
        assert isinstance(payload["value"], dict)
        assert "mean_iid_mean" in payload["value"]
        assert "mean_iid_std" in payload["value"]
        assert isinstance(payload["value"]["mean_iid_mean"], float)
        assert payload["value"]["mean_iid_mean"] > 0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] ") :])
        assert result_payload["metric"] == "inter_individual_distance"

    def test_missing_inputs_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.shoaling.compute_inter_individual_distance",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "inputs" in result.stderr.lower()

    def test_insufficient_subjects_returns_null(self, tmp_path: Path):
        """Single fish cannot compute IID — returns None."""
        from tests.scripts.conftest import _make_shoaling_df, _df_to_ethovision_file

        df = _make_shoaling_df(n_frames=50)
        path = tmp_path / "single_fish.txt"
        _df_to_ethovision_file(df, path, subject="Fish 1")

        inputs_file = _build_inputs_json([path], tmp_path / "inputs.json")
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.shoaling.compute_inter_individual_distance",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "inter_individual_distance"
        assert payload["value"] is None


class TestComputeNearestNeighborDistance:
    def test_happy_path(self, shoaling_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.shoaling.compute_nearest_neighbor_distance",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "nearest_neighbor_distance"
        assert isinstance(payload["value"], dict)
        assert isinstance(payload["value"]["mean_nnd"], float)
        assert payload["value"]["mean_nnd"] > 0


class TestComputeGroupPolarity:
    def test_happy_path(self, shoaling_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.shoaling.compute_group_polarity",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "group_polarity"
        assert isinstance(payload["value"], dict)
        assert isinstance(payload["value"]["mean_polarity"], float)
        assert 0.0 <= payload["value"]["mean_polarity"] <= 1.0


class TestPlotBoxIid:
    def test_plot_without_groups(
        self, shoaling_trajectory_files: list[Path], tmp_path: Path
    ):
        """Plot without groups uses all subjects in a default group."""
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )
        out_path = tmp_path / "box.png"
        result = _run_script(
            "ethoinsight.scripts.shoaling.plot_box_iid",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000

    def test_plot_with_groups(
        self, shoaling_trajectory_files: list[Path], tmp_path: Path
    ):
        """Plot with two fake groups."""
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )

        # Split 5 fish into two groups
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Fish 1", "Fish 2", "Fish 3"],
                    "treatment": ["Fish 4", "Fish 5"],
                }
            )
        )

        out_path = tmp_path / "box.png"
        result = _run_script(
            "ethoinsight.scripts.shoaling.plot_box_iid",
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

    def test_requires_inputs(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.shoaling.plot_box_iid",
            ["--input", str(tmp_path / "f.txt"), "--output", str(tmp_path / "x.png")],
        )
        # make_plot_parser requires either --input or --inputs; --input is valid
        # but reading the file will fail (doesn't exist)
        assert result.returncode != 0


class TestRunGroupwiseStats:
    def test_stats_with_two_groups(
        self, shoaling_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Fish 1", "Fish 2", "Fish 3"],
                    "treatment": ["Fish 4", "Fish 5"],
                }
            )
        )

        out_path = tmp_path / "stats.json"
        result = _run_script(
            "ethoinsight.scripts.shoaling.run_groupwise_stats",
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
        assert payload["paradigm"] == "shoaling"
        assert "comparisons" in payload
        assert "alpha" in payload

    def test_requires_groups(
        self, shoaling_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = _build_inputs_json(
            shoaling_trajectory_files, tmp_path / "inputs.json"
        )
        result = _run_script(
            "ethoinsight.scripts.shoaling.run_groupwise_stats",
            ["--inputs", str(inputs_file), "--output", str(tmp_path / "stats.json")],
        )
        assert result.returncode != 0
        assert "groups" in result.stderr.lower() or "required" in result.stderr.lower()
