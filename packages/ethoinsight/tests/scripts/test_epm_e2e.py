"""End-to-end test: simulate code-executor orchestrating multiple EPM scripts.

This mirrors what the subagent will actually do in production:
1. Receive a list of trajectory files + group assignment from lead
2. Run each required compute_*.py script per subject
3. Run plot_box_open_arm with groups
4. Run run_groupwise_stats with groups
5. Aggregate everything into handoff_code_executor.json

The test does NOT exercise the agent itself — it exercises the script
interface contract that the agent's prompt will rely on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, check=False,
    )


def test_epm_full_orchestration(epm_trajectory_files: list[Path], tmp_path: Path) -> None:
    """Happy path: 6 subjects * 5 metrics + box plot + groupwise stats → handoff JSON."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outputs = workspace / "outputs"
    outputs.mkdir()

    # Subagent would write these JSON files before calling scripts
    inputs_file = workspace / "inputs.json"
    inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

    groups_file = workspace / "groups.json"
    groups_file.write_text(json.dumps({
        "control": ["Subject 1", "Subject 2", "Subject 3"],
        "treatment": ["Subject 4", "Subject 5", "Subject 6"],
    }))

    # ----- Step 1: per-subject compute scripts (simulate subagent loop) -----
    per_subject: dict[str, dict[str, object]] = {}
    for traj_file in epm_trajectory_files:
        subject_results: dict[str, object] = {}
        for metric_module in [
            "compute_open_arm_time_ratio",
            "compute_open_arm_entry_count",
            "compute_open_arm_entry_ratio",
            "compute_open_arm_time",
            "compute_total_entry_count",
        ]:
            out_path = outputs / f"{traj_file.stem}__{metric_module}.json"
            result = _run(
                f"ethoinsight.scripts.epm.{metric_module}",
                ["--input", str(traj_file), "--output", str(out_path)],
            )
            assert result.returncode == 0, f"{metric_module} failed: {result.stderr}"
            payload = json.loads(out_path.read_text())
            subject_results[payload["metric"]] = payload["value"]
        per_subject[traj_file.stem] = subject_results

    # ----- Step 2: group-level box plot -----
    box_path = outputs / "epm_box.png"
    result = _run(
        "ethoinsight.scripts.epm.plot_box_open_arm",
        ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(box_path)],
    )
    assert result.returncode == 0, f"box plot failed: {result.stderr}"
    assert box_path.exists()

    # ----- Step 3: groupwise stats -----
    stats_path = outputs / "epm_stats.json"
    result = _run(
        "ethoinsight.scripts.epm.run_groupwise_stats",
        ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(stats_path)],
    )
    assert result.returncode == 0, f"stats failed: {result.stderr}"
    stats = json.loads(stats_path.read_text())

    # ----- Step 4: aggregate into handoff JSON (subagent does this) -----
    handoff = {
        "paradigm": "epm",
        "per_subject": per_subject,
        "charts": [str(box_path)],
        "statistics": stats,
    }
    handoff_path = workspace / "handoff_code_executor.json"
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2))

    # ----- Assertions: handoff is well-formed -----
    assert handoff["paradigm"] == "epm"
    assert len(handoff["per_subject"]) == 6
    # Every subject has all 5 metrics
    for subject, metrics in handoff["per_subject"].items():
        assert set(metrics.keys()) == {
            "open_arm_time_ratio",
            "open_arm_entry_count",
            "open_arm_entry_ratio",
            "open_arm_time",
            "total_entry_count",
        }
    assert "comparisons" in handoff["statistics"]


def test_epm_single_subject_descriptive(epm_trajectory_file: Path, tmp_path: Path) -> None:
    """n=1 single-subject scenario: skip stats / group plots, only run compute_* + plot_trajectory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outputs = workspace / "outputs"
    outputs.mkdir()

    # Compute all 5 EPM metrics for the single subject
    metrics_result: dict[str, object] = {}
    for metric_module in [
        "compute_open_arm_time_ratio",
        "compute_open_arm_entry_count",
        "compute_open_arm_entry_ratio",
        "compute_open_arm_time",
        "compute_total_entry_count",
    ]:
        out_path = outputs / f"{metric_module}.json"
        result = _run(
            f"ethoinsight.scripts.epm.{metric_module}",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"{metric_module}: {result.stderr}"
        payload = json.loads(out_path.read_text())
        metrics_result[payload["metric"]] = payload["value"]

    # Trajectory plot
    traj_path = outputs / "trajectory.png"
    result = _run(
        "ethoinsight.scripts._common.plot_trajectory",
        ["--input", str(epm_trajectory_file), "--output", str(traj_path)],
    )
    assert result.returncode == 0, f"trajectory: {result.stderr}"
    assert traj_path.exists()

    # No stats, no group plots → handoff has no `statistics` / no group charts
    handoff = {
        "paradigm": "epm",
        "per_subject": {"Subject 1": metrics_result},
        "charts": [str(traj_path)],
    }
    assert len(handoff["per_subject"]) == 1
    assert "statistics" not in handoff
