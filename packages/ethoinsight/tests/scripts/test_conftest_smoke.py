"""Smoke test for conftest fixtures: verifies synthetic EthoVision files parse correctly."""

from __future__ import annotations

from pathlib import Path

from ethoinsight.parse import parse_trajectory


def test_epm_trajectory_file_parses(epm_trajectory_file: Path):
    df = parse_trajectory(str(epm_trajectory_file))
    assert "in_zone_open_arm_1" in df.columns
    assert "trial_time" in df.columns
    assert len(df) > 0


def test_epm_trajectory_files_have_expected_count(epm_trajectory_files: list[Path]):
    assert len(epm_trajectory_files) == 6
    for p in epm_trajectory_files:
        df = parse_trajectory(str(p))
        assert len(df) > 0
