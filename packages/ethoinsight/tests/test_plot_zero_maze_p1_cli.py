"""Zero Maze box_open_zone (orphan) + bar_open_zone + trajectory + heatmap CLI tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    columns = list(df.columns)
    n_header_lines = 6
    lines: list[str] = []
    lines.append(f'"标题行数";"{n_header_lines}"')
    lines.append(f'"对象名称";"{subject}"')
    lines.append('"试验名称";"Trial 1"')
    lines.append('"竞技场名称";"Arena 1"')
    lines.append(";".join(f'"{c}"' for c in columns))
    lines.append(";".join(['""'] * len(columns)))
    for _, row in df.iterrows():
        values = []
        for v in row.values:
            if pd.isna(v):
                values.append('"-"')
            else:
                values.append(f'"{v}"')
        lines.append(";".join(values))
    content = "\n".join(lines) + "\n"
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")
        f.write(content.encode("utf-16-le"))


def _make_zero_maze_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    in_open = np.zeros(n, dtype=int)
    in_open[10:25] = 1
    return pd.DataFrame({
        "trial_time": np.arange(n, dtype=float) * 0.04,
        "x_center": rng.uniform(100, 500, n),
        "y_center": rng.uniform(100, 500, n),
        "in_zone_open_1__center_point_": in_open,
        "distance_moved": rng.uniform(0, 2, n),
    })


@pytest.fixture
def single_file(tmp_path: Path) -> Path:
    df = _make_zero_maze_df()
    p = tmp_path / "s1.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    return p


@pytest.fixture
def three_files(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(1, 4):
        df = _make_zero_maze_df()
        p = tmp_path / f"s{i}.txt"
        _df_to_ethovision_file(df, p, subject=f"Subject {i}")
        paths.append(p)
    return paths


def test_box_open_zone_n3(three_files: list[Path], tmp_path: Path):
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(p) for p in three_files]))
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(json.dumps({"ctrl": ["Subject 1", "Subject 2"], "tx": ["Subject 3"]}))
    output = tmp_path / "box.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.zero_maze.plot_box_open_zone",
         "--inputs", str(inputs_json), "--groups", str(groups_json), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_bar_open_zone_single(single_file: Path, tmp_path: Path):
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(single_file)]))
    output = tmp_path / "bar.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.zero_maze.plot_bar_open_zone",
         "--inputs", str(inputs_json), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_bar_open_zone_n3(three_files: list[Path], tmp_path: Path):
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(p) for p in three_files]))
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(json.dumps({"ctrl": ["Subject 1", "Subject 2"], "tx": ["Subject 3"]}))
    output = tmp_path / "bar.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.zero_maze.plot_bar_open_zone",
         "--inputs", str(inputs_json), "--groups", str(groups_json), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


def test_trajectory_zero_maze(single_file: Path, tmp_path: Path):
    output = tmp_path / "traj.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts._common.plot_trajectory",
         "--input", str(single_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


def test_heatmap_zero_maze(single_file: Path, tmp_path: Path):
    output = tmp_path / "heatmap.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts._common.plot_heatmap",
         "--input", str(single_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout
