"""W6: plot_timeseries CLI 包装。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    """Write df as a minimal EthoVision-style trajectory file (UTF-16-LE BOM, semicolon-delimited)."""
    columns = list(df.columns)
    n_header_lines = 6  # 1 title-count + 3 metadata + 1 column-names + 1 units

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
        f.write(b"\xff\xfe")  # UTF-16-LE BOM
        f.write(content.encode("utf-16-le"))


@pytest.fixture
def sample_trajectory_file(tmp_path: Path) -> Path:
    """Write a minimal EthoVision trajectory file with trial_time + x_center + y_center."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "distance_moved": rng.uniform(0, 5, n),
        }
    )
    path = tmp_path / "subject1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


def test_plot_timeseries_cli_runs_with_single_input(sample_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(sample_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stderr={result.stderr}")
    assert output.exists()
    assert "[result]" in result.stdout


def test_plot_timeseries_cli_accepts_y_col_override(sample_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(sample_trajectory_file),
            "--output", str(output),
            "--y-col", "x_center",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI with --y-col failed: stderr={result.stderr}")
    assert output.exists()
