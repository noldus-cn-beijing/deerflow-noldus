"""OFT trajectory + heatmap CLI tests (P2)."""
from __future__ import annotations

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


@pytest.fixture
def oft_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "trial_time": np.arange(n, dtype=float) * 0.04,
        "x_center": rng.uniform(100, 500, n),
        "y_center": rng.uniform(100, 500, n),
        "in_zone_center__center_point_": np.where(np.arange(n) < 10, 1, 0),
    })
    p = tmp_path / "oft.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    return p


def test_trajectory_oft(oft_file: Path, tmp_path: Path):
    output = tmp_path / "traj.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts._common.plot_trajectory",
         "--input", str(oft_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


def test_heatmap_oft(oft_file: Path, tmp_path: Path):
    output = tmp_path / "heatmap.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts._common.plot_heatmap",
         "--input", str(oft_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout
