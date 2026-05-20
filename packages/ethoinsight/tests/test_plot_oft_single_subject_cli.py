"""2026-05-20: OFT 单样本场景的两张基础图脚本 CLI (handoff #2 同步补全)."""
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
def oft_trajectory_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 60
    in_center = np.zeros(n, dtype=int)
    in_center[10:25] = 1
    in_center[40:50] = 1
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "distance_moved": rng.uniform(0, 5, n),
            "in_zone_center__center_point_": in_center,
        }
    )
    path = tmp_path / "subject1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


def test_plot_center_time_ratio_bar_cli_writes_png(oft_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "center_time_ratio.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts.oft.plot_center_time_ratio_bar",
            "--input", str(oft_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_plot_center_entry_summary_cli_writes_png(oft_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "center_entry_summary.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts.oft.plot_center_entry_summary",
            "--input", str(oft_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout
