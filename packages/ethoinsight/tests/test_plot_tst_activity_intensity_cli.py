"""TST activity_intensity CLI test."""
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
def tst_velocity_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "trial_time": np.arange(n, dtype=float) * 0.04,
        "velocity": rng.uniform(0, 5, n),
        "mobility_state_immobile": np.zeros(n, dtype=int),
    })
    p = tmp_path / "tst_velocity.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    return p


def test_activity_intensity_cli_writes_png(tst_velocity_file: Path, tmp_path: Path):
    output = tmp_path / "activity_intensity.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.tst.plot_activity_intensity",
         "--input", str(tst_velocity_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout
