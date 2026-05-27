"""OFT time_progress CLI tests: 600s trajectory → 2 bins; 350s trajectory → exit 1."""
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


def _make_oft_df(duration_sec: float, fps: float = 25.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Use ceil+1 to ensure actual duration >= duration_sec
    n = int(duration_sec * fps) + 2
    t = np.arange(n, dtype=float) / fps
    in_center = np.zeros(n, dtype=int)
    in_center[: n // 5] = 1
    return pd.DataFrame({
        "trial_time": t,
        "x_center": rng.uniform(100, 500, n),
        "y_center": rng.uniform(100, 500, n),
        "in_zone_center": in_center,
        "distance_moved": rng.uniform(0, 3, n),
    })


@pytest.fixture
def oft_600s_file(tmp_path: Path) -> Path:
    df = _make_oft_df(600)
    p = tmp_path / "oft_600s.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    return p


@pytest.fixture
def oft_350s_file(tmp_path: Path) -> Path:
    df = _make_oft_df(350)
    p = tmp_path / "oft_350s.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    return p


def test_time_progress_600s_produces_png(oft_600s_file: Path, tmp_path: Path):
    output = tmp_path / "time_progress.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.oft.plot_time_progress",
         "--input", str(oft_600s_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout
    # 600s → floor(600/300) = 2 bins
    import json
    for line in result.stdout.splitlines():
        if line.startswith("[result]"):
            data = json.loads(line[len("[result]"):].strip())
            assert data["n_bins"] == 2
            break


def test_time_progress_350s_one_bin(oft_350s_file: Path, tmp_path: Path):
    """350s > 300s → 1 bin, succeeds (catalog when filter handles the <=300 case)."""
    output = tmp_path / "time_progress.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.oft.plot_time_progress",
         "--input", str(oft_350s_file), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout
    import json
    for line in result.stdout.splitlines():
        if line.startswith("[result]"):
            data = json.loads(line[len("[result]"):].strip())
            assert data["n_bins"] == 1
            break
