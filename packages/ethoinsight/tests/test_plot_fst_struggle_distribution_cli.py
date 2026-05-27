"""FST struggle_distribution CLI tests: single subject + multi-subject."""
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


def _make_fst_df(n: int = 60) -> pd.DataFrame:
    immobile = np.zeros(n, dtype=int)
    immobile[20:40] = 1
    return pd.DataFrame({
        "trial_time": np.arange(n, dtype=float) * 0.04,
        "mobility_state_immobile": immobile,
    })


def test_struggle_distribution_single_subject(tmp_path: Path):
    df = _make_fst_df()
    p = tmp_path / "s1.txt"
    _df_to_ethovision_file(df, p, subject="Subject 1")
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(p)]))
    output = tmp_path / "struggle.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.fst.plot_struggle_distribution",
         "--inputs", str(inputs_json), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_struggle_distribution_multi_subject(tmp_path: Path):
    paths = []
    for i in range(1, 4):
        df = _make_fst_df()
        p = tmp_path / f"s{i}.txt"
        _df_to_ethovision_file(df, p, subject=f"Subject {i}")
        paths.append(p)
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(p) for p in paths]))
    output = tmp_path / "struggle.png"
    result = subprocess.run(
        [sys.executable, "-m", "ethoinsight.scripts.fst.plot_struggle_distribution",
         "--inputs", str(inputs_json), "--output", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout
