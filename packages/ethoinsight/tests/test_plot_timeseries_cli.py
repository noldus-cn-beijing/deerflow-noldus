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


@pytest.fixture
def fst_trajectory_file_no_distance_moved(tmp_path: Path) -> Path:
    """FST trajectory file mirroring the real EthoVision Porsolt export columns.

    Real FST exports do NOT contain `distance_moved` — only `mobility_continuous` /
    `mobility_state_immobile` etc. This fixture catches the regression where
    plot_timeseries fell back to a missing column and drew "No data".
    """
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(-20, -10, n),
            "y_center": rng.uniform(5, 11, n),
            "mobility_continuous": rng.uniform(0, 100, n),
            "mobility_state_immobile": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "fst_subject1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


def test_plot_timeseries_fst_paradigm_picks_mobility_continuous(
    fst_trajectory_file_no_distance_moved: Path, tmp_path: Path
):
    """`--paradigm fst` must default y_col to mobility_continuous, not distance_moved.

    Regression for thread f3fbce44 (2026-05-21) where chart-maker drew empty
    "No data" plots because the FST mapping was missing from
    `_DEFAULT_Y_COL_BY_PARADIGM`.
    """
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(fst_trajectory_file_no_distance_moved),
            "--output", str(output),
            "--paradigm", "fst",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"FST timeseries CLI failed: stderr={result.stderr}")
    assert output.exists()
    assert '"y_col": "mobility_continuous"' in result.stdout, (
        f"Expected y_col=mobility_continuous for paradigm=fst, got stdout={result.stdout}"
    )


def test_plot_timeseries_fallback_when_default_column_missing(
    fst_trajectory_file_no_distance_moved: Path, tmp_path: Path
):
    """When neither --y-col nor --paradigm is given and the default column
    (distance_moved) is absent, the CLI must pick a present numeric column
    instead of producing an empty "No data" plot.

    Defensive: protects future paradigms whose authors forget to add a mapping.
    """
    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(fst_trajectory_file_no_distance_moved),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"timeseries CLI failed when default column missing: stderr={result.stderr}")
    assert output.exists()
    assert '"y_col": "distance_moved"' not in result.stdout, (
        "Falling back to distance_moved when the column is absent produces "
        "an empty 'No data' plot — the CLI should pick another numeric column."
    )


def test_plot_timeseries_ldb_paradigm_maps_to_light_time_ratio(tmp_path: Path):
    """LDB paradigm default y_col must exist in real LDB exports."""
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(0, 50, n),
            "y_center": rng.uniform(0, 50, n),
            "in_zone_light": rng.integers(0, 2, n),
            "in_zone_dark": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "ldb_subject1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")

    output = tmp_path / "out.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts._common.plot_timeseries",
            "--input", str(path),
            "--output", str(output),
            "--paradigm", "light_dark_box",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"LDB timeseries CLI failed: stderr={result.stderr}")
    assert output.exists()
    # in_zone_light is the actual EthoVision column; the CLI may use it
    # directly or via a derived light_time_ratio — either is fine as long as
    # the result is NOT the absent distance_moved default.
    assert '"y_col": "distance_moved"' not in result.stdout
