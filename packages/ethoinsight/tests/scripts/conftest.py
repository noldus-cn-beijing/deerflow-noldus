"""Shared pytest fixtures for scripts subprocess tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Synthetic EPM DataFrame builder (lifted from tests/test_metrics_epm.py)
# ============================================================================


def _make_epm_df(
    n_frames: int = 100,
    *,
    open_arm_cols: list[str] | None = None,
    closed_arm_cols: list[str] | None = None,
    center_cols: list[str] | None = None,
    open_arm_pattern: list[int] | None = None,
    closed_arm_pattern: list[int] | None = None,
    center_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if open_arm_cols is None:
        open_arm_cols = ["in_zone_open_arm_1"]
    if closed_arm_cols is None:
        closed_arm_cols = ["in_zone_closed_arm_1"]
    if center_cols is None:
        center_cols = ["in_zone_center-point"]

    df = pd.DataFrame({
        "trial_time": np.arange(n_frames, dtype=float) * 0.04,
        "x_center": rng.uniform(100, 500, n_frames),
        "y_center": rng.uniform(100, 500, n_frames),
        "distance_moved": rng.uniform(0, 5, n_frames),
        "velocity": rng.uniform(0, 20, n_frames),
    })

    if open_arm_pattern is None:
        # Alternating: 20 frames in, 20 out, repeat
        pat = ([1] * 20 + [0] * 20) * (n_frames // 40 + 1)
        open_arm_pattern = pat[:n_frames]
    if closed_arm_pattern is None:
        # Inverse of open arm by default
        closed_arm_pattern = [1 - v for v in open_arm_pattern]
    if center_pattern is None:
        center_pattern = [0] * n_frames

    for col in open_arm_cols:
        df[col] = open_arm_pattern
    for col in closed_arm_cols:
        df[col] = closed_arm_pattern
    for col in center_cols:
        df[col] = center_pattern

    return df


# ============================================================================
# EthoVision trajectory file fixture
# ============================================================================


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    """Write `df` as a minimal EthoVision-style trajectory file (UTF-16-LE BOM, semicolon-delimited).

    Format matches what ``ethoinsight.parse.parse_trajectory()`` expects:
    - First line: ``"标题行数";"<N>"`` where N = number of header lines
    - Header lines include "对象名称" pointing to `subject`
    - Then a unit row, then the data rows

    parse_header layout (0-indexed lines):
      [0] header_count → header_lines = N
      [1..N-3] metadata K-V pairs
      [N-2] column names
      [N-1] units
      [N..] data rows
    """
    columns = list(df.columns)
    n_header_lines = 6  # 1 title-count + 3 metadata + 1 column-names + 1 units

    lines: list[str] = []
    # Line 0: header count
    lines.append(f'"标题行数";"{n_header_lines}"')
    # Lines 1-3: metadata
    lines.append(f'"对象名称";"{subject}"')
    lines.append('"试验名称";"Trial 1"')
    lines.append('"竞技场名称";"Arena 1"')
    # Line 4: column names
    lines.append(";".join(f'"{c}"' for c in columns))
    # Line 5: units (placeholder)
    lines.append(";".join(['""'] * len(columns)))
    # Lines 6+: data rows
    for _, row in df.iterrows():
        values = []
        for v in row.values:
            if pd.isna(v):
                values.append('"-"')
            else:
                values.append(f'"{v}"')
        lines.append(";".join(values))

    content = "\n".join(lines) + "\n"
    # Prepend BOM and write as UTF-16-LE
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")  # UTF-16-LE BOM
        f.write(content.encode("utf-16-le"))


@pytest.fixture
def epm_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic EPM trajectory file with mixed open/closed arm occupancy."""
    df = _make_epm_df(n_frames=200)
    path = tmp_path / "epm_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def epm_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic EPM trajectory files (3 control, 3 treatment) with diverging patterns."""
    files = []
    for i in range(1, 7):
        # control: more open arm time; treatment: less
        if i <= 3:
            pattern = ([1] * 30 + [0] * 10) * 10  # 75% open arm
        else:
            pattern = ([1] * 5 + [0] * 35) * 10   # 12.5% open arm
        df = _make_epm_df(n_frames=400, open_arm_pattern=pattern[:400])
        path = tmp_path / f"epm_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_epm_df():
    """Expose `_make_epm_df` directly for tests that don't need a file."""
    return _make_epm_df
