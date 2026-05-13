"""Shared pytest fixtures for scripts subprocess tests."""

from __future__ import annotations

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

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

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


def _df_to_ethovision_file(
    df: pd.DataFrame, path: Path, *, subject: str = "Subject 1"
) -> None:
    """Write `df` as a minimal EthoVision-style trajectory file (UTF-16-LE BOM, semicolon-delimited).

    Format matches what ``ethoinsight.parse.parse_trajectory()`` expects:
    - First line: ``"标题行数";"<N>"`` where N = number of header lines
    - Header lines include "对象名称" pointing to `subject`
    - Then a unit row, then the data rows

    parse_header layout (0-indexed lines):
      [0] header_count -> header_lines = N
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
            pattern = ([1] * 5 + [0] * 35) * 10  # 12.5% open arm
        df = _make_epm_df(n_frames=400, open_arm_pattern=pattern[:400])
        path = tmp_path / f"epm_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_epm_df():
    """Expose `_make_epm_df` directly for tests that don't need a file."""
    return _make_epm_df


# ============================================================================
# Zero Maze DataFrame builder
# ============================================================================


def _make_zero_maze_df(
    n_frames: int = 100,
    *,
    open_zone_cols: list[str] | None = None,
    closed_zone_cols: list[str] | None = None,
    open_zone_pattern: list[int] | None = None,
    closed_zone_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if open_zone_cols is None:
        open_zone_cols = ["in_zone_open_1", "in_zone_open_2"]
    if closed_zone_cols is None:
        closed_zone_cols = ["in_zone_closed_1", "in_zone_closed_2"]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    if open_zone_pattern is None:
        pat = ([1] * 20 + [0] * 20) * (n_frames // 40 + 1)
        open_zone_pattern = pat[:n_frames]
    if closed_zone_pattern is None:
        closed_zone_pattern = [1 - v for v in open_zone_pattern]

    for col in open_zone_cols:
        df[col] = open_zone_pattern
    for col in closed_zone_cols:
        df[col] = closed_zone_pattern

    return df


@pytest.fixture
def zero_maze_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic Zero Maze trajectory file with mixed open/closed zone occupancy."""
    df = _make_zero_maze_df(n_frames=200)
    path = tmp_path / "zm_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def zero_maze_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic Zero Maze trajectory files (3 control, 3 treatment) with diverging patterns."""
    files = []
    for i in range(1, 7):
        if i <= 3:
            pattern = ([1] * 30 + [0] * 10) * 10  # 75% open zone
        else:
            pattern = ([1] * 5 + [0] * 35) * 10  # 12.5% open zone
        df = _make_zero_maze_df(n_frames=400, open_zone_pattern=pattern[:400])
        path = tmp_path / f"zm_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


# ============================================================================
# Synthetic OFT DataFrame builder
# ============================================================================


def _make_oft_df(
    n_frames: int = 100,
    *,
    center_zone_col: str = "in_zone_center",
    center_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if center_pattern is None:
        # Alternating: 20 frames in, 20 out, repeat
        pat = ([1] * 20 + [0] * 20) * (n_frames // 40 + 1)
        center_pattern = pat[:n_frames]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    df[center_zone_col] = center_pattern
    # Periphery zone = inverse of center (needed for thigmotaxis_index)
    df["in_zone_periphery"] = [1 - v for v in center_pattern]

    return df


@pytest.fixture
def oft_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic OFT trajectory file with mixed center/periphery occupancy."""
    df = _make_oft_df(n_frames=200)
    path = tmp_path / "oft_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def oft_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic OFT trajectory files (3 control, 3 treatment) with diverging patterns."""
    files = []
    for i in range(1, 7):
        # control: more center time; treatment: less
        if i <= 3:
            pattern = ([1] * 30 + [0] * 10) * 10  # 75% center
        else:
            pattern = ([1] * 5 + [0] * 35) * 10  # 12.5% center
        df = _make_oft_df(n_frames=400, center_pattern=pattern[:400])
        path = tmp_path / f"oft_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_oft_df():
    """Expose `_make_oft_df` directly for tests that don't need a file."""
    return _make_oft_df


# ============================================================================
# Synthetic Shoaling DataFrame builder (multi-fish, simultaneous tracking)
# ============================================================================


def _make_shoaling_df(
    n_frames: int = 100,
    *,
    start_x: float = 250.0,
    start_y: float = 250.0,
    dx_per_frame: float = 0.5,
    dy_per_frame: float = 0.3,
    noise_scale: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a single-fish trajectory DataFrame for shoaling multi-subject tests.

    Each fish starts at (start_x, start_y) and moves linearly with small
    Gaussian noise added per frame. Fish with different start positions and
    drift directions produce non-zero IID, NND, and polarity values.
    """
    rng = np.random.default_rng(seed)
    dt = 0.04  # seconds per frame (25 fps)
    times = np.arange(n_frames, dtype=float) * dt

    # Base positions with linear drift + noise
    x_base = start_x + np.arange(n_frames, dtype=float) * dx_per_frame
    y_base = start_y + np.arange(n_frames, dtype=float) * dy_per_frame
    x_noise = np.cumsum(rng.normal(0, noise_scale, n_frames))
    y_noise = np.cumsum(rng.normal(0, noise_scale, n_frames))

    x_center = x_base + x_noise
    y_center = y_base + y_noise

    # Distance moved between consecutive frames
    dx = np.diff(x_center, prepend=x_center[0])
    dy = np.diff(y_center, prepend=y_center[0])
    distance_moved = np.sqrt(dx**2 + dy**2)

    velocity = distance_moved / dt

    return pd.DataFrame(
        {
            "trial_time": times,
            "x_center": x_center,
            "y_center": y_center,
            "distance_moved": distance_moved,
            "velocity": velocity,
        }
    )


@pytest.fixture
def shoaling_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 5 synthetic shoaling trajectory files (one per fish, simultaneously tracked).

    Each fish starts at a distinct position and drifts in a different direction,
    producing meaningful IID, NND, and polarity metrics.
    """
    fish_configs = [
        ("Fish 1", 200.0, 200.0, 0.5, 0.3, 42),
        ("Fish 2", 260.0, 240.0, -0.3, 0.5, 43),
        ("Fish 3", 240.0, 280.0, 0.4, -0.4, 44),
        ("Fish 4", 280.0, 220.0, -0.5, -0.2, 45),
        ("Fish 5", 220.0, 260.0, 0.2, -0.5, 46),
    ]
    files = []
    for name, sx, sy, dx, dy, seed in fish_configs:
        df = _make_shoaling_df(
            n_frames=200,
            start_x=sx,
            start_y=sy,
            dx_per_frame=dx,
            dy_per_frame=dy,
            seed=seed,
        )
        path = tmp_path / f"shoaling_{name.replace(' ', '_').lower()}.txt"
        _df_to_ethovision_file(df, path, subject=name)
        files.append(path)
    return files


# ============================================================================
# Synthetic FST DataFrame builder
# ============================================================================


def _make_fst_df(
    n_frames: int = 100,
    *,
    mobility_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic FST DataFrame with Mobility_State column.

    Mobility_State: 0 = immobile, 1 = mobile.
    """
    rng = np.random.default_rng(seed)

    if mobility_pattern is None:
        # Default: alternating 20 immobile, 20 mobile
        pat = ([0] * 20 + [1] * 20) * (n_frames // 40 + 1)
        mobility_pattern = pat[:n_frames]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "Mobility_State": mobility_pattern,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    return df


@pytest.fixture
def fst_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic FST trajectory file with mixed mobility states."""
    df = _make_fst_df(n_frames=200)
    path = tmp_path / "fst_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def fst_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic FST trajectory files (3 control, 3 treatment)."""
    files = []
    for i in range(1, 7):
        # control: more immobility (Mobility_State=0 dominant) -> depression-like
        # treatment: less immobility (Mobility_State=1 dominant) -> antidepressant effect
        if i <= 3:
            pattern = ([0] * 30 + [1] * 10) * 10  # 75% immobile
        else:
            pattern = ([1] * 30 + [0] * 10) * 10  # 25% immobile
        df = _make_fst_df(n_frames=400, mobility_pattern=pattern[:400])
        path = tmp_path / f"fst_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_fst_df():
    """Expose `_make_fst_df` directly for tests that don't need a file."""
    return _make_fst_df


# ============================================================================
# Synthetic TST DataFrame builder
# ============================================================================


def _make_tst_df(
    n_frames: int = 100,
    *,
    mobility_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic TST DataFrame with Activity_State column.

    Activity_State: 0 = immobile, 1 = mobile.
    """
    rng = np.random.default_rng(seed)

    if mobility_pattern is None:
        # Default: alternating 20 immobile, 20 mobile
        pat = ([0] * 20 + [1] * 20) * (n_frames // 40 + 1)
        mobility_pattern = pat[:n_frames]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "Activity_State": mobility_pattern,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    return df


@pytest.fixture
def tst_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic TST trajectory file with mixed activity states."""
    df = _make_tst_df(n_frames=200)
    path = tmp_path / "tst_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def tst_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic TST trajectory files (3 control, 3 treatment)."""
    files = []
    for i in range(1, 7):
        # control: more immobility (Activity_State=0 dominant) -> depression-like
        # treatment: less immobility (Activity_State=1 dominant) -> antidepressant effect
        if i <= 3:
            pattern = ([0] * 30 + [1] * 10) * 10  # 75% immobile
        else:
            pattern = ([1] * 30 + [0] * 10) * 10  # 25% immobile
        df = _make_tst_df(n_frames=400, mobility_pattern=pattern[:400])
        path = tmp_path / f"tst_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_tst_df():
    """Expose `_make_tst_df` directly for tests that don't need a file."""
    return _make_tst_df


# ============================================================================
# Synthetic LDB DataFrame builder
# ============================================================================


def _make_ldb_df(
    n_frames: int = 100,
    *,
    light_zone: str = "in_zone_light",
    dark_zone: str = "in_zone_dark",
    light_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic LDB DataFrame with light/dark zone columns.

    Columns built:
      - ``in_zone_light`` (1 in light, 0 in dark)
      - ``in_zone_dark`` (inverse of light)
      - ``trial_time``, ``x_center``, ``y_center``, etc.
    """
    rng = np.random.default_rng(seed)

    if light_pattern is None:
        # Start in dark, alternate: 30 dark, 20 light, repeat
        pat = ([0] * 30 + [1] * 20) * (n_frames // 50 + 1)
        light_pattern = pat[:n_frames]

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
        }
    )

    df[light_zone] = light_pattern
    df[dark_zone] = [1 - v for v in light_pattern]

    return df


@pytest.fixture
def ldb_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic LDB trajectory file with mixed light/dark zone occupancy."""
    df = _make_ldb_df(n_frames=200)
    path = tmp_path / "ldb_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def ldb_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic LDB trajectory files (3 control, 3 treatment) with diverging patterns."""
    files = []
    for i in range(1, 7):
        # control: more light time (less anxious); treatment: less light time
        if i <= 3:
            pattern = ([1] * 30 + [0] * 10) * 10  # 75% light
        else:
            pattern = ([1] * 5 + [0] * 35) * 10  # 12.5% light
        df = _make_ldb_df(n_frames=400, light_pattern=pattern[:400])
        path = tmp_path / f"ldb_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_ldb_df():
    """Expose `_make_ldb_df` directly for tests that don't need a file."""
    return _make_ldb_df
