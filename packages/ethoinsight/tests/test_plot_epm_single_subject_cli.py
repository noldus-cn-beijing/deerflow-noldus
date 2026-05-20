"""2026-05-20: EPM 单样本场景下的两张基础图脚本 CLI。

handoff #2 修复:之前 epm.yaml 仅注册了 plot_box_open_arm (n_per_group>=3 组间箱线图),
单样本场景命中 0 + fallback 仅给 1 张 trajectory_plot,缺
- 开臂时间占比柱状图
- 区域进入次数分布柱状图

两个新脚本读单个 trajectory file,生成单样本可解读的柱状图。
"""
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
def epm_trajectory_file(tmp_path: Path) -> Path:
    """Write a minimal EPM trajectory with open/closed arm zone columns."""
    rng = np.random.default_rng(42)
    n = 60
    # Spend ~25% of time in open arms, ~50% in closed arms, ~25% in center
    in_open_1 = np.zeros(n, dtype=int)
    in_open_1[5:10] = 1
    in_open_1[30:40] = 1
    in_closed_1 = np.zeros(n, dtype=int)
    in_closed_1[10:25] = 1
    in_closed_1[40:55] = 1
    in_center = np.ones(n, dtype=int) - in_open_1 - in_closed_1
    in_center = np.clip(in_center, 0, 1)

    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "in_zone_open_arms_1__center_point_": in_open_1,
            "in_zone_closed_arms_1__center_point_": in_closed_1,
            "in_zone_center_point__center_point_": in_center,
        }
    )
    path = tmp_path / "subject1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


def test_plot_open_arm_time_ratio_bar_cli_writes_png(epm_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "open_arm_time_ratio.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar",
            "--input", str(epm_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_plot_zone_entry_distribution_cli_writes_png(epm_trajectory_file: Path, tmp_path: Path):
    output = tmp_path / "zone_entry_distribution.png"
    result = subprocess.run(
        [
            sys.executable, "-m", "ethoinsight.scripts.epm.plot_zone_entry_distribution",
            "--input", str(epm_trajectory_file),
            "--output", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"CLI failed: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists()
    assert output.stat().st_size > 0
    assert "[result]" in result.stdout
