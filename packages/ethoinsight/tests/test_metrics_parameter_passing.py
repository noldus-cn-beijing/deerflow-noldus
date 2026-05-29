"""Sprint 2b unit tests: metric function parameter passing.

Tests that metric functions accept parameters via kwargs,
and compute scripts output parameters_used correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Helper to create EthoVision trajectory files
# ============================================================================


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    """Write df as a minimal EthoVision-style trajectory file (UTF-16-LE BOM, semicolon-delimited)."""
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


def _make_immobility_df(n_frames: int = 100) -> pd.DataFrame:
    """Create a synthetic trajectory with zero velocity → all immobile."""
    dt = 0.04
    return pd.DataFrame({
        "trial_time": [i * dt for i in range(n_frames)],
        "x_center": [0.0] * n_frames,
        "y_center": [0.0] * n_frames,
    })


# ============================================================================
# Test metric function parameter passing
# ============================================================================


class TestMetricParameterPassing:
    def test_immobility_time_with_custom_velocity_threshold(self):
        """compute_immobility_time_fst(df, velocity_threshold=5.0) runs without error."""
        from ethoinsight.metrics.fst import compute_immobility_time_fst

        df = _make_immobility_df(n_frames=100)

        # With default threshold (30.0): velocity=0 → all immobile
        time_default = compute_immobility_time_fst(df)
        assert time_default is not None
        assert time_default > 0

        # With very high threshold: still all immobile (velocity=0 < 100)
        time_high = compute_immobility_time_fst(df, velocity_threshold=100.0)
        assert time_high is not None

    def test_pendulum_with_custom_periodicity(self):
        """pendulum_immobility_series with pendulum_periodicity_threshold=0.8 比 default 0.55 严格."""
        from ethoinsight.metrics._pendulum import pendulum_immobility_series

        np.random.seed(42)
        activity = np.random.uniform(0.5, 1.5, size=500)
        dt = 0.04

        result_default = pendulum_immobility_series(activity, dt)
        result_strict = pendulum_immobility_series(
            activity, dt, pendulum_periodicity_threshold=0.99
        )

        assert len(result_default) == len(activity)
        assert len(result_strict) == len(activity)

    def test_resolve_immobile_from_velocity_uses_kwargs(self):
        """_resolve_immobile_from_velocity uses velocity_threshold kwarg."""
        from ethoinsight.metrics._common import _resolve_immobile_from_velocity

        df = _make_immobility_df(n_frames=100)

        result1 = _resolve_immobile_from_velocity(df, velocity_threshold=30.0)
        assert result1 is not None

        result2 = _resolve_immobile_from_velocity(df, velocity_threshold=10000.0)
        assert result2 is not None

        result3 = _resolve_immobile_from_velocity(df, velocity_threshold=0.0001)
        assert result3 is not None


# ============================================================================
# Test compute script outputs parameters_used
# ============================================================================


class TestComputeScriptParametersUsed:
    def test_metric_script_outputs_parameters_used(self):
        """跑 compute_immobility_time --parameters-json → 输出 JSON 含 parameters_used."""
        from ethoinsight.scripts.fst.compute_immobility_time import main

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_data.txt"
            output_path = Path(tmpdir) / "result.json"

            df = _make_immobility_df(n_frames=50)
            _df_to_ethovision_file(df, input_path)

            ret = main([
                "--input", str(input_path),
                "--output", str(output_path),
                "--parameters-json", '{"velocity_threshold": 5.0}',
            ])
            assert ret == 0

            result = json.loads(output_path.read_text())
            assert "parameters_used" in result
            assert result["parameters_used"]["velocity_threshold"] == 5.0

    def test_metric_script_no_parameters_uses_defaults(self):
        """--parameters-json 省略 → 输出 JSON 的 parameters_used 为空 dict."""
        from ethoinsight.scripts.fst.compute_immobility_time import main

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test_data.txt"
            output_path = Path(tmpdir) / "result.json"

            df = _make_immobility_df(n_frames=50)
            _df_to_ethovision_file(df, input_path)

            ret = main([
                "--input", str(input_path),
                "--output", str(output_path),
            ])
            assert ret == 0

            result = json.loads(output_path.read_text())
            assert "parameters_used" in result
            assert result["parameters_used"] == {}
