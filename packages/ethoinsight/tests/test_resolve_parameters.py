"""Sprint 2b unit tests: resolve.py parameter pipeline.

Tests _compute_parameters_in_use, _metric_uses_pendulum, CLI --overrides-file,
PlanMetric.args injection with --parameters-json.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ethoinsight.catalog.loader import CommonCatalog, SharedParameters
from ethoinsight.catalog.resolve import (
    _compute_parameters_in_use,
    _metric_uses_pendulum,
    _metric_uses_velocity_immobility,
    resolve_metrics,
)
from ethoinsight.catalog.schema import MetricEntry, ParamSpec, ParadigmParameters


# ============================================================================
# Fixtures
# ============================================================================

def _make_metric(
    id: str = "immobility_time",
    script: str = "ethoinsight.scripts.fst.compute_immobility_time",
    parameters_ref: list[str] | None = None,
    parameters: dict | None = None,
) -> MetricEntry:
    return MetricEntry(
        id=id,
        script=script,
        requires_columns=["mobility_state*"],
        output_unit="seconds",
        display_name_zh="不动时间",
        unit_zh="秒",
        one_liner="test",
        direction_for_anxiety=None,
        statistical_default="groupwise_compare",
        parameters_ref=parameters_ref or [],
        parameters=parameters or {},
    )


def _make_shared_params() -> dict[str, ParamSpec]:
    return {
        "velocity_threshold": ParamSpec(
            default=30.0, unit="mm/s", description="vel thresh",
            tunable_by_user=True, valid_range=[1.0, 100.0],
        ),
        "velocity_min_duration": ParamSpec(
            default=25, unit="samples", description="min dur",
            tunable_by_user=True, valid_range=[5, 250],
        ),
        "pendulum_periodicity_threshold": ParamSpec(
            default=0.55, unit="ratio", description="pend thresh",
            tunable_by_user=True, valid_range=[0.1, 1.0],
        ),
    }


def _make_paradigm_params() -> dict[str, ParamSpec]:
    return {
        "pendulum_periodicity_threshold": ParamSpec(
            default=0.55, unit="ratio", description="pend thresh",
            tunable_by_user=True, valid_range=[0.1, 1.0],
        ),
    }


# ============================================================================
# Test _compute_parameters_in_use
# ============================================================================


class TestComputeParametersInUse:
    def test_parameters_in_use_default_no_overrides(self):
        """overrides 为空 → parameters_in_use 含 catalog default (via parameters_ref)."""
        m = _make_metric(parameters_ref=["velocity_threshold", "velocity_min_duration"])
        result = _compute_parameters_in_use(
            metric=m,
            shared_params=_make_shared_params(),
            paradigm_params={},
            overrides={},
        )
        # velocity_* via ref; pendulum_* via heuristic (id contains 'immobility')
        assert result["velocity_threshold"] == 30.0
        assert result["velocity_min_duration"] == 25
        # pendulum injected via heuristic too
        assert result["pendulum_periodicity_threshold"] == 0.55

    def test_parameters_in_use_with_override(self):
        """overrides={"velocity_threshold": 5.0} → 该字段被覆盖，其他保持 default."""
        m = _make_metric(parameters_ref=["velocity_threshold", "velocity_min_duration"])
        result = _compute_parameters_in_use(
            metric=m,
            shared_params=_make_shared_params(),
            paradigm_params={},
            overrides={"velocity_threshold": 5.0},
        )
        assert result["velocity_threshold"] == 5.0
        assert result["velocity_min_duration"] == 25

    def test_parameters_in_use_unrelated_override_silently_skipped(self):
        """overrides 含 metric 不需要的 key → 不报错，不进 parameters_in_use."""
        m = _make_metric(parameters_ref=["velocity_threshold"])
        result = _compute_parameters_in_use(
            metric=m,
            shared_params=_make_shared_params(),
            paradigm_params={},
            overrides={"fake_param": 999},
        )
        assert "fake_param" not in result
        assert "velocity_threshold" in result

    def test_parameters_in_use_pendulum_only_for_swim_test(self):
        """FST/TST immobility metric 应含 pendulum_* 参数；EPM metric 不含."""
        fst_metric = _make_metric(
            id="immobility_time",
            script="ethoinsight.scripts.fst.compute_immobility_time",
        )
        assert _metric_uses_pendulum(fst_metric) is True

        epm_metric = _make_metric(
            id="open_arm_time",
            script="ethoinsight.scripts.epm.compute_open_arm_time",
        )
        assert _metric_uses_pendulum(epm_metric) is False

    def test_parameters_in_use_velocity_immobility_heuristic(self):
        """metric.id 含 'immobility' → _metric_uses_velocity_immobility 为 True."""
        assert _metric_uses_velocity_immobility(_make_metric(id="immobility_time")) is True
        assert _metric_uses_velocity_immobility(_make_metric(id="open_arm_time")) is False

    def test_parameters_in_use_no_ref_no_params(self):
        """metric 无 parameters_ref 无 parameters → 空 dict (unless pendulum/velocity heuristic matches)."""
        m = _make_metric(id="open_arm_time", script="ethoinsight.scripts.epm.compute_open_arm_time")
        result = _compute_parameters_in_use(
            metric=m,
            shared_params=_make_shared_params(),
            paradigm_params={},
            overrides={},
        )
        assert result == {}


# ============================================================================
# Test CLI --overrides-file
# ============================================================================


class TestResolveCliOverridesFile:
    def test_resolve_cli_overrides_file_loaded(self):
        """CLI 跑 --overrides-file overrides.json → plan.json 含 overrides 后的 parameters_in_use."""
        from ethoinsight.catalog.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write columns.json
            cols_path = Path(tmpdir) / "columns.json"
            cols_path.write_text(json.dumps({"columns": ["mobility_state", "trial_time"]}))

            # Write raw-files.json
            raw_path = Path(tmpdir) / "raw_files.json"
            raw_path.write_text(json.dumps(["/tmp/test_data.txt"]))

            # Write overrides.json
            overrides_path = Path(tmpdir) / "overrides.json"
            overrides_path.write_text(json.dumps({"velocity_threshold": 5.0}))

            output_path = Path(tmpdir) / "plan.json"

            ret = cli_main([
                "--paradigm", "fst",
                "--columns-file", str(cols_path),
                "--raw-files-json", str(raw_path),
                "--workspace-dir", tmpdir,
                "--overrides-file", str(overrides_path),
                "--output", str(output_path),
            ])
            assert ret == 0

            plan = json.loads(output_path.read_text())
            # Check that immobility_time metric has velocity_threshold overridden
            immobility_metrics = [m for m in plan["metrics"] if m["id"] == "immobility_time"]
            assert len(immobility_metrics) >= 1
            m = immobility_metrics[0]
            assert m["parameters_in_use"].get("velocity_threshold") == 5.0

    def test_resolve_cli_overrides_file_invalid_json(self):
        """--overrides-file 指向非 JSON 文件 → exit 1."""
        from ethoinsight.catalog.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            cols_path = Path(tmpdir) / "columns.json"
            cols_path.write_text(json.dumps({"columns": ["mobility_state", "trial_time"]}))

            raw_path = Path(tmpdir) / "raw_files.json"
            raw_path.write_text(json.dumps(["/tmp/test_data.txt"]))

            overrides_path = Path(tmpdir) / "overrides.json"
            overrides_path.write_text("NOT VALID JSON{{{}}")

            output_path = Path(tmpdir) / "plan.json"

            ret = cli_main([
                "--paradigm", "fst",
                "--columns-file", str(cols_path),
                "--raw-files-json", str(raw_path),
                "--workspace-dir", tmpdir,
                "--overrides-file", str(overrides_path),
                "--output", str(output_path),
            ])
            assert ret == 1


# ============================================================================
# Test PlanMetric.args includes --parameters-json
# ============================================================================


class TestPlanMetricArgs:
    def test_metric_cli_args_include_parameters_json(self):
        """PlanMetric.args 包含 --parameters-json."""
        from ethoinsight.catalog.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmpdir:
            cols_path = Path(tmpdir) / "columns.json"
            cols_path.write_text(json.dumps({"columns": ["mobility_state", "trial_time"]}))

            raw_path = Path(tmpdir) / "raw_files.json"
            raw_path.write_text(json.dumps(["/tmp/test_data.txt"]))

            output_path = Path(tmpdir) / "plan.json"

            ret = cli_main([
                "--paradigm", "fst",
                "--columns-file", str(cols_path),
                "--raw-files-json", str(raw_path),
                "--workspace-dir", tmpdir,
                "--output", str(output_path),
            ])
            assert ret == 0

            plan = json.loads(output_path.read_text())
            immobility_metrics = [m for m in plan["metrics"] if m["id"] == "immobility_time"]
            assert len(immobility_metrics) >= 1
            m = immobility_metrics[0]
            # args should contain --parameters-json
            assert "--parameters-json" in m["args"]
            # The JSON value should be parseable
            idx = m["args"].index("--parameters-json")
            params_json = m["args"][idx + 1]
            params = json.loads(params_json)
            assert isinstance(params, dict)
