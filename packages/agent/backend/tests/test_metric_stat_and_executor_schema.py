"""Unit tests for MetricStat.parameters_used and CodeExecutorHandoff new fields (Sprint 0)."""

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, MetricStat, ParameterAuditFinding


class TestMetricStatParametersUsed:
    """parameters_used: dict[str, float|int|str], default empty dict."""

    def test_default_empty(self):
        m = MetricStat(mean=1.0)
        assert m.parameters_used == {}

    def test_with_parameters(self):
        m = MetricStat(
            mean=42.5,
            std=3.1,
            parameters_used={"velocity_threshold": 30.0, "velocity_min_duration": 25, "unit": "mm/s"},
        )
        assert m.parameters_used["velocity_threshold"] == 30.0
        assert m.parameters_used["velocity_min_duration"] == 25
        assert m.parameters_used["unit"] == "mm/s"

    def test_extra_fields_still_allowed(self):
        m = MetricStat(mean=1.0, median=2.0, iqr=0.5)
        assert m.median == 2.0  # extra="allow"

    def test_list_valued_zone_params(self):
        """EPM/Zero Maze 多列 zone 聚合参数是 list[str]，必须合法（thread 38be2753 实证）。"""
        m = MetricStat(mean=None, parameters_used={"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]})
        assert m.parameters_used["open_arm_zones"] == ["open"]
        assert m.parameters_used["closed_arm_zones"] == ["closed"]

    def test_scalar_params_still_work(self):
        """放宽 union 不得破坏旧标量参数（smart-union 不坍缩 str↔list 回归）。"""
        m = MetricStat(parameters_used={"velocity_threshold": 30.0, "unit": "mm/s", "n": 25})
        assert m.parameters_used == {"velocity_threshold": 30.0, "unit": "mm/s", "n": 25}


class TestParameterAuditFindingListUsedValue:
    """A1b: ParameterAuditFinding.used_value 接受 list[str]（data_analyst.py:210 指示逐字拷贝 parameters_used 值）。"""

    def test_list_used_value_accepted(self):
        """used_value=["open"] 不抛 ValidationError。"""
        finding = ParameterAuditFinding(
            parameter="open_arm_zones",
            metric="open_arm_time_ratio",
            severity="warning",
            used_value=["open"],
            observed_distribution={"n_subjects": 10},
            mismatch_kind="category_mismatch",
            suggestion="建议确认分析区映射。",
        )
        assert finding.used_value == ["open"]


class TestCodeExecutorHandoffNewFields:
    """paradigm, ev19_template, analysis_config_id are Sprint 0 additions."""

    def _base_payload(self, **overrides):
        payload = {
            "status": "completed",
            "summary": "done",
            "paradigm": "fst",
            "analysis_config_id": "PENDING_SPRINT_4.5",
        }
        payload.update(overrides)
        return payload

    def test_paradigm_required(self):
        with pytest.raises(ValidationError, match="paradigm"):
            CodeExecutorHandoff(status="completed", summary="done", analysis_config_id="x")

    def test_analysis_config_id_required(self):
        # CodeExecutorHandoff 是 handoff 链源头，由 seal tool 从 experiment-context.json
        # 总会注入 analysis_config_id（seal_handoff_tools.py setdefault），故源头强制 required
        # 保 lineage 完整。下游 handoff (DataAnalyst/ChartMaker/ReportWriter) 仍 optional
        # default='PENDING'（见 test_analysis_config_id.py 的 *_for_downstream 测试）。
        with pytest.raises(ValidationError, match="analysis_config_id"):
            CodeExecutorHandoff(status="completed", summary="done", paradigm="fst")

    def test_ev19_template_optional(self):
        h = CodeExecutorHandoff(**self._base_payload())
        assert h.ev19_template is None

    def test_ev19_template_set(self):
        h = CodeExecutorHandoff(**self._base_payload(ev19_template="fst-modified"))
        assert h.ev19_template == "fst-modified"

    def test_all_existing_fields_still_work(self):
        h = CodeExecutorHandoff(
            status="partial",
            summary="partial run",
            paradigm="epm",
            analysis_config_id="abc123",
            metrics_summary={"control": {"mean_nnd": {"mean": 42.5, "std": 3.1}}},
            per_subject={"Subject 1": {"mean_nnd": 42.5}},
            errors=["something"],
            confidence=0.8,
        )
        assert h.paradigm == "epm"
        assert h.analysis_config_id == "abc123"
        assert h.confidence == 0.8
