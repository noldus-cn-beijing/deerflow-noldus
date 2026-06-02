"""Unit tests for MetricStat.parameters_used and CodeExecutorHandoff new fields (Sprint 0)."""

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, MetricStat


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
