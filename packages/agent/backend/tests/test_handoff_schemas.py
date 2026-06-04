"""2026-05-20: GateSignals.statistical_validity enum 扩展 (handoff #3).

之前枚举只有 "ok" / "warning" / "failed",单样本场景下 code-executor 正确跳过
统计后写 "ok",语义不准确(没做 != 统计 OK)。扩 "skipped" 值表达"未运行统计检验"。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import GateSignals


class TestStatisticalValiditySkipped:
    def test_accepts_skipped(self):
        """skipped 是合法值。"""
        signals = GateSignals(statistical_validity="skipped")
        assert signals.statistical_validity == "skipped"

    def test_accepts_existing_values(self):
        """ok / warning / failed 仍合法。"""
        for v in ("ok", "warning", "failed"):
            signals = GateSignals(statistical_validity=v)
            assert signals.statistical_validity == v

    def test_rejects_unknown_value(self):
        """非枚举值仍被拒绝。"""
        with pytest.raises(ValidationError):
            GateSignals(statistical_validity="bogus")

    def test_default_is_ok(self):
        """缺省值不变,仍是 ok。"""
        assert GateSignals().statistical_validity == "ok"


# ============================================================================
# task_context 向后/向前兼容测试 (spec §四.2-3)
# ============================================================================


class TestTaskContextBackwardCompat:
    """旧 handoff 无 task_context 字段 → model_validate 不抛、task_context 为 None."""

    def test_code_executor_without_task_context(self):
        d = {
            "status": "completed",
            "summary": "old handoff",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
        }
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert handoff.task_context is None

    def test_data_analyst_without_task_context(self):
        d = {"status": "completed", "analysis_config_id": "deadbeef12345678"}
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        handoff = DataAnalystHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert handoff.task_context is None

    def test_chart_maker_without_task_context(self):
        d = {
            "status": "completed",
            "paradigm": "fst",
            "summary": "charts done",
            "analysis_config_id": "deadbeef12345678",
        }
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        handoff = ChartMakerHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert handoff.task_context is None

    def test_report_writer_without_task_context(self):
        d = {
            "status": "completed",
            "report_path": "/mnt/user-data/outputs/report.md",
            "analysis_config_id": "deadbeef12345678",
        }
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        handoff = ReportWriterHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert handoff.task_context is None


class TestTaskContextForwardCompat:
    """新 handoff 带 task_context → model_validate 通过、extra="allow" 保兼容."""

    def test_code_executor_with_task_context(self):
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        d = {
            "status": "completed",
            "summary": "new handoff",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
            "task_context": {
                "file_changes": ["/mnt/user-data/outputs/metrics.json"],
                "verify_commands": ["python -m json.tool /mnt/user-data/outputs/metrics.json > /dev/null"],
                "failed_paths": ["group B n=1 skipped"],
                "pending_items": [],
            },
        }
        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.task_context is not None
        assert handoff.task_context.file_changes == ["/mnt/user-data/outputs/metrics.json"]
        assert handoff.task_context.failed_paths == ["group B n=1 skipped"]
        assert handoff.task_context.pending_items == []

    def test_extra_fields_in_task_context_not_lost(self):
        """extra="allow" 确保 task_context 里未声明的额外字段也不丢失."""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        d = {
            "status": "completed",
            "summary": "with extra",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
            "task_context": {
                "file_changes": ["/a.txt"],
                "verify_commands": [],
                "failed_paths": [],
                "pending_items": [],
                "custom_future_field": "survives",
            },
        }
        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.task_context is not None
        # extra="allow" on TaskContext → custom_future_field accessible
        assert handoff.task_context.file_changes == ["/a.txt"]
