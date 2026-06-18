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
    """ethoinsight handoff 不再声明 task_context（spec 2026-06-18 移除死重量）。

    旧测试曾断言 ``handoff.task_context is None``；字段移除后改为断言字段不存在 +
    旁路引用字段（outlier_diagnostics_* / output_files_*，CodeExecutorHandoff）有默认。
    """

    def test_code_executor_without_task_context(self):
        d = {
            "status": "completed",
            "summary": "old handoff",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
            "metrics_summary": {"group_a": {"immobility": {"mean": 50.0}}},
        }
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert "task_context" not in CodeExecutorHandoff.model_fields, "task_context 字段应已从 CodeExecutorHandoff 移除"
        # 旁路引用字段默认值（spec 2026-06-18）。
        assert handoff.outlier_diagnostics_count == 0
        assert handoff.outlier_diagnostics_ref is None

    def test_data_analyst_without_task_context(self):
        d = {"status": "completed", "analysis_config_id": "deadbeef12345678", "key_findings": ["Treatment group shows significant difference (p=0.03)"]}
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        handoff = DataAnalystHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert "task_context" not in DataAnalystHandoff.model_fields

    def test_chart_maker_without_task_context(self):
        d = {
            "status": "completed",
            "paradigm": "fst",
            "summary": "charts done",
            "analysis_config_id": "deadbeef12345678",
            "chart_files": ["/mnt/user-data/outputs/fst_bar.png"],
        }
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        handoff = ChartMakerHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert "task_context" not in ChartMakerHandoff.model_fields

    def test_report_writer_without_task_context(self):
        d = {
            "status": "completed",
            "report_path": "/mnt/user-data/outputs/report.md",
            "analysis_config_id": "deadbeef12345678",
            "sections_written": ["Results"],
        }
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        handoff = ReportWriterHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert "task_context" not in ReportWriterHandoff.model_fields


class TestTaskContextForwardCompat:
    """旧 handoff 带 task_context → 新 schema 仍能 parse（extra="allow" 吞进 model_extra）.

    spec 2026-06-18：ethoinsight 4 个 handoff 移除 task_context 字段，但 extra="allow"
    保证旧 handoff 文件（带 task_context）不抛、可读。
    """

    def test_code_executor_with_task_context(self):
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        d = {
            "status": "completed",
            "summary": "new handoff",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
            "metrics_summary": {"group_a": {"immobility": {"mean": 50.0}}},
            "task_context": {
                "file_changes": ["/mnt/user-data/outputs/metrics.json"],
                "verify_commands": ["python -m json.tool /mnt/user-data/outputs/metrics.json > /dev/null"],
                "failed_paths": ["group B n=1 skipped"],
                "pending_items": [],
            },
        }
        # 字段已移除，但 extra="allow" 吞掉 task_context 不抛。
        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.status == "completed"
        assert "task_context" not in CodeExecutorHandoff.model_fields
        # 旧 task_context 落入 model_extra（向前兼容审计可追溯）。
        assert handoff.model_extra is not None
        assert "task_context" in handoff.model_extra

    def test_extra_fields_in_task_context_not_lost(self):
        """extra="allow" 确保旧 task_context 子字段随整块落入 model_extra 不丢."""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        d = {
            "status": "completed",
            "summary": "with extra",
            "paradigm": "fst",
            "analysis_config_id": "deadbeef12345678",
            "metrics_summary": {"group_a": {"immobility": {"mean": 50.0}}},
            "task_context": {
                "file_changes": ["/a.txt"],
                "verify_commands": [],
                "failed_paths": [],
                "pending_items": [],
                "custom_future_field": "survives",
            },
        }
        handoff = CodeExecutorHandoff.model_validate(d)
        assert handoff.model_extra is not None
        tc = handoff.model_extra["task_context"]
        assert tc["file_changes"] == ["/a.txt"]
        assert tc["custom_future_field"] == "survives"


# ============================================================================
# 2026-06-08: handoff status 三态一致性 — partial 补全 (spec §1 改动 A)
# ============================================================================


class TestHandoffStatusPartialConsistency:
    """四个 handoff schema 的 status 必须统一支持 completed / partial / failed。

    DataAnalyst/ReportWriter 此前缺 partial，prompt 教传但 schema 拒收 → Pydantic
    ValidationError → seal 失败 → subagent 卡死 (EPM dogfood 1bda1847 trace=acdfb7e5)。
    """

    def test_data_analyst_accepts_partial(self):
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        # red 锚点：修复前这里抛 ValidationError
        h = DataAnalystHandoff(status="partial")
        assert h.status == "partial"

    def test_report_writer_accepts_partial(self):
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        h = ReportWriterHandoff(status="partial", report_path="/x/r.md")
        assert h.status == "partial"

    def test_all_four_handoffs_share_same_status_enum(self):
        from deerflow.subagents.handoff_schemas import (
            CodeExecutorHandoff,
            ChartMakerHandoff,
            DataAnalystHandoff,
            ReportWriterHandoff,
        )

        for cls, extra in [
            (CodeExecutorHandoff, {"summary": "s", "paradigm": "epm", "analysis_config_id": "deadbeef12345678", "metrics_summary": {"g": {"m": {"mean": 1.0}}}}),
            (ChartMakerHandoff, {"summary": "s", "paradigm": "epm", "chart_files": ["/mnt/user-data/outputs/x.png"]}),
            (DataAnalystHandoff, {"key_findings": ["finding 1"]}),
            (ReportWriterHandoff, {"report_path": "/x/r.md", "sections_written": ["Results"]}),
        ]:
            for st in ("completed", "partial", "failed"):
                obj = cls(status=st, **extra)
                assert obj.status == st

    def test_invalid_status_still_rejected(self):
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        with pytest.raises(ValidationError):
            DataAnalystHandoff(status="garbage")


# ============================================================================
# 2026-06-12: handoff schema fail-open 防护 — completed 要求核心产物非空 (Spec S3 Part A)
# ============================================================================


class TestCompletedRequiresCoreOutput:
    """四个 handoff schema：status=completed 时核心字段必须非空。

    Fable 实测：extra="allow" + default_factory=list/dict 下，producer 字段名打错
    → 正确字段拿空默认 → 仍可标 completed → 下游读到"合法空值"（哑故障）。
    本校验把这类哑故障换成响亮 ValidationError。
    """

    # ── CodeExecutorHandoff ────────────────────────────────────────────────

    def test_code_executor_completed_empty_metrics_summary_raises(self):
        """red 锚点：status=completed + metrics_summary={} → ValidationError。"""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        with pytest.raises(ValidationError, match="metrics_summary is empty"):
            CodeExecutorHandoff(
                status="completed",
                summary="done",
                paradigm="fst",
                analysis_config_id="deadbeef12345678",
                metrics_summary={},
            )

    def test_code_executor_completed_nonempty_metrics_summary_passes(self):
        """正常 completed + 非空 metrics_summary → 通过。"""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff, MetricStat

        h = CodeExecutorHandoff(
            status="completed",
            summary="done",
            paradigm="fst",
            analysis_config_id="deadbeef12345678",
            metrics_summary={"group_a": {"immobility": MetricStat(mean=50.0)}},
        )
        assert h.status == "completed"

    def test_code_executor_partial_empty_metrics_summary_passes(self):
        """partial + 空 metrics_summary → 放行（n=1 partial 路径）。"""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        h = CodeExecutorHandoff(
            status="partial",
            summary="n=1 skipped",
            paradigm="fst",
            analysis_config_id="deadbeef12345678",
            metrics_summary={},
        )
        assert h.status == "partial"

    def test_code_executor_failed_empty_metrics_summary_passes(self):
        """failed + 空 metrics_summary → 放行。"""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        h = CodeExecutorHandoff(
            status="failed",
            summary="script crashed",
            paradigm="fst",
            analysis_config_id="deadbeef12345678",
            metrics_summary={},
        )
        assert h.status == "failed"

    def test_code_executor_typo_field_name_triggers_validator(self):
        """Fable 核心场景：字段名打错（parms）→ metrics_summary 拿空默认 → 抛错。"""
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        # extra="allow" 让 typo 字段静默进 model_extra，metrics_summary 拿空 dict
        with pytest.raises(ValidationError, match="metrics_summary is empty"):
            CodeExecutorHandoff(
                status="completed",
                summary="done",
                paradigm="fst",
                analysis_config_id="deadbeef12345678",
                parms={"velocity_threshold": 30.0},  # typo: should be parameters_used
            )

    # ── DataAnalystHandoff ──────────────────────────────────────────────────

    def test_data_analyst_completed_empty_key_findings_raises(self):
        """red 锚点：status=completed + key_findings=[] → ValidationError。"""
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        with pytest.raises(ValidationError, match="key_findings is empty"):
            DataAnalystHandoff(status="completed", key_findings=[])

    def test_data_analyst_completed_nonempty_key_findings_passes(self):
        """正常 completed + 非空 key_findings → 通过。"""
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        h = DataAnalystHandoff(status="completed", key_findings=["Treatment 组 immobility 显著升高 (p=0.03)"])
        assert h.status == "completed"

    def test_data_analyst_partial_empty_key_findings_passes(self):
        """partial + 空 key_findings → 放行（n=1 描述性路径）。"""
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        h = DataAnalystHandoff(status="partial", key_findings=[])
        assert h.status == "partial"

    def test_data_analyst_failed_empty_key_findings_passes(self):
        """failed + 空 key_findings → 放行。"""
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        h = DataAnalystHandoff(status="failed", key_findings=[])
        assert h.status == "failed"

    # ── ChartMakerHandoff ───────────────────────────────────────────────────

    def test_chart_maker_completed_empty_chart_files_raises(self):
        """red 锚点：status=completed + chart_files=[] → ValidationError。"""
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        with pytest.raises(ValidationError, match="chart_files is empty"):
            ChartMakerHandoff(
                status="completed",
                paradigm="fst",
                summary="charts done",
                chart_files=[],
            )

    def test_chart_maker_completed_nonempty_chart_files_passes(self):
        """正常 completed + 非空 chart_files → 通过。"""
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        h = ChartMakerHandoff(
            status="completed",
            paradigm="fst",
            summary="charts done",
            chart_files=["/mnt/user-data/outputs/fst_bar.png"],
        )
        assert h.status == "completed"

    def test_chart_maker_partial_empty_chart_files_passes(self):
        """partial + 空 chart_files → 放行。"""
        from deerflow.subagents.handoff_schemas import ChartMakerHandoff

        h = ChartMakerHandoff(
            status="partial",
            paradigm="fst",
            summary="only 1 chart of 3 succeeded",
            chart_files=[],
        )
        assert h.status == "partial"

    # ── ReportWriterHandoff ─────────────────────────────────────────────────

    def test_report_writer_completed_empty_sections_written_raises(self):
        """red 锚点：status=completed + sections_written=[] → ValidationError。"""
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        with pytest.raises(ValidationError, match="sections_written is empty"):
            ReportWriterHandoff(
                status="completed",
                report_path="/mnt/user-data/outputs/report.md",
                sections_written=[],
            )

    def test_report_writer_completed_nonempty_sections_passes(self):
        """正常 completed + 非空 sections_written → 通过。"""
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        h = ReportWriterHandoff(
            status="completed",
            report_path="/mnt/user-data/outputs/report.md",
            sections_written=["Results", "Discussion"],
        )
        assert h.status == "completed"

    def test_report_writer_partial_empty_sections_written_passes(self):
        """partial + 空 sections_written → 放行。"""
        from deerflow.subagents.handoff_schemas import ReportWriterHandoff

        h = ReportWriterHandoff(
            status="partial",
            report_path="/mnt/user-data/outputs/report.md",
            sections_written=[],
        )
        assert h.status == "partial"

    # ── 综合：四个 schema 的 model_validate 路径也触发 validator ──────────

    def test_model_validate_triggers_completed_check(self):
        """model_validate 路径也触发 completed validator（非仅构造函数）。"""
        from deerflow.subagents.handoff_schemas import DataAnalystHandoff

        with pytest.raises(ValidationError, match="key_findings is empty"):
            DataAnalystHandoff.model_validate({"status": "completed", "key_findings": []})
