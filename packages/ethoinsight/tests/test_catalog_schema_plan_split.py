"""W2: Plan dataclass 拆为 PlanMetrics + PlanCharts。"""
from __future__ import annotations

import pytest

from ethoinsight.catalog.schema import (
    PlanChart,
    PlanCharts,
    PlanInputs,
    PlanMetric,
    PlanMetrics,
    PlanSkipped,
    PlanStatistics,
)


def _sample_inputs() -> PlanInputs:
    return PlanInputs(raw_files=["/tmp/raw.txt"], groups_file=None, columns_file=None)


def test_plan_metrics_dataclass_has_required_fields():
    pm = PlanMetrics(
        paradigm="epm",
        ev19_template=None,
        generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(),
        metrics=[],
        statistics=None,
        skipped=[],
        notes=[],
    )
    assert pm.paradigm == "epm"
    assert pm.schema_version == "1.0"
    assert pm.metrics == []
    assert pm.statistics is None


def test_plan_charts_dataclass_has_required_fields():
    pc = PlanCharts(
        paradigm="epm",
        ev19_template=None,
        generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(),
        charts=[],
        charts_fallback_available=[],
        skipped=[],
        user_intent=None,
        notes=[],
    )
    assert pc.paradigm == "epm"
    assert pc.schema_version == "1.0"
    assert pc.charts_fallback_available == []
    assert pc.user_intent is None


def test_plan_metrics_can_hold_real_entries():
    metric = PlanMetric(id="open_arm_time", script="ethoinsight.scripts.epm.compute_open_arm_time",
                       input="/tmp/raw.txt", output="/tmp/m.json", required=True, reason="paradigm.default")
    pm = PlanMetrics(
        paradigm="epm", ev19_template="EPM_v1", generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), metrics=[metric], statistics=None, skipped=[], notes=["n=3"],
    )
    assert pm.metrics[0].id == "open_arm_time"


def test_plan_charts_can_hold_charts_and_fallback():
    main = PlanChart(id="box_open_arm", script="ethoinsight.scripts.epm.plot_box", input="/tmp/raw.txt", output="/tmp/p.png")
    fallback = PlanChart(id="trajectory_plot", script="ethoinsight.scripts._common.plot_trajectory", input="/tmp/raw.txt", output="/tmp/p.png")
    pc = PlanCharts(
        paradigm="epm", ev19_template=None, generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), charts=[main], charts_fallback_available=[fallback],
        skipped=[], user_intent="再画几个图", notes=[],
    )
    assert pc.charts[0].id == "box_open_arm"
    assert pc.charts_fallback_available[0].id == "trajectory_plot"
    assert pc.user_intent == "再画几个图"


def test_plan_metrics_and_plan_charts_coexist_with_legacy_plan():
    """W2 过渡期:旧 Plan dataclass 保留,新 PlanMetrics + PlanCharts 并列。"""
    from ethoinsight.catalog.schema import Plan
    assert hasattr(Plan, "__dataclass_fields__")
    assert "metrics" in Plan.__dataclass_fields__
    assert "charts" in Plan.__dataclass_fields__
    # 新 PlanMetrics 不再含 charts
    assert "charts" not in PlanMetrics.__dataclass_fields__
    # 新 PlanCharts 不含 metrics
    assert "metrics" not in PlanCharts.__dataclass_fields__


def test_plan_metrics_serialize_roundtrip():
    """W2 验收:serialize 可逆 — dataclasses.asdict() 完整表达所有字段。"""
    import dataclasses, json
    pm = PlanMetrics(
        paradigm="epm", ev19_template=None, generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), metrics=[], statistics=None, skipped=[], notes=[],
    )
    d = dataclasses.asdict(pm)
    assert d["schema_version"] == "1.0"
    assert d["paradigm"] == "epm"
    assert d["inputs"]["raw_files"] == ["/tmp/raw.txt"]
    # json round-trip
    raw = json.dumps(d, ensure_ascii=False)
    back = json.loads(raw)
    assert back["paradigm"] == "epm"


def test_plan_charts_serialize_roundtrip():
    import dataclasses, json
    pc = PlanCharts(
        paradigm="epm", ev19_template=None, generated_at="2026-05-18T00:00:00Z",
        inputs=_sample_inputs(), charts=[], charts_fallback_available=[],
        skipped=[], user_intent="画图", notes=[],
    )
    d = dataclasses.asdict(pc)
    assert d["schema_version"] == "1.0"
    assert d["user_intent"] == "画图"
    assert d["charts_fallback_available"] == []
    # json round-trip
    raw = json.dumps(d, ensure_ascii=False)
    back = json.loads(raw)
    assert back["user_intent"] == "画图"
