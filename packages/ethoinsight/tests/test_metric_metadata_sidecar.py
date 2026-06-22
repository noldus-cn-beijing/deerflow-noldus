"""spec 2026-06-22-metric-metadata-sidecar：去重展示/判读元数据旁路文件。

report-writer / data-analyst 啃 133K plan_metrics.json（140 条按 subject 重复）找 5 条
元数据，撑爆 thinking 撞 turn 超时。治本：生成去重投影 _metric_metadata.json（按 metric id
一条）。本文件坐实三件套：
  1. metric_metadata_to_dict 按 id 去重（140→5），每条 6 字段；
  2. prep_metric_plan 落盘时同源同次写旁路；
  3. 旁路反映 plan 实际集合（被 skip 的 metric 不进旁路），非 catalog 全集。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import metric_metadata_to_dict, resolve_metrics
from ethoinsight.catalog.schema import (
    PlanInputs,
    PlanMetric,
    PlanMetrics,
)

# 见 test_plan_metrics_interpretation_fields.py：让 default_metrics 全过 columns 检查的列名。
_MINIMAL_COLUMNS: dict[str, list[str]] = {
    "epm": ["in_zone_open_arms_subject", "in_zone_closed_arms_subject", "x_center", "y_center", "velocity"],
    "oft": ["in_zone_center_subject", "in_zone_periphery_subject", "x_center", "y_center", "velocity", "in_zone_subject", "distance_moved"],
}

_METADATA_FIELDS = {
    "display_name_zh",
    "unit_zh",
    "one_liner",
    "output_unit",
    "direction_for_anxiety",
    "statistical_default",
}


def _make_pm(n_subjects: int, n_metrics: int) -> PlanMetrics:
    """构造 n_subjects × n_metrics 条 PlanMetric（按 subject 重复），模拟 EPM 28×5=140 结构。"""
    metric_ids = [f"metric_{i}" for i in range(n_metrics)]
    metrics: list[PlanMetric] = []
    for subj in range(n_subjects):
        for mid in metric_ids:
            metrics.append(
                PlanMetric(
                    id=mid,
                    script=f"ethoinsight.scripts.dummy.compute_{mid}",
                    input="/mnt/user-data/uploads/raw.txt",
                    output=f"/mnt/user-data/workspace/m_{mid}_{subj}.json",
                    required=True,
                    reason="default",
                    subject_index=subj,
                    display_name_zh=f"中文指标名_{mid}",
                    unit_zh="%",
                    one_liner=f"{mid} 的一句话解释",
                    output_unit="ratio",
                    direction_for_anxiety="lower_is_anxious",
                    statistical_default="groupwise_compare",
                )
            )
    return PlanMetrics(
        paradigm="epm",
        ev19_template=None,
        generated_at="2026-06-22T00:00:00Z",
        inputs=PlanInputs(raw_files=[f"/tmp/raw_{i}.txt" for i in range(n_subjects)], groups_file=None, columns_file=None),
        metrics=metrics,
        statistics=None,
        skipped=[],
        notes=[],
    )


def test_metric_metadata_sidecar_is_deduplicated() -> None:
    """红线：28 subject × 5 metric = 140 条 → 去重后 metrics 字典只有 5 个 key，每条 6 字段。"""
    pm = _make_pm(n_subjects=28, n_metrics=5)
    assert len(pm.metrics) == 140, "fixture 前置：原始 plan 应有 140 条（28×5）"

    d = metric_metadata_to_dict(pm)

    # 顶层结构
    assert set(d.keys()) == {"paradigm", "metrics"}
    assert d["paradigm"] == "epm"

    # 去重契约：5 条而非 140 条
    assert len(d["metrics"]) == 5, f"去重后应 5 个 metric，实际 {len(d['metrics'])}"
    assert set(d["metrics"].keys()) == {f"metric_{i}" for i in range(5)}

    # 每条 6 字段且值与首个（所有同 id 行一致）相符
    for mid, meta in d["metrics"].items():
        assert set(meta.keys()) == _METADATA_FIELDS, f"{mid}: 字段集 != 6 字段契约"
        assert meta["display_name_zh"] == f"中文指标名_{mid}"
        assert meta["unit_zh"] == "%"
        assert meta["one_liner"] == f"{mid} 的一句话解释"
        assert meta["output_unit"] == "ratio"
        assert meta["direction_for_anxiety"] == "lower_is_anxious"
        assert meta["statistical_default"] == "groupwise_compare"


def test_metric_metadata_sidecar_dedup_takes_first() -> None:
    """去重取首个：同一 id 多行时，元数据来自 subject_index=0 那条。"""
    pm = _make_pm(n_subjects=3, n_metrics=1)
    # 把第 2 个 subject（index 1）的 display_name_zh 改成不同值，验证取首个
    pm.metrics[1].display_name_zh = "SHOULD_NOT_APPEAR"
    d = metric_metadata_to_dict(pm)
    assert d["metrics"]["metric_0"]["display_name_zh"] == "中文指标名_metric_0"


def test_metric_metadata_sidecar_single_subject() -> None:
    """单 subject 场景：n×1 条 → n 条，去重不丢。"""
    pm = _make_pm(n_subjects=1, n_metrics=3)
    d = metric_metadata_to_dict(pm)
    assert len(d["metrics"]) == 3


@pytest.mark.parametrize("paradigm", sorted(_MINIMAL_COLUMNS))
def test_metric_metadata_reflects_resolved_metrics_not_catalog_full(paradigm: str, tmp_path: Path) -> None:
    """旁路反映 plan 实际集合（resolve 出来的 metric），非 catalog 全集。

    构造列缺失场景：只给一个会让部分 metric skip 的列子集无法稳定构造（catalog 列检查是全或无），
    故改用「resolve 后实际产出的 metric id 集合」做对照——旁路的 key 集合必须 == resolve 出的
    PlanMetrics.metrics 的 id 去重集合，不多不少。这坐实「反映 plan 实际集合」语义。
    """
    raw_file = str(tmp_path / "dummy.txt")
    Path(raw_file).write_text("placeholder", encoding="utf-8")
    pm = resolve_metrics(
        paradigm=paradigm,
        columns=_MINIMAL_COLUMNS[paradigm],
        raw_files=[raw_file],
        workspace_dir=str(tmp_path),
    )
    assert pm.metrics, f"{paradigm}: resolve 未产出 metric"

    d = metric_metadata_to_dict(pm)

    resolved_ids = {m.id for m in pm.metrics}
    assert set(d["metrics"].keys()) == resolved_ids, (
        f"{paradigm}: 旁路 metric id 集合 != resolve 实际集合（旁路应反映 plan，非 catalog 全集）"
    )
    # 字段与 PlanMetric 同源
    for m in pm.metrics:
        meta = d["metrics"][m.id]
        assert meta["display_name_zh"] == m.display_name_zh
        assert meta["unit_zh"] == m.unit_zh
        assert meta["one_liner"] == m.one_liner
        assert meta["output_unit"] == m.output_unit
        assert meta["direction_for_anxiety"] == m.direction_for_anxiety
        assert meta["statistical_default"] == m.statistical_default
