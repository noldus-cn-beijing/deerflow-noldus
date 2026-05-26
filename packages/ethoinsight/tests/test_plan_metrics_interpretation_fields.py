"""W27: 验证 resolve_metrics 把 catalog MetricEntry 的判读 / 展示字段透传到 PlanMetric。

不实际跑 metric 脚本,只验证字段透传契约。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.catalog.loader import load_catalog
from ethoinsight.catalog.resolve import resolve_metrics

# 每个范式列举一组足以让 default_metrics 全部通过 columns 检查的列名。
# 来源:packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml 的 requires_columns 字段。
_MINIMAL_COLUMNS: dict[str, list[str]] = {
    "epm": ["in_zone_open_arms_subject", "in_zone_closed_arms_subject", "x_center", "y_center", "velocity"],
    "oft": ["in_zone_center_subject", "in_zone_periphery_subject", "x_center", "y_center", "velocity", "in_zone_subject", "distance_moved"],
    "fst": ["mobility_state", "mobility_state_highly_mobile", "velocity"],
    "ldb": ["in_zone_light_subject", "in_zone_dark_subject", "x_center", "y_center", "velocity"],
    "zero_maze": ["in_zone_open_subject", "in_zone_closed_subject", "x_center", "y_center", "velocity", "in_zone_subject", "distance_moved"],
}

_EXPECTED_NEW_FIELDS = {
    "unit_zh",
    "one_liner",
    "output_unit",
    "direction_for_anxiety",
    "statistical_default",
}


@pytest.mark.parametrize("paradigm", sorted(_MINIMAL_COLUMNS))
def test_resolve_metrics_transfers_interpretation_fields(paradigm: str, tmp_path: Path) -> None:
    """每个范式 resolve 后,每个 PlanMetric 必须含 catalog MetricEntry 的同源字段值。"""
    raw_file = str(tmp_path / "dummy.txt")
    Path(raw_file).write_text("placeholder", encoding="utf-8")

    pm = resolve_metrics(
        paradigm=paradigm,
        columns=_MINIMAL_COLUMNS[paradigm],
        raw_files=[raw_file],
        workspace_dir=str(tmp_path),
    )

    assert pm.metrics, f"{paradigm}: 没有 metric 输出,说明 _MINIMAL_COLUMNS 不全"

    # 取 catalog 原文做断言对照(确保字段值同源,不是凭空填的默认值)
    cat = load_catalog(paradigm)
    catalog_by_id = {m.id: m for m in cat.default_metrics}

    for plan_metric in pm.metrics:
        # 1. dataclass 必须有这 5 个 attribute(默认值就行,但下一步检查同源)
        for fld in _EXPECTED_NEW_FIELDS:
            assert hasattr(plan_metric, fld), f"{paradigm}/{plan_metric.id}: PlanMetric 缺 {fld}"

        # 2. 字段值必须等于 catalog MetricEntry 同名字段
        entry = catalog_by_id[plan_metric.id]
        assert plan_metric.unit_zh == entry.unit_zh
        assert plan_metric.one_liner == entry.one_liner
        assert plan_metric.output_unit == entry.output_unit
        assert plan_metric.direction_for_anxiety == entry.direction_for_anxiety
        assert plan_metric.statistical_default == entry.statistical_default
