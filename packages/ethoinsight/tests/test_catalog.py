"""Tests for ethoinsight.catalog module."""

from __future__ import annotations

import pytest


def test_schema_classes_importable():
    from ethoinsight.catalog import schema
    assert hasattr(schema, "MetricEntry")
    assert hasattr(schema, "Catalog")
    assert hasattr(schema, "Plan")


def test_metric_entry_minimal_construction():
    from ethoinsight.catalog.schema import MetricEntry
    m = MetricEntry(
        id="open_arm_time_ratio",
        script="ethoinsight.scripts.epm.compute_open_arm_time_ratio",
        requires_columns=["in_zone_open_arms_*"],
        output_unit="ratio",
        display_name_zh="开放臂时间比例",
        unit_zh="比例",
        one_liner="动物在开放臂中停留时间占总时长的比例",
        direction_for_anxiety="lower_is_anxious",
        statistical_default="groupwise_compare",
    )
    assert m.id == "open_arm_time_ratio"
    assert m.direction_for_anxiety == "lower_is_anxious"


# ── load_catalog tests ──────────────────────────────────────────────────────


def test_load_catalog_returns_catalog_instance(tmp_path):
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.schema import Catalog

    yaml_content = """
paradigm: epm_test
ev19_templates:
  - Elevated Plus Maze XT190
default_metrics:
  - id: open_arm_time_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: [in_zone_open_arms_*]
    output_unit: ratio
    display_name_zh: 开放臂时间比例
    unit_zh: 比例
    one_liner: 动物在开放臂中停留时间占总时长的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.epm.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
"""
    yaml_path = tmp_path / "epm_test.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    cat = load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert isinstance(cat, Catalog)
    assert cat.paradigm == "epm_test"
    assert len(cat.default_metrics) == 1
    assert cat.default_metrics[0].id == "open_arm_time_ratio"
    assert cat.statistics_default is not None
    assert cat.statistics_default.id == "groupwise_compare"


def test_load_catalog_unknown_paradigm_raises(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    with pytest.raises(CatalogError) as exc:
        load_catalog("totally_made_up", catalog_dir=str(tmp_path))
    assert "totally_made_up" in str(exc.value)


def test_load_catalog_rejects_invalid_direction_enum(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    bad = """
paradigm: epm_test
ev19_templates: []
default_metrics:
  - id: foo
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: []
    output_unit: ratio
    display_name_zh: 测试
    unit_zh: 单位
    one_liner: 描述
    direction_for_anxiety: very_weirdly_anxious
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics: null
"""
    (tmp_path / "epm_test.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert "direction_for_anxiety" in str(exc.value)


def test_load_catalog_rejects_duplicate_metric_ids(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    bad = """
paradigm: epm_test
ev19_templates: []
default_metrics:
  - id: dup_id
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: []
    output_unit: ratio
    display_name_zh: 一
    unit_zh: u
    one_liner: a
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  - id: dup_id
    script: ethoinsight.scripts.epm.compute_open_arm_time
    requires_columns: []
    output_unit: seconds
    display_name_zh: 二
    unit_zh: u
    one_liner: b
    direction_for_anxiety: null
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics: null
"""
    (tmp_path / "epm_test.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert "dup_id" in str(exc.value)
