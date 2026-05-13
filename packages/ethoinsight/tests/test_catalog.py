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
