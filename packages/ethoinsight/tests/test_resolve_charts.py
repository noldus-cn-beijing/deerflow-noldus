"""W4: resolve_charts 函数验收。"""
from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import resolve_charts, ResolveError


EPM_COLUMNS_SAMPLE = [
    "Trial time", "X center", "Y center",
    "in_zone_open_arms_center", "in_zone_closed_arms_center",
]
RAW_FILES_SAMPLE = ["/tmp/raw1.txt"]


def test_single_subject_triggers_fallback(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path),
        user_intent="再画几个图",
        total_subjects=1, n_per_group=1, n_groups=1,
    )
    assert pc.charts == []
    fallback_ids = {c.id for c in pc.charts_fallback_available}
    assert "trajectory_plot" in fallback_ids
    assert "timeseries_plot" in fallback_ids
    assert pc.user_intent == "再画几个图"
    assert pc.paradigm == "epm"


def test_group_data_uses_catalog_charts_not_fallback(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path),
        user_intent=None,
        total_subjects=6, n_per_group=3, n_groups=2,
    )
    assert len(pc.charts) > 0
    assert pc.charts_fallback_available == []


def test_unknown_paradigm_raises_resolve_error(tmp_path):
    with pytest.raises(ResolveError) as exc:
        resolve_charts(
            paradigm="nonexistent_paradigm", columns=[], raw_files=[],
            workspace_dir=str(tmp_path), total_subjects=1,
        )
    assert exc.value.code == "unknown_paradigm"


def test_plan_charts_schema_version(tmp_path):
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=1,
    )
    assert pc.schema_version == "1.0"
