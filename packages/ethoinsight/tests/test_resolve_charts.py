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
    """Single subject still has EPM catalog charts available now (open_arm_time_ratio_bar
    + zone_entry_distribution, registered 2026-05-20 for handoff #2). Fallback is only
    used when paradigm catalog yields 0 charts."""
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path),
        user_intent="再画几个图",
        total_subjects=1, n_per_group=1, n_groups=1,
    )
    catalog_ids = {c.id for c in pc.charts}
    assert "open_arm_time_ratio_bar" in catalog_ids
    assert "zone_entry_distribution" in catalog_ids
    # n_per_group=1 excludes the group-only box_open_arm
    assert "box_open_arm" not in catalog_ids
    # catalog charts present → fallback list stays empty
    assert pc.charts_fallback_available == []
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
    assert pc.schema_version == "1.1"


# ============================================================================
# requires_columns 过滤（A 修复）—— chart 引用的列不在数据中则跳过、记入 skipped
# 真实证据见 catalog/fst.yaml activity_intensity 在无 velocity 数据上误出图问题
# ============================================================================


# FST 真实采集到的列（来自 newdemodata/强迫游泳_大鼠 dump_headers）：mobility-state 系列、无 velocity / distance
FST_COLUMNS_SAMPLE = [
    "trial_time", "recording_time", "x_center", "y_center", "body_area",
    "area_change", "elongation", "mobility_continuous",
    "mobility_state_highly_mobile", "mobility_state_mobile", "mobility_state_immobile",
]


def test_chart_filtered_when_required_column_missing(tmp_path):
    """FST 数据无 velocity 列时，activity_intensity 不应进 charts，应进 skipped。

    Reason: catalog/fst.yaml.activity_intensity 声明 requires_columns: [velocity]，
    而 FST 模板真实不导出 velocity（只有 mobility-state）。
    之前 resolve_charts 完全忽略 requires_columns，导致脚本跑起来后画"velocity column missing"占位图。
    """
    pc = resolve_charts(
        paradigm="fst", columns=FST_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=2, n_per_group=1, n_groups=2,
    )
    chart_ids = {c.id for c in pc.charts}
    skipped_ids = {s.id for s in pc.skipped}
    assert "activity_intensity" not in chart_ids, (
        f"activity_intensity should be filtered when velocity column is missing. "
        f"Got charts={chart_ids}, skipped={skipped_ids}"
    )
    assert "activity_intensity" in skipped_ids
    # skipped 记录应说明 reason + missing pattern，便于 chart-maker / lead 透明可见
    skipped_record = next(s for s in pc.skipped if s.id == "activity_intensity")
    assert skipped_record.reason == "columns.missing"
    assert "velocity" in skipped_record.detail


def test_chart_kept_when_all_required_columns_present(tmp_path):
    """EPM 数据带 in_zone_open_arms_center → box_open_arm/open_arm_time_ratio_bar 保留。"""
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
    )
    chart_ids = {c.id for c in pc.charts}
    assert "box_open_arm" in chart_ids
    assert "open_arm_time_ratio_bar" in chart_ids


def test_struggle_distribution_kept_on_mobility_state_columns(tmp_path):
    """FST struggle_distribution 依赖 mobility_state*；FST 数据有 mobility_state_immobile 应通过 glob 匹配。"""
    pc = resolve_charts(
        paradigm="fst", columns=FST_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=2, n_per_group=1, n_groups=2,
    )
    chart_ids = {c.id for c in pc.charts}
    assert "struggle_distribution" in chart_ids


def test_oft_charts_match_single_in_zone_column(tmp_path):
    """OFT 数据只有单一 in_zone 列（无 in_zone_center 后缀）；catalog 用 in_zone* 通配应能匹配。"""
    oft_cols = [
        "trial_time", "recording_time", "x_center", "y_center",
        "x_nose", "y_nose", "distance_moved", "velocity", "in_zone", "distance_to_wall",
    ]
    pc = resolve_charts(
        paradigm="oft", columns=oft_cols, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
    )
    chart_ids = {c.id for c in pc.charts}
    assert "box_center" in chart_ids
    assert "center_time_ratio_bar" in chart_ids


def test_trajectory_filtered_when_xy_missing(tmp_path):
    """虚构一种缺 x_center/y_center 的数据 → epm trajectory/heatmap 应跳过。"""
    # 注意：EPM_COLUMNS_SAMPLE 顶部用了"X center"非 normalized 形式（既有测试如此）；
    # 但 normalize 后实际列名是 x_center —— catalog 用 normalized。这里直接用 normalized 列名构造。
    bad_cols_normalized = ["trial_time", "recording_time", "y_center"]
    pc = resolve_charts(
        paradigm="epm", columns=bad_cols_normalized, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=2, n_per_group=1, n_groups=2,
    )
    chart_ids = {c.id for c in pc.charts}
    skipped_ids = {s.id for s in pc.skipped}
    assert "trajectory" not in chart_ids
    assert "heatmap" not in chart_ids
    assert "trajectory" in skipped_ids
    assert "heatmap" in skipped_ids
