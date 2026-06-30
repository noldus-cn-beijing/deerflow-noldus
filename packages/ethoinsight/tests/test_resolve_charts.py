"""W4: resolve_charts 函数验收。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import (
    resolve_charts,
    plan_charts_to_dict,
    ResolveError,
)


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


# ============================================================================
# groups 透传 → aggregate plot 拿到 --groups（Spec 3 / P3 配套修复）
# 病根：``_build_groups_payload`` 只认 {arena_key: group} 短键的子串启发式
# (``arena_key in Path(p).name``)，对 ``prep_metric_plan`` 写的 SSOT
# {full_path: group} 形态匹配失败 → box_open_arm 静默丢 --groups（红线一静默降级）。
# 修法：先按完整路径精确匹配（与 scripts._cli.read_groups_json 同语义），
# 短键子串匹配降级为 fallback。
# ============================================================================


def _box_open_arm_groups_arg(pc):
    """Return the --groups value materialised for box_open_arm, or None."""
    for c in pc.charts:
        if c.id == "box_open_arm" and "--groups" in c.args:
            return c.args[c.args.index("--groups") + 1]
    return None


def test_groups_ssot_fullpath_keys_materialise_into_box(tmp_path):
    """SSOT {full_path: group}（prep_metric_plan 落盘形态）→ box_open_arm 拿到 --groups
    且分组按文件精确映射。修复前必红：旧子串启发式无法匹配完整路径键。"""
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    groups = {p: ("control" if i < 3 else "treatment") for i, p in enumerate(raw_files)}
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups=groups,
    )
    groups_arg = _box_open_arm_groups_arg(pc)
    assert groups_arg is not None, (
        "box_open_arm 必须拿到 --groups（SSOT 完整路径键应精确匹配）；"
        f"args={[c.args for c in pc.charts if c.id == 'box_open_arm']}"
    )
    materialised = json.loads((tmp_path / Path(groups_arg).name).read_text(encoding="utf-8"))
    assert set(materialised) == {"control", "treatment"}
    assert set(materialised["control"]) == set(raw_files[:3])
    assert set(materialised["treatment"]) == set(raw_files[3:])


def test_groups_legacy_arena_key_substring_still_works(tmp_path):
    """向后兼容：旧 {arena_key: group} 短键（子串匹配文件名）仍生效。"""
    raw_files = [
        "/mnt/user-data/uploads/Trial-Arena 1.txt",
        "/mnt/user-data/uploads/Trial-Arena 2.txt",
        "/mnt/user-data/uploads/Trial-Arena 3.txt",
        "/mnt/user-data/uploads/Trial-Arena 4.txt",
        "/mnt/user-data/uploads/Trial-Arena 5.txt",
        "/mnt/user-data/uploads/Trial-Arena 6.txt",
    ]
    groups = {
        "Arena 1": "control", "Arena 2": "control", "Arena 3": "control",
        "Arena 4": "treatment", "Arena 5": "treatment", "Arena 6": "treatment",
    }
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups=groups,
    )
    groups_arg = _box_open_arm_groups_arg(pc)
    assert groups_arg is not None, "legacy arena-key 短键应仍通过子串 fallback 匹配"
    materialised = json.loads((tmp_path / Path(groups_arg).name).read_text(encoding="utf-8"))
    assert set(materialised) == {"control", "treatment"}
    assert len(materialised["control"]) == 3
    assert len(materialised["treatment"]) == 3


def test_groups_finalshape_passthrough(tmp_path):
    """已是 {group: [paths]} 终态形态 → 直通。"""
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    groups = {
        "control": raw_files[:3],
        "treatment": raw_files[3:],
    }
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups=groups,
    )
    groups_arg = _box_open_arm_groups_arg(pc)
    assert groups_arg is not None
    materialised = json.loads((tmp_path / Path(groups_arg).name).read_text(encoding="utf-8"))
    assert set(materialised["control"]) == set(raw_files[:3])
    assert set(materialised["treatment"]) == set(raw_files[3:])


# ============================================================================
# source_filename（spec 2026-06-29-chart-display-name-source-filename）
# per_subject 图每张带来源 raw data 文件 basename，使多文件下 N 张同类图可区分。
# aggregate 图跨所有文件，source_filename 留空。
# ============================================================================


def _per_subject_charts(pc):
    """Return [(chart, source_filename)] for charts expanded per-subject (subject_index>=0
    且 output_mode == 'per_subject')。trajectory/heatmap 是典型 per_subject 图。"""
    return [(c, c.source_filename) for c in pc.charts if c.output_mode == "per_subject"]


def test_per_subject_chart_carries_source_filename_basename(tmp_path):
    """per_subject 图的 source_filename 必须是对应 raw_file 的 basename（保留原始文件名含多空格）。

    修复前必红：PlanChart 无 source_filename 字段。28 张 heatmap 会全显示成 sN/chart_id，
    用户无法分辨哪张对应哪个 trial。
    """
    raw_files = [
        "/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial     1.xlsx",
        "/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial    15.xlsx",
    ]
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=2, n_per_group=1, n_groups=2,
    )
    per_subject = _per_subject_charts(pc)
    assert per_subject, "EPM 应产出 per_subject 图（trajectory/heatmap）"
    # 每个 per_subject 图必须带非空 source_filename
    for _chart, src in per_subject:
        assert src, f"per_subject 图 source_filename 不应为空：{[c.id for c in pc.charts]}"
    # subject_index 0/1 必须分别对到第 0/1 个 raw_file 的 basename
    by_idx = {c.subject_index: c.source_filename for c in pc.charts if c.output_mode == "per_subject"}
    assert by_idx.get(0) == "Raw data-EPM-Xuhui-Trial     1.xlsx"
    assert by_idx.get(1) == "Raw data-EPM-Xuhui-Trial    15.xlsx"


def test_aggregate_chart_source_filename_empty(tmp_path):
    """aggregate 图（output_mode=='aggregate'，跨所有文件）source_filename 必须留空。

    box_open_arm 是典型 aggregate 图（组间对比，1 张/全文件），强加单一来源名会误导。
    """
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups={"control": raw_files[:3], "treatment": raw_files[3:]},
    )
    aggregate = [c for c in pc.charts if c.output_mode == "aggregate"]
    assert aggregate, "EPM 多组数据应产出 aggregate 图（box/bar）"
    for c in aggregate:
        assert c.source_filename == "", f"aggregate 图 {c.id} source_filename 应留空，got {c.source_filename!r}"


def test_source_filename_serialized_into_plan_charts_dict(tmp_path):
    """plan_charts_to_dict 必须把 source_filename 带进 charts[] 每条目（后端 list_chart_artifacts 读它填 ArtifactMeta）。

    修复前必红：序列化推导无 source_filename key。
    """
    raw_files = [
        "/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial     1.xlsx",
        "/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial    15.xlsx",
    ]
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=2, n_per_group=1, n_groups=2,
    )
    d = plan_charts_to_dict(pc)
    charts = d["charts"]
    assert charts, "plan_charts 应有条目"
    per_subject_entries = [e for e in charts if e.get("output_mode") == "per_subject"]
    assert per_subject_entries, "应含 per_subject 条目"
    # 每条 per_subject 条目必须有 source_filename key 且对到 basename
    by_idx = {e["subject_index"]: e.get("source_filename") for e in per_subject_entries}
    assert "source_filename" in per_subject_entries[0], "序列化漏带 source_filename key"
    assert by_idx[0] == "Raw data-EPM-Xuhui-Trial     1.xlsx"
    assert by_idx[1] == "Raw data-EPM-Xuhui-Trial    15.xlsx"
