"""PR-1: catalog schema 1.0→1.1 升级验收。

覆盖 §2.5 要求的 8 个新增单测:
  - display_name_zh 必填校验
  - accepts_paradigm 默认 false
  - PlanChart.output → outputs/
  - accepts_paradigm → --paradigm args 注入
  - 序列化新字段
  - schema_version == "1.1"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ethoinsight.catalog.loader import CatalogError, load_catalog, load_common_catalog
from ethoinsight.catalog.resolve import (
    SCHEMA_VERSION,
    plan_charts_to_dict,
    plan_metrics_to_dict,
    resolve_charts,
    resolve_metrics,
)
from ethoinsight.catalog.schema import Plan, PlanCharts, PlanInputs, PlanMetrics


EPM_COLUMNS_SAMPLE = [
    "Trial time", "X center", "Y center",
    "in_zone_open_arms_center", "in_zone_closed_arms_center",
]
RAW_FILES_SAMPLE = ["/tmp/raw1.txt"]


# ============================================================================
# 1. display_name_zh 必填
# ============================================================================


def test_chart_entry_display_name_zh_required(tmp_path: Path):
    """YAML 缺少 display_name_zh → CatalogError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "paradigm: bad\n"
        "ev19_templates: []\n"
        "default_metrics: []\n"
        "optional_metrics: []\n"
        "charts:\n"
        "  - id: test_chart\n"
        "    script: ethoinsight.scripts._common.plot_trajectory\n"
        "    when: always\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="display_name_zh"):
        load_catalog("bad", catalog_dir=tmp_path)


def test_chart_entry_display_name_zh_required_in_common(tmp_path: Path):
    """_common.yaml 的 common_charts 段也要求 display_name_zh 必填."""
    bad_yaml = tmp_path / "_common.yaml"
    bad_yaml.write_text(
        "common_charts:\n"
        "  - id: test_chart\n"
        "    script: ethoinsight.scripts._common.plot_trajectory\n"
        "    when: always\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="display_name_zh"):
        load_common_catalog(catalog_dir=tmp_path)


# ============================================================================
# 2. accepts_paradigm 默认 false
# ============================================================================


def test_chart_entry_accepts_paradigm_default_false():
    """不写 accepts_paradigm 时 ChartEntry.accepts_paradigm == False."""
    # trajectory_plot 在 _common.yaml 未设 accepts_paradigm → 默认 false
    common = load_common_catalog()
    traj = next(c for c in common.common_charts if c.id == "trajectory_plot")
    assert traj.accepts_paradigm is False
    assert traj.display_name_zh == "轨迹图"


def test_chart_entry_accepts_paradigm_true_when_set():
    """timeseries_plot 设 accepts_paradigm: true → True."""
    common = load_common_catalog()
    ts = next(c for c in common.common_charts if c.id == "timeseries_plot")
    assert ts.accepts_paradigm is True
    assert ts.display_name_zh == "时间序列图"


# ============================================================================
# 3. PlanChart.output → outputs/
# ============================================================================


def test_plan_chart_output_in_outputs_dir(tmp_path: Path):
    """resolve_charts 生成的 PlanChart.output 以 /mnt/user-data/outputs/ 开头."""
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=1,
    )
    for chart in pc.charts:
        assert chart.output.startswith("/mnt/user-data/outputs/"), (
            f"chart {chart.id} output 不在 outputs/: {chart.output}"
        )


# ============================================================================
# 4 & 5. accepts_paradigm → args 注入
# ============================================================================


def test_plan_chart_args_includes_paradigm_when_accepts_paradigm():
    """accepts_paradigm=true 的 chart → PlanChart.args 含 --paradigm."""
    # 需要一组数据触发 fallback（让 timeseries_plot 可用）
    # 用 shoaling — 该范式 charts: [] 会 fallback 到 common
    pc = resolve_charts(
        paradigm="shoaling",
        columns=["x_center", "y_center"],
        raw_files=RAW_FILES_SAMPLE,
        workspace_dir="/tmp",
        total_subjects=1,
    )
    ts_charts = [c for c in pc.charts_fallback_available if c.id == "timeseries_plot"]
    assert len(ts_charts) >= 1, "timeseries_plot should be in fallback"
    for c in ts_charts:
        assert "--paradigm" in c.args, f"timeseries_plot args 缺 --paradigm: {c.args}"
        assert c.args == [
            "--input", c.input,
            "--output", c.output,
            "--paradigm", "shoaling",
        ]


def test_plan_chart_args_excludes_paradigm_otherwise():
    """accepts_paradigm=false 的 chart → PlanChart.args 不含 --paradigm."""
    pc = resolve_charts(
        paradigm="shoaling",
        columns=["x_center", "y_center"],
        raw_files=RAW_FILES_SAMPLE,
        workspace_dir="/tmp",
        total_subjects=1,
    )
    traj_charts = [c for c in pc.charts_fallback_available if c.id == "trajectory_plot"]
    assert len(traj_charts) >= 1, "trajectory_plot should be in fallback"
    for c in traj_charts:
        assert "--paradigm" not in c.args, (
            f"trajectory_plot args 不应含 --paradigm: {c.args}"
        )
        # verify base args are still present
        assert c.args[:4] == ["--input", c.input, "--output", c.output]


def test_plan_chart_args_excludes_paradigm_for_paradigm_specific():
    """范式特有 charts（如 EPM plot_box_open_arm）不接 --paradigm."""
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir="/tmp",
        total_subjects=6, n_per_group=3, n_groups=2,
    )
    for c in pc.charts:
        assert "--paradigm" not in c.args, (
            f"范式特有 chart {c.id} args 不应含 --paradigm: {c.args}"
        )


# ============================================================================
# 6 & 7. 序列化新字段
# ============================================================================


def test_plan_metrics_to_dict_includes_display_name_zh(tmp_path: Path):
    """序列化结果含 display_name_zh 中文标签."""
    physical_ws = str(tmp_path / "workspace")
    Path(physical_ws).mkdir(parents=True)
    pm = resolve_metrics(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE,
        raw_files=["/mnt/user-data/uploads/test.txt"],
        workspace_dir=physical_ws,
        virtual_workspace_dir="/mnt/user-data/workspace",
    )
    payload = plan_metrics_to_dict(pm)
    for m in payload["metrics"]:
        assert "display_name_zh" in m, f"metric {m['id']} 缺 display_name_zh"
        assert isinstance(m["display_name_zh"], str)
        assert m["display_name_zh"] != "", f"metric {m['id']} display_name_zh 为空"


def test_plan_charts_to_dict_includes_args_and_display_name_zh():
    """序列化 charts 含 args 数组 + display_name_zh."""
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=RAW_FILES_SAMPLE,
        workspace_dir="/tmp", total_subjects=6, n_per_group=3, n_groups=2,
    )
    payload = plan_charts_to_dict(pc)
    for c in payload["charts"]:
        assert "display_name_zh" in c, f"chart {c['id']} 缺 display_name_zh"
        assert isinstance(c["display_name_zh"], str)
        assert c["display_name_zh"] != "", f"chart {c['id']} display_name_zh 为空"
        assert "args" in c, f"chart {c['id']} 缺 args"
        assert isinstance(c["args"], list)
        assert len(c["args"]) >= 4  # at least --input X --output Y
    # fallback 也含
    for c in payload["charts_fallback_available"]:
        assert "display_name_zh" in c
        assert "args" in c


# ============================================================================
# 8. schema_version == "1.1"
# ============================================================================


def test_schema_version_is_1_1():
    """SCHEMA_VERSION 常量 + PlanMetrics / PlanCharts 默认值均为 1.1."""
    assert SCHEMA_VERSION == "1.1"
    assert PlanMetrics.__dataclass_fields__["schema_version"].default == "1.1"
    assert PlanCharts.__dataclass_fields__["schema_version"].default == "1.1"
