"""防回归 (2026-05-20 FST E2E): catalog resolve_metrics / resolve_charts 之前
内部 `_metric_to_plan` / `_chart_to_plan` 永远只用 raw_files[0],
导致用户上传多文件时除第一个外的 subject 全部被静默丢弃。

修复后:每个 MetricEntry / ChartEntry × 每个 raw_file = 一个 PlanMetric / PlanChart,
output 路径带 _s<idx> 后缀防覆盖,subject_index 字段标识 subject。
单文件场景仍保持 m_<id>.json 命名向后兼容。
"""

from pathlib import Path

from ethoinsight.catalog.resolve import (
    plan_charts_to_dict,
    plan_metrics_to_dict,
    resolve_charts,
    resolve_metrics,
)


VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"
VIRTUAL_UPLOADS = "/mnt/user-data/uploads"

EPM_MIN_COLUMNS = [
    "in_zone_open_arms_center",
    "in_zone_closed_arms_center",
]


def test_resolve_metrics_expands_per_subject(tmp_path: Path) -> None:
    """2 个 raw_files → 每个指标各出现 2 次,subject_index = 0/1。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [
        f"{VIRTUAL_UPLOADS}/arena1.txt",
        f"{VIRTUAL_UPLOADS}/arena2.txt",
    ]

    plan = resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
    )

    # 每个 unique metric id 出现 2 次(每个 subject 一次)
    from collections import Counter
    per_id_count = Counter(m.id for m in plan.metrics)
    for mid, cnt in per_id_count.items():
        assert cnt == 2, f"metric {mid} 出现 {cnt} 次,应为 2 (每 subject 一次)"

    # subject_index 覆盖 0 和 1
    indices = sorted({m.subject_index for m in plan.metrics})
    assert indices == [0, 1], f"subject_index 应为 [0, 1],得到 {indices}"

    # 每个 metric 的 input 对应它的 subject_index
    for m in plan.metrics:
        expected_input = raw_files[m.subject_index]
        assert m.input == expected_input, (
            f"metric {m.id}@subject_{m.subject_index} input={m.input} != {expected_input}"
        )

    # output 必须带 _s<idx> 后缀防止两个 subject 互相覆盖
    for m in plan.metrics:
        assert f"_s{m.subject_index}" in m.output, (
            f"metric {m.id}@subject_{m.subject_index} output 缺 _s 后缀: {m.output}"
        )


def test_resolve_metrics_single_file_no_suffix(tmp_path: Path) -> None:
    """单文件 output 仍是 m_<id>.json 无 _s 后缀,保持向后兼容。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [f"{VIRTUAL_UPLOADS}/single.txt"]

    plan = resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
    )

    for m in plan.metrics:
        assert "_s" not in m.output.rsplit("/", 1)[-1], (
            f"单文件 output 不应带 _s 后缀: {m.output}"
        )
        assert m.subject_index == 0


def test_plan_metrics_to_dict_includes_subject_index(tmp_path: Path) -> None:
    """plan_metrics_to_dict 序列化必须含 subject_index 字段。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [
        f"{VIRTUAL_UPLOADS}/arena1.txt",
        f"{VIRTUAL_UPLOADS}/arena2.txt",
    ]

    plan = resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
    )

    payload = plan_metrics_to_dict(plan)
    for m in payload["metrics"]:
        assert "subject_index" in m, f"metric {m['id']} 缺 subject_index"
        assert isinstance(m["subject_index"], int)


def test_resolve_charts_expands_per_subject(tmp_path: Path) -> None:
    """charts 也按 raw_files 展开 — 每个 chart × 每个 subject 一个 PlanChart。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [
        f"{VIRTUAL_UPLOADS}/arena1.txt",
        f"{VIRTUAL_UPLOADS}/arena2.txt",
    ]

    plan = resolve_charts(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        total_subjects=2,
        n_groups=1,
        n_per_group=2,
        user_intent=None,
    )

    # charts 与 fallback 都按 raw_files 展开(若有的话)
    all_charts = list(plan.charts) + list(plan.charts_fallback_available)
    if not all_charts:
        # 该范式可能无 chart 注册,跳过
        import pytest
        pytest.skip("epm catalog 此条件下无 chart 注册;chart 展开规则在其他范式上验证")

    indices = sorted({c.subject_index for c in all_charts})
    assert indices == [0, 1], f"chart subject_index 应为 [0, 1],得到 {indices}"

    payload = plan_charts_to_dict(plan)
    for c in payload["charts"] + payload["charts_fallback_available"]:
        assert "subject_index" in c, f"chart {c['id']} 缺 subject_index"
