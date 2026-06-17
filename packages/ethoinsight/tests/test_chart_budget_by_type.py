"""P5 (spec 2026-06-17-chart-budget-by-type): chart 预算按图类型而非数量。

旧预算是"数量上限"（fallback-decision-tree「最多 4 plot」），按数组顺序取前 N：
per_subject 个体图（trajectory/heatmap，N×类 张）排在前面吃光名额，
aggregate 组间对比图（box/bar/rose，最有分析价值）排不上。

修法（两层）：
  1. 数据层：PlanChart 透传 ``output_mode`` + 确定性纯函数
     :func:`select_charts_by_priority`（aggregate 全画优先，per_subject 用剩余预算
     按代表性子集取，subject_index=0 优先 = 每组首个 subject 各一张）。
  2. CLI: ``--chart-budget`` 把截断结果写进 ``charts[]``，被挤掉的 per_subject 图
     写进 ``charts_budget_remaining[]`` + notes 留降级指纹（红线一）。

spec 第 3 节三个验收场景：
  - test_aggregate_charts_prioritized
  - test_per_subject_budget_logged
  - test_no_aggregate_still_picks_per_subject
"""
from __future__ import annotations

import json
from pathlib import Path

from ethoinsight.catalog.resolve import (
    plan_charts_to_dict,
    resolve_charts,
    select_charts_by_priority,
)
from ethoinsight.catalog.schema import PlanChart, PlanCharts, PlanInputs

EPM_COLUMNS_SAMPLE = [
    "Trial time", "X center", "Y center",
    "in_zone_open_arms_center", "in_zone_closed_arms_center",
]


def _agg(cid: str) -> PlanChart:
    return PlanChart(id=cid, script=f"s.{cid}", input=f"/w/i_{cid}.json",
                     output=f"/o/p_{cid}.png", subject_index=0,
                     display_name_zh=cid, confidence="must_have",
                     args=["--inputs", f"/w/i_{cid}.json", "--output", f"/o/p_{cid}.png"],
                     output_mode="aggregate")


def _per(cid: str, idx: int) -> PlanChart:
    suffix = f"_s{idx}" if idx else ""
    return PlanChart(id=cid, script=f"s.{cid}", input=f"/w/i_{cid}{suffix}.json",
                     output=f"/o/p_{cid}{suffix}.png", subject_index=idx,
                     display_name_zh=cid, confidence="must_have",
                     args=["--inputs", f"/w/i_{cid}{suffix}.json", "--output", f"/o/p_{cid}{suffix}.png"],
                     output_mode="per_subject")


# ============================================================================
# spec 场景 1: aggregate 全部入选，per_subject 用剩余名额
# ============================================================================


def test_aggregate_charts_prioritized():
    """5 aggregate + 56 per_subject（7 类 × 8 subject），预算 N=6 →
    5 张 aggregate 全部入选，per_subject 仅剩 1 个名额（subject_index=0 优先）。

    关键：per_subject 排在输入数组**前面**（dogfood 真实顺序——plan_charts 里 28 trajectory +
    28 heatmap 先于 box/bar），断言 aggregate 仍全部入选——证明是「按类型优先」而非
    「按数组顺序取前 N」（naive first-N 会在此输入下漏掉所有 aggregate）。"""
    agg = [_agg(f"agg{i}") for i in range(5)]
    per = [_per(f"per{j}", idx) for j in range(7) for idx in range(8)]  # 56
    charts = per + agg  # per_subject 在前：naive 取前 6 会全是 per_subject，漏掉 aggregate

    selected, remaining = select_charts_by_priority(charts, budget=6)

    # aggregate 全部入选（核心：组间对比不受预算挤占，即使排在数组末尾）
    selected_agg_ids = {c.id for c in selected if c.output_mode == "aggregate"}
    assert selected_agg_ids == {f"agg{i}" for i in range(5)}

    # per_subject 只拿到 1 个名额（6 - 5），且是 subject_index=0 优先（每组首个 subject）
    selected_per = [c for c in selected if c.output_mode != "aggregate"]
    assert len(selected_per) == 1
    assert selected_per[0].subject_index == 0

    # 被挤掉的 per_subject 进 remaining（红线一留指纹）
    assert len(remaining) == 56 - 1
    assert all(c.output_mode != "aggregate" for c in remaining)

    # selected 顺序：aggregate 在前（chart-maker 按数组顺序执行时 aggregate 先跑）
    assert selected[0].output_mode == "aggregate"


def test_aggregate_exceeding_budget_still_all_drawn():
    """aggregate 数量 ≥ 预算时仍全画 aggregate（组间对比是分析核心，不受 floor 挤占）。"""
    agg = [_agg(f"agg{i}") for i in range(5)]
    per = [_per("per", 0), _per("per", 1)]
    selected, remaining = select_charts_by_priority(agg + per, budget=3)

    assert {c.id for c in selected} == {f"agg{i}" for i in range(5)}
    assert selected == agg  # aggregate 全留，per_subject 名额 = max(0, 3-5) = 0
    assert remaining == [_per("per", 0), _per("per", 1)]  # 按 (subject_index, id) 排序


# ============================================================================
# spec 场景 2: per_subject 被预算截断 → remaining 非空（降级指纹）
# ============================================================================


def test_per_subject_budget_logged():
    """per_subject 被预算截断 → remaining 非空，且是确定性结果。

    代表性子集语义：subject_index=0 的全部 per_subject 优先（每组首个 subject 各一张），
    再 subject_index=1，轮转直到预算耗尽。
    """
    # 2 类 per_subject × 3 subject = 6 张；无 aggregate
    per = [_per("traj", idx) for idx in range(3)] + [_per("heat", idx) for idx in range(3)]
    selected, remaining = select_charts_by_priority(per, budget=4)

    # 预算 4 全给 per_subject：subject_index=0,1 的全部图（2 类 × 2 = 4）
    selected_keys = sorted((c.id, c.subject_index) for c in selected)
    assert selected_keys == [("heat", 0), ("heat", 1), ("traj", 0), ("traj", 1)]
    assert len(remaining) == 2
    remaining_keys = sorted((c.id, c.subject_index) for c in remaining)
    assert remaining_keys == [("heat", 2), ("traj", 2)]


def test_budget_none_means_no_limit():
    """budget=None（不限制）→ 全画，remaining=[]。"""
    charts = [_agg("agg")] + [_per("per", i) for i in range(10)]
    selected, remaining = select_charts_by_priority(charts, budget=None)
    assert selected == charts
    assert remaining == []


def test_selection_is_deterministic():
    """相同输入恒定相同输出（无随机性）—— chart-maker 按数组顺序执行可复现。"""
    per = [_per("b", idx) for idx in range(5)] + [_per("a", idx) for idx in range(5)]
    s1, r1 = select_charts_by_priority(per, budget=4)
    s2, r2 = select_charts_by_priority(list(reversed(per)), budget=4)  # 输入顺序打乱
    assert [(c.id, c.subject_index) for c in s1] == [(c.id, c.subject_index) for c in s2]
    assert [(c.id, c.subject_index) for c in r1] == [(c.id, c.subject_index) for c in r2]


# ============================================================================
# spec 场景 3: plan 无 aggregate（纯坐标数据）→ 退回按代表性取 per_subject（不空手）
# ============================================================================


def test_no_aggregate_still_picks_per_subject():
    """plan 无 aggregate → 预算全给 per_subject 代表性子集，绝不空手。

    真实场景：某些范式数据只有坐标列，catalog 只产出 trajectory/heatmap 等个体图，
    没有 box/bar 组间对比图。此时 chart-maker 仍应画出代表性个体图，而非因"没
    aggregate"就一张不出。
    """
    per = [_per("traj", idx) for idx in range(8)]  # 8 张个体图，无 aggregate
    selected, remaining = select_charts_by_priority(per, budget=3)

    assert len(selected) == 3
    # subject_index=0,1,2（每组首个 subject 各一张）
    assert [c.subject_index for c in selected] == [0, 1, 2]
    assert len(remaining) == 5


def test_no_aggregate_budget_none_draws_all():
    """无 aggregate + 不限预算 → 全画。"""
    per = [_per("traj", idx) for idx in range(5)]
    selected, remaining = select_charts_by_priority(per, budget=None)
    assert len(selected) == 5
    assert remaining == []


# ============================================================================
# output_mode 透传：PlanChart.output_mode 从 ChartEntry 正确落到 resolve 产物
# ============================================================================


def test_output_mode_threaded_through_resolve(tmp_path):
    """resolve_charts 产出的 PlanChart.output_mode 与 catalog ChartEntry 一致。

    EPM box_open_arm 标 output_mode: aggregate → 产物里 box_open_arm 是 aggregate；
    open_arm_time_ratio_bar / zone_entry_distribution 默认 per_subject → 产物里是 per_subject。
    (trajectory/heatmap 被 skip：EPM_COLUMNS_SAMPLE 含 "X center" 未 normalize 成 x_center，
    与本测试无关——此处只验 output_mode 透传，用实际进入 charts 的图。)
    """
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    groups = {p: ("control" if i < 3 else "treatment") for i, p in enumerate(raw_files)}
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups=groups,
    )
    by_id_mode = {}
    for c in pc.charts:
        by_id_mode.setdefault(c.id, c.output_mode)
    assert by_id_mode.get("box_open_arm") == "aggregate"
    assert by_id_mode.get("open_arm_time_ratio_bar") == "per_subject"
    assert by_id_mode.get("zone_entry_distribution") == "per_subject"


def test_output_mode_serialized_in_plan_dict():
    """plan_charts_to_dict 把 output_mode 写进 JSON（chart-maker 据此区分图类）。"""
    pc = PlanCharts(
        paradigm="epm", ev19_template=None, generated_at="t",
        inputs=PlanInputs(raw_files=[], groups_file=None, columns_file=None),
        charts=[_agg("box"), _per("traj", 0)],
        charts_fallback_available=[],
        skipped=[],
        user_intent=None,
        notes=[],
    )
    d = plan_charts_to_dict(pc)
    modes = {c["id"]: c["output_mode"] for c in d["charts"]}
    assert modes == {"box": "aggregate", "traj": "per_subject"}
    # charts_budget_remaining 也带 output_mode
    d2 = plan_charts_to_dict(PlanCharts(
        paradigm="epm", ev19_template=None, generated_at="t",
        inputs=PlanInputs(raw_files=[], groups_file=None, columns_file=None),
        charts=[], charts_fallback_available=[],
        charts_budget_remaining=[_per("traj", 5)],
        skipped=[], user_intent=None, notes=[],
    ))
    assert d2["charts_budget_remaining"][0]["output_mode"] == "per_subject"


# ============================================================================
# 真实 EPM 端到端：P5 选图让 box_open_arm（aggregate）优先于个体图
# （复现 spec 触发场景：旧规则下 box_open_arm 排在数组末尾被前 4 个体图挤掉）
# ============================================================================


def test_epm_budget_prioritizes_aggregate_box_over_individual(tmp_path):
    """6 subject EPM（box_open_arm 进入）：budget=4 时 box_open_arm 必入选，
    旧"取前 4"规则下它排在数组末尾会被个体图挤掉。"""
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    groups = {p: ("control" if i < 3 else "treatment") for i, p in enumerate(raw_files)}
    pc = resolve_charts(
        paradigm="epm", columns=EPM_COLUMNS_SAMPLE, raw_files=raw_files,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        groups=groups,
    )
    # 前提：box_open_arm 确实在 catalog 产出里（P3 后 groups 自读生效）
    assert any(c.id == "box_open_arm" and c.output_mode == "aggregate" for c in pc.charts)

    selected, remaining = select_charts_by_priority(pc.charts, budget=4)
    # box_open_arm（aggregate）必须入选并排第一
    assert selected[0].id == "box_open_arm"
    assert selected[0].output_mode == "aggregate"
    # 剩余 3 个名额给 per_subject 代表子集（subject_index 升序轮转 = 每组首个 subject 各一张优先）
    per_selected = [c for c in selected if c.output_mode != "aggregate"]
    assert len(per_selected) == 3
    # 优先取 subject_index=0（2 类 per_subject 各一张），第 3 个名额取 subject_index=1
    subject_indices = sorted(c.subject_index for c in per_selected)
    assert subject_indices[0] == 0  # 首选每组首个 subject
    assert subject_indices[-1] <= 1  # 预算足够覆盖 subject_index=0/1，不跳到更高
    # 被挤掉的个体图留指纹
    assert len(remaining) > 0
    assert all(c.output_mode != "aggregate" for c in remaining)


# ============================================================================
# CLI 端到端：--chart-budget 截断 + charts_budget_remaining + notes 降级指纹
# ============================================================================


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_cli_chart_budget_truncates_and_logs(tmp_path):
    """--chart-budget N 截断 per_subject → charts_budget_remaining 非空 + notes 记降级。

    走真实 CLI（python -m ethoinsight.catalog.resolve --mode charts --chart-budget）。
    """
    from ethoinsight.catalog.cli import main as cli_main

    workspace = tmp_path / "ws"
    workspace.mkdir()
    raw_files = [f"/mnt/user-data/uploads/arena{i}.txt" for i in range(1, 7)]
    groups = {p: ("control" if i < 3 else "treatment") for i, p in enumerate(raw_files)}
    columns_file = tmp_path / "columns.json"
    raw_files_file = tmp_path / "raw.json"
    groups_file = tmp_path / "groups.json"
    _write(columns_file, {"columns": EPM_COLUMNS_SAMPLE})
    _write(raw_files_file, raw_files)
    _write(groups_file, groups)
    out = tmp_path / "plan_charts.json"

    rc = cli_main([
        "--mode", "charts", "--paradigm", "epm",
        "--columns-file", str(columns_file),
        "--raw-files-json", str(raw_files_file),
        "--groups-json", str(groups_file),
        "--workspace-dir", str(workspace),
        "--total-subjects", "6", "--n-per-group", "3", "--n-groups", "2",
        "--chart-budget", "4",
        "--output", str(out),
    ])
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))

    # charts[] 里 aggregate（box_open_arm）必入选
    chart_ids = [c["id"] for c in plan["charts"]]
    assert "box_open_arm" in chart_ids
    assert len(plan["charts"]) == 4

    # 被截断的 per_subject 进 charts_budget_remaining
    assert len(plan["charts_budget_remaining"]) > 0
    rem = plan["charts_budget_remaining"]
    assert all(c["output_mode"] != "aggregate" for c in rem)

    # notes 留降级指纹（红线一：挤掉产出要留痕）
    budget_notes = [n for n in plan["notes"] if "Chart budget" in n]
    assert len(budget_notes) == 1, f"应有一条预算降级 note，实际 notes={plan['notes']}"
    assert "per_subject" in budget_notes[0]


def test_cli_no_budget_draws_all(tmp_path):
    """省略 --chart-budget → 全画，charts_budget_remaining=[]。"""
    from ethoinsight.catalog.cli import main as cli_main

    workspace = tmp_path / "ws"
    workspace.mkdir()
    columns_file = tmp_path / "columns.json"
    raw_files_file = tmp_path / "raw.json"
    _write(columns_file, {"columns": EPM_COLUMNS_SAMPLE})
    _write(raw_files_file, ["/mnt/user-data/uploads/arena1.txt"])
    out = tmp_path / "plan_charts.json"

    rc = cli_main([
        "--mode", "charts", "--paradigm", "epm",
        "--columns-file", str(columns_file),
        "--raw-files-json", str(raw_files_file),
        "--workspace-dir", str(workspace),
        "--total-subjects", "1",
        "--output", str(out),
    ])
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["charts_budget_remaining"] == []
