"""spec 2026-06-22: chart --parameters-json 按 chart.requires_columns 裁剪 zone 参数。

根因：resolve_charts 给每张图注入的 --parameters-json 是**全范式 zone 参数并集**
（open_arm_zones + closed_arm_zones），不按该 chart 实际用到的 zone 概念裁剪。
底层 compute 函数签名严格的图（compute_open_arm_time_ratio 只收 open_arm_zones）
被强塞了它不认识的 closed_arm_zones → TypeError → chart-maker 靠手删参数重跑兜底。

治本：chart 的 requires_columns 里 in_zone_<concept>_* glob 就是该图用到的 zone 概念
的权威声明，按它裁剪 zone_overrides 到该图子集。覆盖：
1. open_arm_time_ratio_bar（只要 open）只收 open_arm_zones、不收 closed_arm_zones
2. zone_entry_distribution（open+closed）收两者
3. trajectory（坐标列，无 in_zone glob）不注入 --parameters-json
4. box_open_arm（只要 open）只收 open
5. 无 column_aliases 时所有 chart 不注入 --parameters-json
6. 多范式回归（OFT center / Zero Maze closed）裁剪不破坏各自 chart 注入
"""
from __future__ import annotations

import json

from ethoinsight.catalog.resolve import resolve_charts


def _parameters_json_of(charts, chart_id: str) -> dict | None:
    """从 resolve_charts 结果里取指定 chart 的 --parameters-json 反序列化结果。

    不含 --parameters-json 时返回 None（与「含但值为空 dict」区分）。
    """
    matched = [c for c in charts if c.id.startswith(chart_id)]
    assert matched, f"chart {chart_id} 未生成，charts={[c.id for c in charts]}"
    args = matched[0].args
    if "--parameters-json" not in args:
        return None
    idx = args.index("--parameters-json")
    return json.loads(args[idx + 1])


# ============================================================================
# 1. open_arm_time_ratio_bar 只收 open_arm_zones（复现 dogfood 红线）
# ============================================================================


def test_open_arm_bar_gets_only_open_zone_param(tmp_path):
    """bar（requires_columns=in_zone_open_arms_*）只该收 open_arm_zones。

    改动前：--parameters-json 同时含 open_arm_zones + closed_arm_zones（全并集）。
    改动后：只含 open_arm_zones。
    """
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    payload = _parameters_json_of(pc.charts, "open_arm_time_ratio_bar")
    assert payload is not None, "bar 应注入 --parameters-json"
    assert payload.get("open_arm_zones") == ["open"]
    assert "closed_arm_zones" not in payload, (
        f"bar 不该收 closed_arm_zones: {payload}"
    )


# ============================================================================
# 2. zone_entry_distribution 收 open + closed
# ============================================================================


def test_zone_entry_distribution_gets_both_zone_params(tmp_path):
    """zone_entry_distribution（requires_columns open+closed）应同时收两个 zone param。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    payload = _parameters_json_of(pc.charts, "zone_entry_distribution")
    assert payload is not None
    assert payload.get("open_arm_zones") == ["open"]
    assert payload.get("closed_arm_zones") == ["closed"]


# ============================================================================
# 3. trajectory 不注入 --parameters-json
# ============================================================================


def test_trajectory_gets_no_parameters_json(tmp_path):
    """trajectory（requires_columns 是坐标列，无 in_zone glob）不该注入 --parameters-json。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    assert _parameters_json_of(pc.charts, "trajectory") is None


# ============================================================================
# 4. box_open_arm 只收 open（aggregate 路径防回归）
# ============================================================================


def test_box_open_arm_gets_only_open_zone_param(tmp_path):
    """box_open_arm（只要 open）也只收 open_arm_zones（底层签名宽容碰巧没炸，裁剪后仍只该收 open）。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/r1.txt", "/tmp/r2.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=6, n_per_group=3, n_groups=2,
        groups={"control": ["/tmp/r1.txt", "/tmp/r2.txt"]},
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    payload = _parameters_json_of(pc.charts, "box_open_arm")
    assert payload is not None
    assert payload.get("open_arm_zones") == ["open"]
    assert "closed_arm_zones" not in payload


# ============================================================================
# 5. 无 column_aliases 时所有 chart 不注入 --parameters-json
# ============================================================================


def test_no_column_aliases_no_injection(tmp_path):
    """无 column_aliases（标准列名场景）→ _build_zone_aliases_overrides 返回 {} → 裁剪后仍 {}。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["in_zone_open_arms_center", "in_zone_closed_arms_center",
                 "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
    )
    for cid in ("open_arm_time_ratio_bar", "zone_entry_distribution", "trajectory"):
        assert _parameters_json_of(pc.charts, cid) is None, f"{cid} 不该注入（无 aliases）"


# ============================================================================
# 6. 多范式回归：OFT center / Zero Maze closed 裁剪不破坏各自注入
# ============================================================================


def test_oft_center_bar_gets_only_center_zone_param(tmp_path):
    """OFT 回归：宽通配 chart（requires_columns=in_zone*，无具体 concept keyword）不裁剪。

    OFT/ZM chart 的 requires_columns 写成 in_zone*（裸通配），_extract_concept_keyword 提不出
    具体 concept → 裁剪逻辑视为「未明确排除任何 zone」→ 返回原 zone_overrides（不回归）。
    用标准列名（in_zone_center_point，匹配 in_zone*）让 chart 生成，断言 center_zone 仍注入。
    """
    pc = resolve_charts(
        paradigm="oft",
        columns=["in_zone_center_point", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"in_zone_center_point": "center"},
    )
    payload = _parameters_json_of(pc.charts, "center_time_ratio_bar")
    assert payload is not None, "OFT center bar 应注入 center_zone（宽通配不裁剪）"
    assert payload.get("center_zone") == "in_zone_center_point"


def test_zero_maze_chart_gets_correct_zone_params(tmp_path):
    """Zero Maze 回归：宽通配 chart（in_zone*）不裁剪，closed_zones（list）仍正确注入。"""
    pc = resolve_charts(
        paradigm="zero_maze",
        columns=["in_zone_open", "in_zone_closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"in_zone_open": "open", "in_zone_closed": "closed"},
    )
    # ZM charts 用 in_zone* 宽通配 → 不裁剪 → 收到全量 zone_overrides（含 closed_zones）
    matched = [c for c in pc.charts if c.id in ("box_open_zone", "bar_open_zone")]
    assert matched, f"ZM zone charts 未生成: {[c.id for c in pc.charts]}"
    for ch in matched:
        args = ch.args
        if "--parameters-json" not in args:
            continue
        idx = args.index("--parameters-json")
        payload = json.loads(args[idx + 1])
        assert "closed_zones" in payload, f"ZM chart {ch.id} 应含 closed_zones（宽通配不裁剪）: {payload}"
