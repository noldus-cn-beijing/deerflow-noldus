"""catalog 裸 `in_zone*` requires_columns 越界 bug 修复回归。

背景：chart 的 requires_columns 用裸 `in_zone*`（无 concept keyword，如
zero_maze/ldb/oft 的 box/bar 图）。当数据经 column_aliases 对齐成 `open`/`closed`/
`light`/`center` 等 concept 后，物理列名又非 in_zone 开头时，`_missing_columns` 走
`_any_concept_matches_pattern` → `_concept_matches_pattern(concept, "in_zone*")`，
后者做 `pat[len("in_zone_"):]`（len==8）对长 7 的裸 `in_zone*` 越界成空串→keyword=""
→短路 False → box/bar 图被误判缺列而 skip（"指标能算、图不能画"）。

治法（路 B，门内放行）：裸 `in_zone*` 是「存在分析 zone 列即可」的弱声明。在门函数
`_any_concept_matches_pattern` 内，只要任一 available 列对齐到非忽略 concept，弱声明即满足
（concept 必是 HITL 已对齐的真实 zone 列）。带 concept 的 `in_zone_open*` 精确匹配行为不变。
路由枚举 `_build_zone_aliases_overrides` 不经此函数，裸 in_zone* 在那里仍无 keyword→不路由。
"""
from __future__ import annotations

from ethoinsight.catalog.resolve import (
    _any_concept_matches_pattern,
    _concept_matches_pattern,
    resolve_charts,
)

RAW_FILES_SAMPLE = ["/tmp/raw1.txt"]


# ============================================================================
# 1-3. 三范式裸 in_zone* 经门函数断言（原 bug 处，现绿）
#   直接断言 _any_concept_matches_pattern（修复所在函数）：available 列经 alias
#   对齐成 concept 后，裸 in_zone* 弱声明应满足。
# ============================================================================


def test_ezm_open_concept_matches_bare_in_zone_pattern():
    """EZM：open/closed concept + 裸 in_zone* → 门函数 True（原 bug 处，修复后绿）。

    修复前必红：_concept_matches_pattern("open","in_zone*") 因 pat[8:] 越界成空串
    → keyword="" → 短路 False。
    """
    available = ["in_open", "in_closed"]
    aliases = {"in_open": "open", "in_closed": "closed"}
    assert _any_concept_matches_pattern(available, aliases, "in_zone*") is True


def test_ldb_light_concept_matches_bare_in_zone_pattern():
    """LDB：light/dark concept + 裸 in_zone* → 门函数 True。"""
    available = ["in_light_zone", "in_dark_zone"]
    aliases = {"in_light_zone": "light", "in_dark_zone": "dark"}
    assert _any_concept_matches_pattern(available, aliases, "in_zone*") is True


def test_oft_center_concept_matches_bare_in_zone_pattern():
    """OFT：center concept + 裸 in_zone* → 门函数 True。"""
    available = ["in_center"]
    aliases = {"in_center": "center"}
    assert _any_concept_matches_pattern(available, aliases, "in_zone*") is True


def test_bare_in_zone_pattern_false_when_no_aligned_concept():
    """裸 in_zone* 弱声明在无任何对齐 zone concept 时仍 False（防过度放行）。

    available 列虽有 alias 但全为 __ignore__ → 无可用 zone 列 → 门不应放行。
    """
    available = ["x_center", "y_center"]
    aliases = {"x_center": "__ignore__", "y_center": "__ignore__"}
    assert _any_concept_matches_pattern(available, aliases, "in_zone*") is False


# ============================================================================
# 4. 对照防回归（防 seesaw：带 concept 的精确匹配行为不变）
# ============================================================================


def test_concept_keyword_in_zone_open_star_unchanged():
    """带 concept 的 `in_zone_open*` 精确 keyword 匹配行为不变（防回归）。

    _concept_matches_pattern 未改动：open 仍 True、closed 仍 False。
    """
    assert _concept_matches_pattern("open", "in_zone_open*") is True
    assert _concept_matches_pattern("closed", "in_zone_open*") is False
    # 门函数同样：带 concept 的精确匹配不变
    available = ["in_open_arms", "in_closed_arms"]
    aliases = {"in_open_arms": "open", "in_closed_arms": "closed"}
    assert _any_concept_matches_pattern(available, aliases, "in_zone_open*") is True
    assert _any_concept_matches_pattern(
        ["in_closed_arms"], {"in_closed_arms": "closed"}, "in_zone_open*"
    ) is False


# ============================================================================
# 5. 端到端验收（reward-hacking 自检：看 resolve_charts 真实输出，非自述）
#    EZM 真实列经 column_aliases 对齐成 open/closed concept 后，box_open_zone 不再被误 skip。
# ============================================================================


EZM_COLUMNS_ALIASED = [
    "trial_time", "recording_time", "x_center", "y_center",
    "in_open", "in_closed",  # 物理列名非 in_zone*；HITL 对齐成 open/closed concept
]


def test_ezm_box_open_zone_not_skipped_with_bare_in_zone_pattern(tmp_path):
    """EZM 裸 in_zone* box/bar 图在 alias 对齐后不再被误 skip。

    修复前必红：column_aliases 把 in_open→open / in_closed→closed 对齐，
    物理列名 'in_open'/'in_closed' 不命中 `in_zone*` glob（非 in_zone 开头），
    落到 _concept_matches_pattern("open", "in_zone*") 这条 bug 路径 → False
    → box_open_zone/bar_open_zone 进 skipped、charts 保持空。
    修复后：门函数对裸 in_zone* 放行（有对齐 zone concept）→ 两图保留在 charts。
    """
    pc = resolve_charts(
        paradigm="zero_maze", columns=EZM_COLUMNS_ALIASED, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        column_aliases={"in_open": "open", "in_closed": "closed"},
    )
    chart_ids = {c.id for c in pc.charts}
    skipped_ids = {s.id for s in pc.skipped}
    # 看真实输出：box_open_zone 必须在 charts，不在 skipped
    assert "box_open_zone" in chart_ids, (
        f"box_open_zone 应在 charts（裸 in_zone* 修复后）；"
        f"charts={sorted(chart_ids)}, skipped={sorted(skipped_ids)}"
    )
    assert "box_open_zone" not in skipped_ids


# ============================================================================
# 5b. LDB / OFT 端到端同 bug 路径（三范式各有断言，守 spec §五验收 1）
# ============================================================================


def test_ldb_box_light_not_skipped_with_bare_in_zone_pattern(tmp_path):
    """LDB 裸 in_zone* box_light 图在 alias 对齐后不再被误 skip。

    LDB catalog（ldb.yaml:80）box_light requires_columns 用裸 in_zone*。
    """
    ldb_cols = ["trial_time", "x_center", "y_center", "in_light", "in_dark"]
    pc = resolve_charts(
        paradigm="ldb", columns=ldb_cols, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        column_aliases={"in_light": "light", "in_dark": "dark"},
    )
    chart_ids = {c.id for c in pc.charts}
    assert "box_light" in chart_ids, (
        f"box_light 应在 charts（裸 in_zone* 修复后）；charts={sorted(chart_ids)}"
    )


def test_oft_center_charts_not_skipped_with_bare_in_zone_pattern(tmp_path):
    """OFT 裸 in_zone* center 图在 alias 对齐后不再被误 skip。

    OFT catalog（oft.yaml:183/190/197）box_center/center_time_ratio_bar/
    center_entry_summary requires_columns 均用裸 in_zone*。
    """
    oft_cols = ["trial_time", "x_center", "y_center", "in_center"]
    pc = resolve_charts(
        paradigm="oft", columns=oft_cols, raw_files=RAW_FILES_SAMPLE,
        workspace_dir=str(tmp_path), total_subjects=6, n_per_group=3, n_groups=2,
        column_aliases={"in_center": "center"},
    )
    chart_ids = {c.id for c in pc.charts}
    assert "box_center" in chart_ids, (
        f"box_center 应在 charts（裸 in_zone* 修复后）；charts={sorted(chart_ids)}"
    )
    assert "center_time_ratio_bar" in chart_ids

