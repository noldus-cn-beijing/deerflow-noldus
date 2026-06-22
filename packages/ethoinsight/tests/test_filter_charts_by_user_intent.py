"""_filter_charts_by_user_intent 单元测试。

回归根因（2026-06-18 第四轮 dogfood 后续）：lead 派遣 chart-maker 时把开放意图
"把刚才的结论画成图" paraphrase 成 "箱线图/小提琴图等"，user_intent 含"箱线图"
触发硬过滤 → EPM 6 张图被砍到只剩 box_open_arm 1 张。

修复：意图含开放标记（"等"/"之类"/"such as"/"etc" 等）时，图型关键词只作提示
不作硬约束，全部图通过。
"""

from __future__ import annotations

from dataclasses import dataclass

from ethoinsight.catalog.resolve import _filter_charts_by_user_intent


@dataclass
class _FakeChart:
    id: str


def _charts() -> list[_FakeChart]:
    # 模拟 EPM catalog 的 6 张图
    return [
        _FakeChart("box_open_arm"),
        _FakeChart("open_arm_time_ratio_bar"),
        _FakeChart("zone_entry_distribution"),
        _FakeChart("trajectory"),
        _FakeChart("heatmap"),
        _FakeChart("rose"),
    ]


def _ids(charts) -> list[str]:
    return [c.id for c in charts]


def test_none_intent_returns_all():
    assert _ids(_filter_charts_by_user_intent(_charts(), None)) == _ids(_charts())


def test_no_keyword_intent_returns_all():
    # ASKVIZ "A. 画图" 不含具体图型词 → 全画（legacy 行为不变）
    assert _ids(_filter_charts_by_user_intent(_charts(), "把刚才的结论画成图")) == _ids(_charts())


def test_exact_chart_type_hard_filters():
    # 用户明确"只画箱线图"（无开放标记）→ 仍硬过滤到 box（不破坏精确意图）
    out = _filter_charts_by_user_intent(_charts(), "只画箱线图")
    assert _ids(out) == ["box_open_arm"]


def test_open_ended_box_keeps_all():
    # 根因复现：开放意图 paraphrase 成"箱线图/小提琴图等" → 不硬过滤，全画
    intent = "用户说把刚才的结论画成图，希望看到组间对比的箱线图/小提琴图等可视化图表"
    out = _filter_charts_by_user_intent(_charts(), intent)
    assert _ids(out) == _ids(_charts()), "开放标记'等'后图型词应只作提示，不砍其余图"


def test_open_ended_markers_variants():
    # 各种开放标记都应放行全部图
    for marker_intent in [
        "画箱线图之类的",
        "画箱线图等图",
        "draw box plots such as the comparison",
        "box plot etc",
        "包括箱线图在内的图",
    ]:
        out = _filter_charts_by_user_intent(_charts(), marker_intent)
        assert _ids(out) == _ids(_charts()), f"开放标记意图 {marker_intent!r} 应返回全部图"


def test_multi_explicit_keywords_still_filter():
    # 明确多图型（无开放标记）→ 累加匹配，硬过滤
    out = _filter_charts_by_user_intent(_charts(), "画箱线图和轨迹图")
    assert set(_ids(out)) == {"box_open_arm", "trajectory"}


def test_explicit_keyword_no_catalog_match_falls_back_to_all():
    # 意图图型在本范式 catalog 没有对应 → 回退全画（别给空 plan）
    out = _filter_charts_by_user_intent(_charts(), "只画热力图")
    # heatmap 在列表里，这条会过滤到 heatmap；改用一个 catalog 没有的精确意图
    out2 = _filter_charts_by_user_intent(
        [_FakeChart("trajectory"), _FakeChart("rose")], "只画箱线图"
    )
    assert _ids(out2) == ["trajectory", "rose"]
    assert _ids(out) == ["heatmap"]
