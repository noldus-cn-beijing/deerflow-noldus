"""chart-maker SKILL.md 内容契约：防止「逐字拼接 entry.args」纪律 + `--parameters-json`
说明被未来编辑悄悄删掉。

背景（dogfood thread 339512dd）：chart-maker 重跑批次时重构命令、漏掉
``--parameters-json``，导致需 zone 列对齐参数的图（如 EPM open_arm_time_ratio_bar）
``compute`` 返回 None、退出码 1、不出图，靠第三次单独重试才救活。根因是 SKILL.md
第 5 步描述 entry.args 时只枚举了 ``--inputs/--groups/--output/--paradigm``，**漏列
``--parameters-json``**，诱导 LLM 重构命令时丢参。

修法是把 args 描述改成「整体逐字拼接、不增不减不重构」并显式点名 ``--parameters-json``。
这些断言锁住那段纪律文字（数据层「resolve 是否真注入 --parameters-json」由
packages/ethoinsight/tests/test_chart_zone_overrides_filtered.py 守，两层互补）。
"""
from __future__ import annotations

from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "custom" / "ethoinsight-chart-maker"


def _skill_md() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


def _fallback_tree() -> str:
    return (SKILL_DIR / "references" / "fallback-decision-tree.md").read_text(encoding="utf-8")


def test_skill_md_exists():
    assert (SKILL_DIR / "SKILL.md").exists()


def test_skill_mentions_parameters_json_in_args():
    """args 描述必须显式点名 --parameters-json（否则 LLM 重构命令会漏掉它）。"""
    md = _skill_md()
    assert "--parameters-json" in md, (
        "SKILL.md 第 5 步 args 描述必须显式列出 --parameters-json —— 漏列会诱导 "
        "chart-maker 重跑时丢掉 zone 对齐参数（dogfood 339512dd 根因）"
    )


def test_skill_has_verbatim_splice_rule():
    """必须有「逐字拼接 entry.args、不增不减不重构」的正面纪律。"""
    md = _skill_md()
    # 正面指令关键词（deepseek 正面提示：描述想要的行为，而非「不要丢参数」）
    assert "逐项" in md or "逐字" in md, "SKILL.md 缺「逐项/逐字拼接 entry.args」纪律"
    assert "不增不减不重构" in md, "SKILL.md 缺「不增不减不重构」的 args 拼接铁律"


def test_skill_warns_against_reconstructing_command_on_rerun():
    """重跑某张图必须从 plan 的 args 重取，而非凭记忆手敲（最易漏 --parameters-json）。"""
    md = _skill_md()
    assert "重跑" in md, "SKILL.md 缺重跑场景的命令重取纪律"
    # 重跑段必须指向从 plan_charts.json 的 args 取，而非手敲
    assert "plan_charts.json" in md


def test_fallback_tree_representative_subset_is_group_balanced():
    """代表性子集文字应反映「按组轮转」（与 select_charts_by_priority 的 group_of 行为一致），
    而非旧的「subject_index=0 优先」单组描述。"""
    tree = _fallback_tree()
    assert "按组轮转" in tree or "每组" in tree, (
        "fallback-decision-tree 的代表性子集描述应体现按组均衡（各组首个 subject 各一张）"
    )
