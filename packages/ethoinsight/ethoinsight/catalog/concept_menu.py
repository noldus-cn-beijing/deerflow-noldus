"""从 catalog 统一概念模型生成概念菜单，消除双存。

Stage 4 (PR #115 Q3) — 构建期生成器。读 catalog 的 ``resolved_zone_concepts``
（Stage 2/3 产出），渲染为 markdown 菜单片段，产出独立的 ``.generated.md`` 文件。
手写的 SKILL.md / answer-mapping.md 只链接指向这些生成文件，不再内嵌手写概念表。

用法（构建期，整文件产出）::

    python -m ethoinsight.catalog.concept_menu --style skill          > zone-concepts.generated.md
    python -m ethoinsight.catalog.concept_menu --style answer-mapping > zone-concepts-mapping.generated.md
"""

from __future__ import annotations

import argparse
import sys

from ethoinsight.catalog.loader import _PARADIGM_ALIASES, load_catalog

# paradigm key → skill 风格的展示简称（纯展示标签，不是概念知识双存）
_PARADIGM_DISPLAY_SHORT: dict[str, str] = {
    "epm": "EPM",
    "open_field": "OFT",
    "light_dark_box": "LDB",
    "zero_maze": "Zero Maze",
    "forced_swim": "FST",
    "tail_suspension": "TST",
}

_GENERATED_HEADER = (
    "<!-- GENERATED FILE — DO NOT EDIT.\n"
    "Run `make gen-references` to regenerate. -->\n"
)


def _supported_paradigms() -> list[str]:
    """返回 v0.1 全部范式的 academic key 列表。

    范式清单 SSOT = ``loader._PARADIGM_ALIASES`` 的键，禁止在生成器内联第三份。
    """
    return list(_PARADIGM_ALIASES.keys())


def list_zone_concepts(paradigm: str) -> list[str]:
    """返回 *paradigm* 的合法分析区概念关键词有序列表。

    读 ``cat.resolved_zone_concepts`` 的键集合（concept 名），对 ``binding`` 透明
    （binding=None 的一等概念如 OFT border 同样入列）。空集范式返回 ``[]``。
    """
    cat = load_catalog(paradigm)
    return sorted(cat.resolved_zone_concepts.keys())


def render_skill_list(paradigms: list[str] | None = None) -> str:
    """渲染 skill 风格的 markdown 概念列表（简称 label）。

    每行格式: ``- {简称}: `concept1` / `concept2` | 无自定义分析区``
    """
    if paradigms is None:
        paradigms = _supported_paradigms()
    lines = [_GENERATED_HEADER]
    for p in paradigms:
        concepts = list_zone_concepts(p)
        label = _PARADIGM_DISPLAY_SHORT.get(p, p)
        if concepts:
            joined = " / ".join(f"`{c}`" for c in concepts)
            lines.append(f"- {label}: {joined}")
        else:
            lines.append(f"- {label}: 无自定义分析区")
    return "\n".join(lines) + "\n"


def render_answer_mapping_table(paradigms: list[str] | None = None) -> str:
    """渲染 answer-mapping 风格的 markdown 概念表格（academic key label）。

    表格列: paradigm | 分析区概念关键词
    """
    if paradigms is None:
        paradigms = _supported_paradigms()
    lines = [
        _GENERATED_HEADER.rstrip("\n"),
        "",
        "| paradigm | 分析区概念关键词 |",
        "|----------|----------------|",
    ]
    for p in paradigms:
        concepts = list_zone_concepts(p)
        if concepts:
            joined = " / ".join(f"`{c}`" for c in concepts)
            lines.append(f"| {p} | {joined} |")
        else:
            lines.append(f"| {p} | （无自定义分析区） |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="从 catalog 统一概念模型生成概念菜单 markdown 片段"
    )
    parser.add_argument(
        "--style",
        required=True,
        choices=["skill", "answer-mapping"],
        help="输出风格: skill (简称列表) 或 answer-mapping (academic key 表格)",
    )
    parser.add_argument(
        "--paradigms",
        nargs="*",
        default=None,
        help=(
            "要生成的范式列表（academic key）。"
            "默认使用全部已支持范式（取自 _PARADIGM_ALIASES）"
        ),
    )
    args = parser.parse_args(argv)
    paradigms = args.paradigms if args.paradigms else _supported_paradigms()

    if args.style == "skill":
        print(render_skill_list(paradigms), end="")
    else:
        print(render_answer_mapping_table(paradigms), end="")


if __name__ == "__main__":
    main()
