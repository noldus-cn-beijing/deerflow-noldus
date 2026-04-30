"""扫描 demodata/ev19 templates/ 生成 EV19 模板事实表 + by-template 草稿。

事实表（EV19_TEMPLATES_INDEX）只装从 templateMetaData.xml 和目录名能机器可读地解析出来的字段。
不包含任何领域解读。领域解读由行为学同事在 by-template/<大类>.md 里以 markdown 维护。

输出：
- docs/review-packages/2026-04-29-ev19-templates/by-template/<大类>.md  (20 个文件)
- docs/review-packages/2026-04-29-ev19-templates/_facts.json            (机器可读事实表)
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demodata" / "ev19 templates"
OUT_DIR = ROOT / "docs" / "review-packages" / "2026-04-29-ev19-templates"
BY_TEMPLATE_DIR = OUT_DIR / "by-template"

# 把目录名分解成 (大类, 后缀串)。手写规则比通用拆分更准确。
# 关键：大类是 EV 软件里 arena 的概念分组，不一定等于命名前缀。
CATEGORY_PREFIXES = [
    # 多词大类（先匹配长前缀，避免被误拆）
    ("DanioVision DVOC 004x", "DanioVision DVOC 004x"),
    ("Cross Maze-Fish", "Cross Maze-Fish"),
    ("T-Maze-Fish", "T-Maze"),  # T-Maze-Fish-* 都归 T-Maze 大类
    ("T-Maze-Rodents-Other", "T-Maze"),
    ("UgoBasileActiveAvoidance", "UgoBasileActiveAvoidance"),
    ("UgoBasileFCS", "UgoBasileFCS"),
    ("Y-Maze", "Y-Maze"),
    ("Radial-8-arm", "Radial-8-arm"),
    # 单词大类
    ("AquariumTrack3D", "AquariumTrack3D"),
    ("BarnesMaze", "BarnesMaze"),
    ("FlightChamberTrack3D", "FlightChamberTrack3D"),
    ("MWM", "MWM"),
    ("NoTemplate", "NoTemplate"),
    ("OpenFieldCircle", "OpenFieldCircle"),
    ("OpenFieldRectangle", "OpenFieldRectangle"),
    ("PhenoTyper", "PhenoTyper"),
    ("PlusMaze", "PlusMaze"),
    ("PorsoltCylinder", "PorsoltCylinder"),
    ("Sociability", "Sociability"),
    ("WellPlate", "WellPlate"),
    ("ZeroMaze", "ZeroMaze"),
]


@dataclass
class Variant:
    template_id: str  # 目录名
    category: str  # 大类，如 "OpenFieldRectangle"
    arena_template: str  # m_strArenaTemplate
    zone_template: str  # m_strZoneTemplate
    has_other_types: bool  # m_bOtherTypes
    bypass_arena_grid: bool  # m_bBypassArenaGrid
    bypass_subject_count_and_roles: bool  # m_bBypassSubjectCountAndRoles
    bypass_subject_features: bool  # m_bBypassSubjectFeatures
    inferred_subject_hint: str  # 从目录名后缀猜的 subject 范围（仅作提示，不可作为事实）
    inferred_zone_config: str  # 从目录名后缀猜的 zone 配置维度
    inferred_array_size: str  # 从目录名后缀猜的阵列规模
    raw_dir_suffix: str  # 大类后剩下的尾巴，方便人工校验


def classify(template_id: str) -> tuple[str, str]:
    """返回 (category, suffix)。"""
    for prefix, category in CATEGORY_PREFIXES:
        if template_id == prefix:
            return category, ""
        if template_id.startswith(prefix + "-"):
            return category, template_id[len(prefix) + 1 :]
    raise ValueError(f"unmapped template: {template_id}")


def infer_zone_config(suffix: str) -> str:
    s = suffix.lower()
    if "allzones" in s:
        return "AllZones"
    if "novobjzones" in s:
        return "NovObjZones"
    if "feedingshelter" in s:
        return "FeedingShelter"
    if "subdivided4x4" in s:
        return "Subdivided4x4"
    if "subdivided3x3" in s:
        return "Subdivided3x3"
    if "subdivided2x2" in s:
        return "Subdivided2x2"
    if "afewzones" in s:
        return "AFewZones"
    if "fewzones" in s:
        return "FewZones"
    if "nozones" in s:
        return "NoZones"
    if "20holes" in s:
        return "20Holes"
    return "Default"


def infer_array_size(suffix: str) -> str:
    s = suffix.lower()
    if "16x" in s:
        return "16x"
    if "quad" in s:
        return "Quad"
    if "1cubicle" in s:
        return "1cubicle"
    if "4cubicles" in s:
        return "4cubicles"
    if "96w" in s:
        return "96w"
    return "Single"


def infer_subject_hint(template_id: str, suffix: str) -> str:
    """从目录名后缀猜适用 subject。注意：XML 字段都是空的，只能这样猜。"""
    s = (template_id + " " + suffix).lower()
    hints = []
    if "fish" in s:
        hints.append("fish")
    if "mice" in s:
        hints.append("mice")
    if "rodent" in s and "ratother" not in s:
        hints.append("rodent")
    if "ratother" in s or "rat" in s:
        hints.append("rat_or_other")
    if "insect" in s:
        hints.append("insect")
    if not hints:
        hints.append("not specified by name")
    return ", ".join(hints)


def parse_xml(xml_path: Path) -> dict:
    if not xml_path.exists():
        return {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta = root.find("TemplateMetaData")
    if meta is None:
        return {}

    def text(tag: str, default: str = "") -> str:
        elt = meta.find(tag)
        return (elt.text or default).strip() if elt is not None and elt.text is not None else default

    def b(tag: str) -> bool:
        return text(tag, "0").strip() == "1"

    return {
        "arena_template": text("m_strArenaTemplate"),
        "zone_template": text("m_strZoneTemplate"),
        "has_other_types": b("m_bOtherTypes"),
        "bypass_arena_grid": b("m_bBypassArenaGrid"),
        "bypass_subject_count_and_roles": b("m_bBypassSubjectCountAndRoles"),
        "bypass_subject_features": b("m_bBypassSubjectFeatures"),
    }


def scan() -> list[Variant]:
    variants: list[Variant] = []
    for tdir in sorted(DEMO_DIR.iterdir()):
        if not tdir.is_dir():
            continue
        template_id = tdir.name
        category, suffix = classify(template_id)
        xml_data = parse_xml(tdir / "templateMetaData.xml")
        v = Variant(
            template_id=template_id,
            category=category,
            arena_template=xml_data.get("arena_template", ""),
            zone_template=xml_data.get("zone_template", ""),
            has_other_types=xml_data.get("has_other_types", False),
            bypass_arena_grid=xml_data.get("bypass_arena_grid", False),
            bypass_subject_count_and_roles=xml_data.get("bypass_subject_count_and_roles", False),
            bypass_subject_features=xml_data.get("bypass_subject_features", False),
            inferred_subject_hint=infer_subject_hint(template_id, suffix),
            inferred_zone_config=infer_zone_config(suffix),
            inferred_array_size=infer_array_size(suffix),
            raw_dir_suffix=suffix,
        )
        variants.append(v)
    return variants


CATEGORY_DISPLAY_HINT = {
    "OpenFieldRectangle": "矩形旷场（待行为学同事确认中文）",
    "OpenFieldCircle": "圆形旷场（待行为学同事确认中文）",
    "PhenoTyper": "PhenoTyper 居家观察舱（待行为学同事确认中文）",
    "MWM": "Morris 水迷宫（待行为学同事确认中文）",
    "PlusMaze": "高架十字迷宫（待行为学同事确认中文）",
    "ZeroMaze": "零迷宫（待行为学同事确认中文）",
    "BarnesMaze": "Barnes 迷宫（待行为学同事确认中文）",
    "T-Maze": "T 迷宫（待行为学同事确认中文）",
    "Y-Maze": "Y 迷宫（待行为学同事确认中文）",
    "Cross Maze-Fish": "鱼用十字迷宫（待行为学同事确认中文）",
    "Radial-8-arm": "8 臂放射迷宫（待行为学同事确认中文）",
    "Sociability": "社交偏好箱（待行为学同事确认中文）",
    "PorsoltCylinder": "强迫游泳圆筒（待行为学同事确认中文）",
    "WellPlate": "孔板（待行为学同事确认中文）",
    "AquariumTrack3D": "鱼缸 3D 跟踪（待行为学同事确认中文）",
    "FlightChamberTrack3D": "飞行舱 3D 跟踪（待行为学同事确认中文）",
    "DanioVision DVOC 004x": "DanioVision 96 孔板（待行为学同事确认中文）",
    "UgoBasileActiveAvoidance": "Ugo Basile 主动回避（待行为学同事确认中文）",
    "UgoBasileFCS": "Ugo Basile 恐惧条件化（待行为学同事确认中文）",
    "NoTemplate": "无模板/自定义（待行为学同事确认中文）",
}


def render_by_template_md(category: str, variants: list[Variant]) -> str:
    """每个大类一个 markdown 草稿。机器字段已经填好；🟡 区域留给行为学同事填。"""
    display = CATEGORY_DISPLAY_HINT.get(category, "（待行为学同事确认中文名）")
    lines: list[str] = []
    lines.append(f"# EV19 大类：{category}")
    lines.append("")
    lines.append(f"**中文名**（待补充）：{display}")
    lines.append("")
    lines.append("> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。")
    lines.append("> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。")
    lines.append("")
    lines.append(f"该大类下共 **{len(variants)}** 个变体。")
    lines.append("")

    # 🟡 大类层面的领域知识
    lines.append("## 🟡 这个大类用来做什么？（待补充）")
    lines.append("")
    lines.append("<!-- 例：「测试啮齿动物在新环境中的探索-焦虑行为」「测试鱼群同伴聚集程度」 -->")
    lines.append("")
    lines.append("- 主要研究对象：")
    lines.append("- 典型实验类型：")
    lines.append("- 学术范式名（中英）：")
    lines.append("")
    lines.append("## 🟡 何时不该选这个大类？（待补充）")
    lines.append("")
    lines.append("<!-- 例：「如果测试对象是鱼，不应选 OpenFieldRectangle，应选 OpenFieldCircle 或 AquariumTrack3D」 -->")
    lines.append("")
    lines.append("## 🟡 关键参考文献（待补充）")
    lines.append("")
    lines.append("- ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 每个变体一节
    for v in variants:
        lines.append(f"## 变体：{v.template_id}")
        lines.append("")
        lines.append("### 🟢 EV19 机器字段（自动抽取，请勿修改）")
        lines.append("")
        lines.append(f"- **目录名**：`{v.template_id}`")
        lines.append(f"- **arena_template**：`{v.arena_template}`")
        lines.append(f"- **zone_template**：`{v.zone_template}`")
        lines.append(f"- **bypass_arena_grid**：{v.bypass_arena_grid}")
        lines.append(f"- **bypass_subject_count_and_roles**：{v.bypass_subject_count_and_roles}")
        lines.append(f"- **bypass_subject_features**：{v.bypass_subject_features}")
        lines.append(f"- **m_bOtherTypes**：{v.has_other_types}")
        lines.append("")
        lines.append('### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）')
        lines.append("")
        lines.append(f"- **推测适用 subject**：{v.inferred_subject_hint}")
        lines.append(f"- **推测 zone 配置**：{v.inferred_zone_config}")
        lines.append(f"- **推测阵列规模**：{v.inferred_array_size}")
        lines.append(f"- **目录名尾缀**：`{v.raw_dir_suffix}`")
        lines.append("")
        lines.append("### 🟡 这个变体相对其他变体的核心差异（待补充）")
        lines.append("")
        lines.append("<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->")
        lines.append("")
        lines.append("### 🟡 推荐的实验场景（待补充）")
        lines.append("")
        lines.append("<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->")
        lines.append("")
        lines.append("### 🟡 不该用这个变体的场景（待补充）")
        lines.append("")
        lines.append("<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->")
        lines.append("")
        lines.append("### 🟡 对应学术范式（待补充）")
        lines.append("")
        lines.append("<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->")
        lines.append("")
        lines.append("- ")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BY_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    variants = scan()

    # 1. 写机器可读事实表
    facts = {
        "schema_version": 1,
        "source": "demodata/ev19 templates/ + templateMetaData.xml",
        "total_categories": len({v.category for v in variants}),
        "total_variants": len(variants),
        "variants": [asdict(v) for v in variants],
    }
    facts_path = OUT_DIR / "_facts.json"
    facts_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {facts_path}")

    # 2. 按大类分组写 by-template 草稿
    by_cat: dict[str, list[Variant]] = defaultdict(list)
    for v in variants:
        by_cat[v.category].append(v)

    for category, vs in sorted(by_cat.items()):
        # 文件名安全化
        fname = category.replace(" ", "_") + ".md"
        path = BY_TEMPLATE_DIR / fname
        path.write_text(render_by_template_md(category, vs), encoding="utf-8")
        print(f"wrote {path} ({len(vs)} variant(s))")


if __name__ == "__main__":
    main()
