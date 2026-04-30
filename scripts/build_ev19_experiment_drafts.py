"""生成 by-experiment 空模板。

行为学同事需要补充每个实验的："适用模板（推荐顺序+取舍）"、"必须的指标"、
"常见脱险点"、"参考文献"。

实验清单基于：
1. 现有 packages/ethoinsight/ethoinsight/templates/ 已有模板（shoaling）
2. CLAUDE.md 路线图说要做的范式（EPM, OFT）
3. EV19 demodata 大类反推的常见学术范式
4. Phase 1 飞轮要覆盖的范式

清单保守：宁可少写、让同事补充，也不写一堆 agent 用不到的。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "review-packages" / "2026-04-29-ev19-templates" / "by-experiment"


@dataclass
class Experiment:
    slug: str  # 文件名 + skill 内部 paradigm key
    cn_name: str
    en_name: str
    candidate_templates_hint: list[str]  # 给同事提示候选；不一定全选
    notes: str = ""  # 起草时的简短说明（不进入最终 markdown）


EXPERIMENTS: list[Experiment] = [
    Experiment(
        slug="open_field",
        cn_name="旷场实验",
        en_name="Open Field Test (OFT)",
        candidate_templates_hint=[
            "OpenFieldRectangle-AllZones",
            "OpenFieldRectangle-NoZones",
            "OpenFieldCircle-AllZones",
            "OpenFieldCircle-NoZones-Rodents-Other",
        ],
        notes="经典啮齿焦虑/探索范式，方形 vs 圆形要看物种习惯",
    ),
    Experiment(
        slug="novel_object",
        cn_name="新物体识别",
        en_name="Novel Object Recognition (NOR)",
        candidate_templates_hint=[
            "OpenFieldRectangle-NovObjZones",
            "OpenFieldCircle-NovObjZones",
        ],
    ),
    Experiment(
        slug="epm",
        cn_name="高架十字迷宫",
        en_name="Elevated Plus Maze (EPM)",
        candidate_templates_hint=[
            "PlusMaze-AllZones",
            "PlusMaze-FewZones",
            "PlusMaze-NoZones",
        ],
        notes="焦虑金标范式之一",
    ),
    Experiment(
        slug="zero_maze",
        cn_name="零迷宫",
        en_name="Zero Maze",
        candidate_templates_hint=[
            "ZeroMaze-AllZones",
            "ZeroMaze-NoZones",
        ],
    ),
    Experiment(
        slug="light_dark_box",
        cn_name="明暗箱",
        en_name="Light-Dark Box",
        candidate_templates_hint=[
            "OpenFieldRectangle-Subdivided2x2",
            "NoTemplate",
        ],
        notes="EV19 没有专门的 LightDark 模板？同事确认是不是用 Subdivided 或自定义",
    ),
    Experiment(
        slug="morris_water_maze",
        cn_name="Morris 水迷宫",
        en_name="Morris Water Maze (MWM)",
        candidate_templates_hint=[
            "MWM-AllZones",
            "MWM-AFewZones",
            "MWM-NoZones",
        ],
    ),
    Experiment(
        slug="barnes_maze",
        cn_name="Barnes 迷宫",
        en_name="Barnes Maze",
        candidate_templates_hint=[
            "BarnesMaze-20Holes",
            "BarnesMaze-NoZones",
        ],
    ),
    Experiment(
        slug="t_maze",
        cn_name="T 迷宫",
        en_name="T-Maze",
        candidate_templates_hint=[
            "T-Maze-Rodents-Other-AllZones",
            "T-Maze-Rodents-Other-NoZones",
            "T-Maze-Fish-AllZones",
            "T-Maze-Fish-NoZones",
        ],
        notes="鱼/啮齿都能用，要按物种区分",
    ),
    Experiment(
        slug="y_maze",
        cn_name="Y 迷宫",
        en_name="Y-Maze",
        candidate_templates_hint=[
            "Y-Maze-AllZones",
            "Y-Maze-NoZones",
        ],
    ),
    Experiment(
        slug="cross_maze_fish",
        cn_name="鱼用十字迷宫",
        en_name="Cross Maze (Fish)",
        candidate_templates_hint=[
            "Cross Maze-Fish-AllZones",
            "Cross Maze-Fish-NoZones",
        ],
    ),
    Experiment(
        slug="radial_8_arm",
        cn_name="8 臂放射迷宫",
        en_name="Radial 8-Arm Maze",
        candidate_templates_hint=[
            "Radial-8-arm-AllZones",
            "Radial-8-arm-NoZones",
        ],
    ),
    Experiment(
        slug="forced_swim",
        cn_name="强迫游泳",
        en_name="Forced Swim Test (FST)",
        candidate_templates_hint=[
            "PorsoltCylinder-AllZones",
            "PorsoltCylinder-NoZones",
        ],
        notes="抑郁/绝望经典范式",
    ),
    Experiment(
        slug="sociability",
        cn_name="社会交互/三厢测试",
        en_name="Sociability / Three-Chamber",
        candidate_templates_hint=[
            "Sociability-AllZones",
            "Sociability-NoZones",
        ],
    ),
    Experiment(
        slug="shoaling",
        cn_name="斑马鱼鱼群行为",
        en_name="Shoaling (Zebrafish Group Behavior)",
        candidate_templates_hint=[
            "OpenFieldCircle-NoZones-Fish",
            "AquariumTrack3D",
            "DanioVision DVOC 004x-96w-circ",
        ],
        notes="多鱼同竞技场，需要 IID/NND/polarity 指标。已有 ethoinsight/templates/shoaling.py",
    ),
    Experiment(
        slug="aquatic_open_field",
        cn_name="鱼旷场（单鱼）",
        en_name="Aquatic Open Field (Single Fish)",
        candidate_templates_hint=[
            "OpenFieldCircle-NoZones-Fish",
            "AquariumTrack3D",
        ],
        notes="与 shoaling 共享模板，但实验目的（探索 vs 群体）不同。Gate 1 第三步要分清",
    ),
    Experiment(
        slug="phenotyper_homecage",
        cn_name="PhenoTyper 居家观察",
        en_name="PhenoTyper Home-Cage Observation",
        candidate_templates_hint=[
            "PhenoTyper-AllZonesMice",
            "PhenoTyper-AllZonesRatOther",
            "PhenoTyper-FeedingShelterMice",
            "PhenoTyper-NoZones",
            "PhenoTyper-Quad-AllZonesMice",
            "PhenoTyper-16x-AllZonesMice",
        ],
        notes="长期居家行为，不是短时 trial。有大量阵列变体",
    ),
    Experiment(
        slug="active_avoidance",
        cn_name="主动回避",
        en_name="Active Avoidance",
        candidate_templates_hint=[
            "UgoBasileActiveAvoidance",
        ],
    ),
    Experiment(
        slug="fear_conditioning",
        cn_name="恐惧条件化",
        en_name="Fear Conditioning",
        candidate_templates_hint=[
            "UgoBasileFCS-1cubicle",
            "UgoBasileFCS-4cubicles",
        ],
    ),
    Experiment(
        slug="insect_open_field",
        cn_name="昆虫旷场",
        en_name="Insect Open Field",
        candidate_templates_hint=[
            "OpenFieldCircle-NoZones-Insects",
            "OpenFieldRectangle-NoZonesFishInsects",
        ],
        notes="昆虫专用。不知道国内有多少需求，同事可以判断要不要保留",
    ),
    Experiment(
        slug="3d_swimming",
        cn_name="3D 游泳行为",
        en_name="3D Swimming Behavior",
        candidate_templates_hint=[
            "AquariumTrack3D",
            "FlightChamberTrack3D",
        ],
        notes="3D 跟踪，可能与 shoaling-3D 重叠",
    ),
]


def render_experiment_md(e: Experiment) -> str:
    lines: list[str] = []
    lines.append(f"# 实验：{e.cn_name} ({e.en_name})")
    lines.append("")
    lines.append(f"**slug**：`{e.slug}` （这是 paradigm key，agent 内部用）")
    lines.append("")
    lines.append("> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。")
    lines.append("")

    if e.notes:
        lines.append(f"<!-- 起草备注（同事可删除）：{e.notes} -->")
        lines.append("")

    lines.append("## 🟡 一句话定义（待补充）")
    lines.append("")
    lines.append("<!-- 例：「测试啮齿动物在新环境中的探索-焦虑权衡」 -->")
    lines.append("")

    lines.append("## 🟡 适用模板（按推荐顺序 + 取舍说明）")
    lines.append("")
    lines.append("候选模板（起草时根据目录名猜的，**同事必须 review 删/改/补**）：")
    lines.append("")
    for t in e.candidate_templates_hint:
        lines.append(f"- `{t}` —— 取舍：（待补充）")
    lines.append("")
    lines.append("<!-- 同事可自由增删。理想格式：")
    lines.append("- `OpenFieldCircle-NoZones-Fish` — 推荐 / 优势 / 局限")
    lines.append("- `AquariumTrack3D` — 何时用，何时别用")
    lines.append("-->")
    lines.append("")

    lines.append("## 🟡 必须计算的指标（待补充）")
    lines.append("")
    lines.append("<!-- 例（shoaling）：mean IID（鱼间距）、mean NND（最近邻距）、polarity（队列度） -->")
    lines.append("")
    lines.append("- ")
    lines.append("")

    lines.append("## 🟡 常见脱险点 / 数据质量风险（待补充）")
    lines.append("")
    lines.append("<!-- 例：「样本量小于 n=8 / 组时统计功效不足」「视频抖动会污染 IID 计算」 -->")
    lines.append("")
    lines.append("- ")
    lines.append("")

    lines.append("## 🟡 报告解读语言（待补充）")
    lines.append("")
    lines.append("<!-- 用什么名词、什么公式表达、避免什么常见误读 -->")
    lines.append("")

    lines.append("## 🟡 关键参考文献（待补充）")
    lines.append("")
    lines.append("- ")
    lines.append("")

    lines.append("## 🟡 与其他实验的区分（待补充）")
    lines.append("")
    lines.append("<!-- 例（shoaling vs aquatic_open_field）：「都用 OpenFieldCircle-NoZones-Fish")
    lines.append("但 shoaling 是多鱼同时记录测群体度，aquatic_open_field 是单鱼测探索-中心回避」 -->")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for e in EXPERIMENTS:
        path = OUT_DIR / f"{e.slug}.md"
        path.write_text(render_experiment_md(e), encoding="utf-8")
        print(f"wrote {path}")
    print(f"\ntotal: {len(EXPERIMENTS)} experiments")


if __name__ == "__main__":
    main()
