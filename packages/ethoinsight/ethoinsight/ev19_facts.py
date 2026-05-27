"""EV19 模板事实表 + 范式兼容性映射。

数据来源：docs/review-packages/2026-04-29-ev19-templates/_facts.json（自动抽取自 EV19 demodata）。
模块在 import 时把 JSON 加载到内存，作为只读字典提供给：
- set_experiment_paradigm 工具的白名单校验
- Ev19TemplateGuardrailProvider 的检查逻辑
- agent skill 中范式 → 默认变体降级
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path

# 包内自包含路径（与 ev19_facts.py 同目录）。
# 不依赖仓库根目录结构，Docker 容器内 COPY backend/ 时也能正常解析。
# _facts.json 由 docs/review-packages/2026-04-29-ev19-templates/_facts.json 同步过来。
_FACTS_JSON_PATH = Path(__file__).parent / "_facts.json"


@lru_cache(maxsize=1)
def _load_facts() -> dict:
    """Load _facts.json from canonical location. Cached after first call."""
    return json.loads(_FACTS_JSON_PATH.read_text(encoding="utf-8"))


def _build_variant_index() -> dict[str, dict]:
    data = _load_facts()
    return {v["template_id"]: v for v in data["variants"]}


# 公开常量 ---------------------------------------------------------------------

EV19_VARIANTS: dict[str, dict] = _build_variant_index()
"""62 变体的字典（template_id → 完整 facts 记录）。只读。"""

EV19_CATEGORIES: set[str] = {v["category"] for v in EV19_VARIANTS.values()}
"""20 大类的集合。"""

# paradigm_key → 推荐变体列表（手工维护；行为学同事 PR 后会扩展）
# 顺序很重要：第一个是该范式的默认变体（用于 agent 反问失败时降级）
EV19_TEMPLATE_PARADIGM_MAP: dict[str, list[str]] = {
    # 焦虑样行为（PRD MVP 4 个）
    "epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"],
    "open_field": [
        "OpenFieldRectangle-AllZones",
        "OpenFieldRectangle-NoZones",
        "OpenFieldCircle-AllZones",
        "OpenFieldCircle-NoZones-Rodents-Other",
    ],
    "zero_maze": ["ZeroMaze-AllZones", "ZeroMaze-NoZones"],
    "light_dark_box": [
        # LDB 在 EV19 表里没有独立大类，先用矩形 OFT 子集兜底
        # 等行为学同事 PR 后修正
        "OpenFieldRectangle-Subdivided2x2",
        "OpenFieldRectangle-AllZones",
    ],
    # 抑郁样行为（PRD MVP 2 个）
    "tail_suspension": ["NoTemplate"],  # TST 不需要 zone，仅活动度
    "forced_swim": ["PorsoltCylinder-AllZones", "PorsoltCylinder-NoZones"],
    # 其他范式 — 知识保留, 但代码层暂未实现 (SUPPORTED_PARADIGMS_V01 之外)
    "novel_object": [
        "OpenFieldCircle-NovObjZones",
        "OpenFieldRectangle-NovObjZones",
    ],
    "y_maze": ["Y-Maze-AllZones", "Y-Maze-NoZones"],
    "barnes_maze": ["BarnesMaze-20Holes", "BarnesMaze-NoZones"],
    "morris_water_maze": ["MWM-AllZones", "MWM-AFewZones", "MWM-NoZones"],
    "sociability": ["Sociability-AllZones", "Sociability-NoZones"],
    "radial_arm_maze": ["Radial-8-arm-AllZones", "Radial-8-arm-NoZones"],
}


# v0.1 实际有代码层实现的 paradigm_key 白名单 (catalog/<paradigm>.yaml 存在 +
# metrics/<paradigm>.py 存在 + scripts/<paradigm>/ 存在)。其他 paradigm_key 仍可识别
# 但 agent 应明示「暂不支持」, 见 identify_ev19_template_tool 主流程。
SUPPORTED_PARADIGMS_V01: frozenset[str] = frozenset({
    "epm",
    "open_field",
    "zero_maze",
    "light_dark_box",
    "forced_swim",
})


# 公开函数 ---------------------------------------------------------------------


def is_valid_ev19_template(template_id: str) -> bool:
    """Check if a template_id is in the 62-variant whitelist."""
    return template_id in EV19_VARIANTS


def get_template_facts(template_id: str) -> dict | None:
    """Return full facts record for a template_id, or None if unknown."""
    return EV19_VARIANTS.get(template_id)


def suggest_nearby_templates(template_id: str, max_results: int = 3) -> list[str]:
    """Return up to max_results template IDs that are close matches (for typo correction)."""
    return difflib.get_close_matches(
        template_id, list(EV19_VARIANTS.keys()), n=max_results, cutoff=0.6
    )


def get_default_template_for_paradigm(paradigm_key: str) -> str | None:
    """Return the recommended default variant for a paradigm, or None if unknown."""
    candidates = EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key)
    if not candidates:
        return None
    return candidates[0]


def is_paradigm_template_compatible(paradigm_key: str, template_id: str) -> bool:
    """Check if template_id is in the recommended list for paradigm_key."""
    candidates = EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key, [])
    return template_id in candidates
