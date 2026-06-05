"""ethoinsight.utils — Column name normalization and paradigm detection.

Handles the mapping of Chinese/mixed EthoVision column names to standardized
English snake_case identifiers. Supports dynamic patterns like "In zone(...)"
and "分析区中(...)".
"""

from __future__ import annotations

import re

# ============================================================================
# Column name mapping: Chinese/mixed → English snake_case
# ============================================================================

# Exact match mapping for common EthoVision column names
COLUMN_MAP: dict[str, str] = {
    # Time columns
    "试用时间": "trial_time",
    "录制时间": "recording_time",
    # Spatial columns - center point
    "X 中心": "x_center",
    "Y 中心": "y_center",
    # Spatial columns - nose point
    "X 鼻点": "x_nose",
    "Y 鼻点": "y_nose",
    # Spatial columns - tail point
    "X 尾": "x_tail",
    "Y 尾": "y_tail",
    # Movement columns
    "移动距离": "distance_moved",
    "Velocity": "velocity",
    "速度": "velocity",
    "方向": "heading",
    # Morphology columns
    "区域": "body_area",
    "面积变化": "area_change",
    "伸长": "elongation",
    # Result columns
    "Result 1": "result_1",
    "结果 1": "result_1",
    "Result 2": "result_2",
    "结果 2": "result_2",
    # Social interaction columns
    "Train": "train",
}

# Patterns for dynamic column names (compiled regex)
_DYNAMIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "In zone(Open arms /中心点)" → "in_zone_open_arms_center"
    (re.compile(r"^In zone\((.+)\)$", re.IGNORECASE), "in_zone"),
    # "分析区中(开放臂 /所有 中心点, 鼻尖)" → "in_zone_open_arms_all_center_nose"
    (re.compile(r"^分析区中\((.+)\)$"), "in_zone"),
    # "分析区中" (no parentheses) → "in_zone"
    (re.compile(r"^分析区中$"), "in_zone"),
    # "分析区轮替(中心点 /轮替)" → "zone_alternation_center_alternation"
    (re.compile(r"^分析区轮替\((.+)\)$"), "zone_alternation"),
    # "Entries to Open arms" → "entries_open_arms"
    (re.compile(r"^Entries to (.+)$", re.IGNORECASE), "entries"),
    # "Nose within object zone(Familiar object 1 /鼻尖)"
    (re.compile(r"^Nose within object zone\((.+)\)$", re.IGNORECASE), "nose_in_zone"),
    # "Distance to objects(Familiar object 1 /鼻尖)"
    (re.compile(r"^Distance to objects\((.+)\)$", re.IGNORECASE), "distance_to"),
    # "Head directed to object 1"
    (re.compile(r"^Head directed to (.+)$", re.IGNORECASE), "head_directed"),
    # "Side by side(Subject 1 /并排, 同向)"
    (re.compile(r"^Side by side\((.+)\)$", re.IGNORECASE), "side_by_side"),
    # "Proximity Subject 1(趋近 /鼻尖 /Subject 2 -中心点)"
    (re.compile(r"^Proximity (.+)$", re.IGNORECASE), "proximity"),
    # "JavaScript state - Subject 1 approaches Subject 2"
    (re.compile(r"^JavaScript state - (.+)$", re.IGNORECASE), "js_state"),
    # "JavaScript continuous - IID"
    (re.compile(r"^JavaScript continuous - (.+)$", re.IGNORECASE), "js_continuous"),
    # "When In open arms > 5 s"
    (re.compile(r"^When (.+)$", re.IGNORECASE), "when"),
    # "Body Length with JS"
    (re.compile(r"^Body Length with JS$", re.IGNORECASE), "body_length_js"),
    # "Test - Familiar vs Novel"
    (re.compile(r"^Test - (.+)$", re.IGNORECASE), "test"),
    # "Mobility continuous 身体填充 平均" (forced swim)
    (re.compile(r"^Mobility (.+)$", re.IGNORECASE), "mobility"),
]


def _slugify(text: str) -> str:
    """Convert arbitrary text to snake_case slug."""
    # Common Chinese → English substitutions
    subs = {
        "中心点": "center",
        "鼻尖": "nose",
        "鼻点": "nose",
        "尾根": "tail",
        "所有": "all",
        "开放臂": "open_arms",
        "封闭臂": "closed_arms",
        "闭合臂": "closed_arms",
        "中央区": "center",
        "中心区": "center",
        "轮替": "alternation",
        "最大轮替次数": "max_alternation",
        "直接再次逗留次数": "direct_revisit",
        "间接再次逗留次数": "indirect_revisit",
        "并排": "parallel",
        "同向": "same_dir",
        "反向": "opposite_dir",
        "趋近": "approach",
        "非趋近": "no_approach",
        "累计持续时间": "cumulative_duration",
        "频率": "frequency",
        "平均值": "mean",
        "标准偏差": "std",
        "身体填充": "body_fill",
        "狂躁": "highly_mobile",
        "活跃": "mobile",
        "静止": "immobile",
        "到墙壁距离": "distance_to_wall",
        # EthoVision 中文版 mobility/activity state 列名
        "活动状态": "mobility_state",
        "活动": "activity",
    }
    for cn, en in subs.items():
        text = text.replace(cn, en)

    # Replace separators with underscores
    text = re.sub(r"[/,\-\(\)\s]+", "_", text)
    # Remove quotes and other special chars
    text = re.sub(r"[\"'><=]+", "", text)
    # Collapse multiple underscores and strip
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


# Zone identifiers that appear as direct column prefixes when the user
# defined zones in EthoVision but didn't check "In zone" in export settings.
# "开放臂(开放臂1 / 中心点)" → "in_zone_open_arms_open_arms1_center"
_ZONE_PREFIX_TOKENS: set[str] = {
    "开放臂", "封闭臂", "闭合臂", "中央区",
    "Open arms", "Closed arms", "Center",
}


def _is_zone_prefix_column(name: str) -> bool:
    """Check if `name` starts with a known zone identifier followed by `(`."""
    for prefix in _ZONE_PREFIX_TOKENS:
        if name.startswith(prefix + "("):
            return True
    return False


def normalize_column_name(raw_name: str) -> str:
    """Normalize a raw EthoVision column name to English snake_case.

    Strategy:
    1. Strip quotes and whitespace
    2. Try exact match in COLUMN_MAP
    3. Try dynamic pattern matching
    4. Try zone-prefix detection (e.g. "开放臂(开放臂1 / 中心点)")
    5. Fall back to slugify
    """
    name = raw_name.strip().strip('"').strip()
    if not name:
        return "unnamed"

    # Exact match
    if name in COLUMN_MAP:
        return COLUMN_MAP[name]

    # Dynamic patterns
    for pattern, prefix in _DYNAMIC_PATTERNS:
        m = pattern.match(name)
        if m:
            if m.groups():
                suffix = _slugify(m.group(1))
                return f"{prefix}_{suffix}"
            return prefix

    # Zone-prefix detection: e.g. "开放臂(开放臂1 / 中心点)" → "in_zone_open_arms_open_arms1_center"
    if _is_zone_prefix_column(name):
        # Split at first '(': zone_prefix = "开放臂", detail = "开放臂1 / 中心点)"
        idx = name.index("(")
        zone_part = name[:idx]
        detail = name[idx:]  # includes the parentheses
        return f"in_zone_{_slugify(zone_part)}_{_slugify(detail)}"

    # Fallback: slugify the whole name
    return _slugify(name)


def normalize_columns(raw_names: list[str]) -> list[str]:
    """Normalize a list of column names, adding suffixes for duplicates."""
    result = []
    seen: dict[str, int] = {}
    for raw in raw_names:
        normalized = normalize_column_name(raw)
        if normalized in seen:
            seen[normalized] += 1
            normalized = f"{normalized}_{seen[normalized]}"
        else:
            seen[normalized] = 0
        result.append(normalized)
    return result


# ============================================================================
# Paradigm detection
# ============================================================================

PARADIGM_KEYWORDS: dict[str, list[str]] = {
    "shoaling": ["shoaling", "鱼群", "群体行为", "群体游动"],
    "open_field": ["open field", "旷场"],
    "epm": ["elevated plus maze", "plus maze", "高架十字"],
    "o_maze": ["zero maze", "o maze", "o迷宫", "零迷宫", "omaze"],
    "novel_object": ["novel object", "新物体"],
    "y_maze": ["y maze", "y迷宫"],
    "forced_swim": ["forced swim", "porsolt", "强迫游泳"],
    "light_dark": ["light dark", "明暗箱", "ldb"],
    "social_interaction": ["social interaction", "社会互动"],
    "morris_water_maze": ["morris water", "水迷宫"],
    "barnes_maze": ["barnes", "巴恩斯"],
    "three_chamber": ["three chamber", "三箱社交"],
    "tail_suspension": ["tail suspension", "悬尾"],
    "phenotyper": ["phenotyper"],
    "footprint": ["footprint", "足迹", "catwalk"],
    "novel_tank": ["novel tank", "新型水箱"],
    "cpp": ["conditioned place", "条件性位置"],
    "cognition_wall": ["cognition wall", "认知墙"],
    "fine_behavior": ["精细行为", "fine behavior"],
    "fear_conditioning": ["fear conditioning", "恐惧条件"],
    "t_maze": ["t maze", "t迷宫"],
}


def detect_paradigm(experiment_name: str) -> str | None:
    """Detect paradigm from experiment name string.

    Returns paradigm key (e.g. "forced_swim", "epm") or None if unrecognized.
    Note: detection only — paradigms not in SUPPORTED_PARADIGMS_V01 (e.g. shoaling,
    tail_suspension) can still be detected but code-layer pipelines are not implemented.
    """
    if not experiment_name:
        return None
    name_lower = experiment_name.lower()
    for paradigm, keywords in PARADIGM_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return paradigm
    return None
