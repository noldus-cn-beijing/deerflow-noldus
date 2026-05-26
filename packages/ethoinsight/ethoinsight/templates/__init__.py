"""Paradigm registry for behavioral data paradigms.

Public API:
- CATEGORIES: 7 major experiment categories
- PARADIGMS: 18 behavioral paradigms with zones, columns, subject type, and status
- list_categories(): return all 7 categories
- list_paradigms(status=, category=): filtered paradigm listing
- get_paradigm(name): single paradigm lookup
- verify_paradigm_columns(paradigm_name, file_path): column match check
"""

import csv
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    # Paradigm registry
    "CATEGORIES",
    "PARADIGMS",
    "list_categories",
    "list_paradigms",
    "get_paradigm",
    "verify_paradigm_columns",
]


# ===== Paradigm Categories (7 major groups for two-level ask_clarification) =====

CATEGORIES: list[dict] = [
    {
        "name": "open_field",
        "cn": "旷场及物体识别",
        "en": "Open Field & Object Recognition",
    },
    {"name": "anxiety", "cn": "焦虑迷宫", "en": "Anxiety Mazes"},
    {"name": "spatial_memory", "cn": "空间学习记忆迷宫", "en": "Spatial Memory Mazes"},
    {"name": "social", "cn": "社会交互与偏好", "en": "Social Interaction & Preference"},
    {"name": "depression", "cn": "抑郁/绝望", "en": "Depression & Despair"},
    {"name": "fear", "cn": "恐惧条件化", "en": "Fear Conditioning"},
    {"name": "zebrafish", "cn": "斑马鱼行为", "en": "Zebrafish Behavior"},
]

# ===== PARADIGMS Registry (18 paradigms, single source of truth) =====
# Based on analysis of 62 EV19 templates in demo-data/ev19 templates/ (2026-04-27).
# PhenoTyper, WellPlate, AquariumTrack3D, FlightChamberTrack3D are devices not paradigms.
#
# Fields:
#   cn / en — display names for ask_clarification options
#   category — high-level group (must match a CATEGORIES name)
#   subject — "rodent" | "fish" | "insect" | "other"
#   zones — meaningful analysis zones
#   expected_columns — columns expected in exported data
#   ev19_arena_templates — matching EV19 templateMetaData.xml <m_strArenaTemplate>
#   status — "ready" | "wip" | "planned"

PARADIGMS: dict[str, dict] = {
    # ===== 旷场及物体识别 (3) =====
    "open_field": {
        "cn": "旷场实验",
        "en": "Open Field Test",
        "category": "open_field",
        "subject": "rodent",
        "zones": ["center", "periphery", "corners"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "distance_moved",
            "time_center",
            "time_periphery",
        ],
        "ev19_arena_templates": ["Open field, round", "Open field, square"],
        "status": "planned",
    },
    "novel_object": {
        "cn": "新物体识别",
        "en": "Novel Object Recognition",
        "category": "open_field",
        "subject": "rodent",
        "zones": ["object_a", "object_b", "arena"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_novel",
            "time_familiar",
            "approaches_novel",
            "approaches_familiar",
        ],
        "ev19_arena_templates": ["Open field, round", "Open field, square"],
        "status": "planned",
    },
    "hole_board": {
        "cn": "孔板实验",
        "en": "Hole Board Test",
        "category": "open_field",
        "subject": "rodent",
        "zones": ["holes", "arena"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "head_dips",
            "total_entries",
            "time_near_holes",
        ],
        "ev19_arena_templates": [],
        "status": "planned",
    },
    # ===== 焦虑迷宫 (3) =====
    "epm": {
        "cn": "高架十字迷宫",
        "en": "Elevated Plus Maze",
        "category": "anxiety",
        "subject": "rodent",
        "zones": ["open_arm", "closed_arm", "center"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_open",
            "time_closed",
            "entries_open",
            "entries_closed",
        ],
        "ev19_arena_templates": ["Elevated plus maze"],
        "status": "planned",
    },
    "zero_maze": {
        "cn": "零迷宫",
        "en": "Zero Maze",
        "category": "anxiety",
        "subject": "rodent",
        "zones": ["open_quadrant", "closed_quadrant"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_open",
            "time_closed",
            "entries_open",
            "head_dips",
        ],
        "ev19_arena_templates": ["O-maze"],
        "status": "planned",
    },
    "light_dark": {
        "cn": "明暗箱",
        "en": "Light-Dark Box",
        "category": "anxiety",
        "subject": "rodent",
        "zones": ["light", "dark"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_light",
            "time_dark",
            "transitions",
        ],
        "ev19_arena_templates": [],
        "status": "planned",
    },
    # ===== 空间学习记忆迷宫 (6) =====
    "morris_water_maze": {
        "cn": "Morris 水迷宫",
        "en": "Morris Water Maze",
        "category": "spatial_memory",
        "subject": "rodent",
        "zones": ["target_quadrant", "other_quadrants", "platform"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "latency",
            "target_time",
            "distance_to_platform",
        ],
        "ev19_arena_templates": ["Morris water maze"],
        "status": "planned",
    },
    "barnes_maze": {
        "cn": "Barnes 迷宫",
        "en": "Barnes Maze",
        "category": "spatial_memory",
        "subject": "rodent",
        "zones": ["target_hole", "other_holes", "arena"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "latency",
            "target_time",
            "errors",
        ],
        "ev19_arena_templates": ["Barnes maze"],
        "status": "planned",
    },
    "radial_arm_maze": {
        "cn": "八臂迷宫",
        "en": "Radial Arm Maze",
        "category": "spatial_memory",
        "subject": "rodent",
        "zones": [
            "arm_1",
            "arm_2",
            "arm_3",
            "arm_4",
            "arm_5",
            "arm_6",
            "arm_7",
            "arm_8",
            "center",
        ],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "working_errors",
            "reference_errors",
            "total_arm_entries",
        ],
        "ev19_arena_templates": ["Radial 8-arm maze"],
        "status": "planned",
    },
    "y_maze": {
        "cn": "Y 迷宫",
        "en": "Y-Maze",
        "category": "spatial_memory",
        "subject": "rodent",
        "zones": ["arm_a", "arm_b", "arm_c", "center"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "alternations",
            "arm_entries_a",
            "arm_entries_b",
            "arm_entries_c",
        ],
        "ev19_arena_templates": ["Y-maze"],
        "status": "planned",
    },
    "t_maze": {
        "cn": "T 迷宫",
        "en": "T-Maze",
        "category": "spatial_memory",
        "subject": "rodent",
        "zones": ["left_arm", "right_arm", "start_arm", "center"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "alternation_rate",
            "left_entries",
            "right_entries",
            "latency",
        ],
        "ev19_arena_templates": ["T-maze"],
        "status": "planned",
    },
    "cross_maze": {
        "cn": "十字迷宫（鱼）",
        "en": "Cross Maze (Fish)",
        "category": "spatial_memory",
        "subject": "fish",
        "zones": ["start_box", "arms", "goal_zones", "center"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "arm_entries",
            "goal_latency",
            "correct_choices",
        ],
        "ev19_arena_templates": ["Cross maze"],
        "status": "planned",
    },
    # ===== 社会交互与偏好 (2) =====
    "social_interaction": {
        "cn": "社会交互",
        "en": "Social Interaction",
        "category": "social",
        "subject": "rodent",
        "zones": ["interaction_zone", "avoidance_zone"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_interaction",
            "time_avoidance",
            "approaches",
        ],
        "ev19_arena_templates": ["Sociability chamber"],
        "status": "planned",
    },
    "conditioned_place_preference": {
        "cn": "条件位置偏好",
        "en": "Conditioned Place Preference",
        "category": "social",
        "subject": "rodent",
        "zones": ["drug_paired", "vehicle_paired", "neutral"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "time_drug",
            "time_vehicle",
            "preference_score",
        ],
        "ev19_arena_templates": [],
        "status": "planned",
    },
    # ===== 抑郁/绝望 (2) =====
    "forced_swim": {
        "cn": "强迫游泳",
        "en": "Forced Swim Test",
        "category": "depression",
        "subject": "rodent",
        "zones": ["water", "bottom"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "immobility_time",
            "swimming_time",
            "climbing_time",
        ],
        "ev19_arena_templates": ["Porsolt cylinder"],
        "status": "planned",
    },
    "tail_suspension": {
        "cn": "悬尾实验",
        "en": "Tail Suspension Test",
        "category": "depression",
        "subject": "rodent",
        "zones": ["upper", "lower"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "immobility_time",
            "mobility_time",
        ],
        "ev19_arena_templates": [],
        "status": "planned",
    },
    # ===== 恐惧条件化 (1) =====
    "fear_conditioning": {
        "cn": "恐惧条件化/主动回避",
        "en": "Fear Conditioning / Active Avoidance",
        "category": "fear",
        "subject": "rodent",
        "zones": ["safe_zone", "shock_zone"],
        "expected_columns": [
            "X_center",
            "Y_center",
            "velocity",
            "freezing_time",
            "avoidance_latency",
            "escapes",
            "shocks_received",
        ],
        "ev19_arena_templates": [
            "Ugo Basile Active Avoidance",
            "Ugo Basile FCS 1 cubicle",
            "Ugo Basile FCS 4 cubicles",
        ],
        "status": "planned",
    },
}


def list_categories() -> list[dict]:
    """Return CATEGORIES list (7 major groups) for two-level ask_clarification."""
    return CATEGORIES


def list_paradigms(
    status: str | None = None, category: str | None = None
) -> list[dict]:
    """Return paradigms filtered by status and/or category.

    Each entry includes name, cn, en, zones, status, category, subject.
    """
    result = []
    for name, info in PARADIGMS.items():
        if status is not None and info.get("status") != status:
            continue
        if category is not None and info.get("category") != category:
            continue
        result.append(
            {
                "name": name,
                "cn": info["cn"],
                "en": info["en"],
                "zones": info["zones"],
                "status": info["status"],
                "category": info["category"],
                "subject": info["subject"],
            }
        )
    return result


def get_paradigm(name: str) -> dict | None:
    """Get full paradigm info dict, or None if not registered."""
    return PARADIGMS.get(name)


def verify_paradigm_columns(paradigm_name: str, file_path: str) -> dict | None:
    """Check whether a data file contains columns expected for the paradigm.

    Args:
        paradigm_name: Key in PARADIGMS registry.
        file_path: Path to a CSV/TSV file.

    Returns:
        {"match": bool, "expected": [...], "found": [...], "missing": [...]}
        Returns None if file does not exist (never raises for missing files).
        Raises ValueError if paradigm_name is not in PARADIGMS.
    """
    if paradigm_name not in PARADIGMS:
        raise ValueError(
            f"Unknown paradigm: {paradigm_name!r}. Available: {sorted(PARADIGMS)}"
        )

    path = Path(file_path)
    if not path.exists():
        logger.debug(
            "verify_paradigm_columns: file not found %s, returning None", file_path
        )
        return None

    expected = PARADIGMS[paradigm_name]["expected_columns"]

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return {
                    "match": False,
                    "expected": expected,
                    "found": [],
                    "missing": expected,
                }
    except Exception:
        logger.debug(
            "verify_paradigm_columns: failed to read %s", file_path, exc_info=True
        )
        return None

    header_lower = [h.strip().lower() for h in header]
    found_set = set(header_lower)

    missing = []
    for col in expected:
        if col.lower() not in found_set:
            missing.append(col)

    return {
        "match": len(missing) == 0,
        "expected": expected,
        "found": header,
        "missing": missing,
    }
