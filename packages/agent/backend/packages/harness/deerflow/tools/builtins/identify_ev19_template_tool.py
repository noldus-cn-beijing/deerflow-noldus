"""identify_ev19_template — 一步识别 EV19 模板变体，无需 lead 自己读 skill 文件。

lead agent 在调 ``set_experiment_paradigm`` 之前必须先调此工具。它内部完成：
- 从文件名/用户消息提取范式 hint
- 解析上传文件的列结构（判断 AllZones/NoZones/NovObjZones）
- 查 62 变体白名单 + 范式→模板映射
- 读取领域知识文件和模板细节文件（Python 侧，不计入 read_file 配额）
- 返回候选列表 + 证据 + 推荐 + 反问话术

这消除了 lead 在模板识别阶段反复 read_file 导致的 LoopDetectionMiddleware 硬限问题。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# ── 中文范式名 → paradigm_key 的模糊匹配表 ──────────────────────────
_PARADIGM_CN_HINTS: dict[str, str] = {
    "强迫游泳": "forced_swim",
    "悬尾": "tail_suspension",
    "高架十字": "epm",
    "十字高架": "epm",
    "旷场": "open_field",
    "开场": "open_field",
    "零迷宫": "zero_maze",
    "高架零迷宫": "zero_maze",
    "明暗箱": "light_dark_box",
    "黑白箱": "light_dark_box",
    "y迷宫": "y_maze",
    "y 迷宫": "y_maze",
    "t迷宫": "t_maze",
    "t 迷宫": "t_maze",
    "巴恩斯": "barnes_maze",
    "水迷宫": "morris_water_maze",
    "新物体": "novel_object",
    "新异物体": "novel_object",
    "社会交往": "sociability",
    "social": "sociability",
    "八臂": "radial_arm_maze",
    "8臂": "radial_arm_maze",
    "条件恐惧": "fear_conditioning",
    "恐惧条件": "fear_conditioning",
    "主动回避": "active_avoidance",
    "鱼群": "shoaling",
    "shoal": "shoaling",
    "3d游泳": "3d_swimming",
    "3d 游泳": "3d_swimming",
    "斑马鱼": "shoaling",
    "强迫游泳实验": "forced_swim",
    "fst": "forced_swim",
    "tst": "tail_suspension",
    "epm": "epm",
    "oft": "open_field",
    "ldb": "light_dark_box",
}

# ── 范式中文 label 权威表（文案 SSOT）──────────────────────────────────
# 已支持 + 部分常见未支持范式的 canonical_key → 中文展示名。
# unsupported 提示文案里「已实现 N 个范式: ...」的清单 label 从此派生，
# 不在文案里手举（守 SSOT：清单权威源是 SUPPORTED_PARADIGMS_V01，label 权威源是本表）。
_SUPPORTED_PARADIGM_CN_LABELS: dict[str, str] = {
    "forced_swim": "强迫游泳 (FST)",
    "tail_suspension": "悬尾实验 (TST)",
    "epm": "高架十字迷宫 (EPM)",
    "open_field": "旷场 (OFT)",
    "zero_maze": "零迷宫 (Zero Maze)",
    "light_dark_box": "明暗箱 (LDB)",
    "y_maze": "Y 迷宫",
    "t_maze": "T 迷宫",
    "barnes_maze": "巴恩斯迷宫",
    "morris_water_maze": "水迷宫 (MWM)",
    "novel_object": "新物体识别",
    "sociability": "社会交往",
    "radial_arm_maze": "八臂迷宫",
    "fear_conditioning": "条件恐惧",
    "active_avoidance": "主动回避",
    "shoaling": "鱼群行为",
    "3d_swimming": "3D 游泳",
}

# 未支持范式（鱼类 / 学习记忆 / 社会等）的中文 label，专用于 unsupported 路径。
# 与 _SUPPORTED_PARADIGM_CN_LABELS 重叠的 key（如 tail_suspension 在 v0.1 已支持）以
# SUPPORTED_PARADIGMS_V01 实际归属为准——本表只是给「确认不在 supported 集」的 key 翻译。
_UNSUPPORTED_PARADIGM_CN_LABELS: dict[str, str] = {
    "shoaling": "斑马鱼鱼群行为",
    "3d_swimming": "3D 游泳",
    "aquatic_open_field": "鱼类旷场",
    "cross_maze_fish": "鱼类十字迷宫",
    "morris_water_maze": "Morris 水迷宫",
    "barnes_maze": "巴恩斯迷宫",
    "y_maze": "Y 迷宫",
    "t_maze": "T 迷宫",
    "novel_object": "新物体识别",
    "sociability": "社会交往",
    "radial_arm_maze": "八臂迷宫",
    "fear_conditioning": "条件恐惧",
    "active_avoidance": "主动回避",
    "insect_open_field": "昆虫旷场",
    "phenotyper_homecage": "PhenoTyper 居家",
}


def _build_unsupported_result(
    *,
    paradigm_key: str,
    paradigm_label: str,
    supported_paradigms: set[str] | frozenset[str],
    supported_cn_labels: dict[str, str],
) -> dict:
    """构造 unsupported 路径的返回 dict（纯函数，可单测）。

    message / hint 里的范式数量与清单 label 全部从 supported_paradigms 派生，
    不含字面量数字或手举清单——范式集扩展时文案自动跟（守 SSOT）。

    Args:
      paradigm_key: 用户数据识别到但不支持的范式 canonical key
      paradigm_label: paradigm_key 的中文展示名
      supported_paradigms: 当前已支持范式集合（权威源 SUPPORTED_PARADIGMS_V01）
      supported_cn_labels: canonical_key → 中文展示名（用于生成清单 label）
    """
    supported_sorted = sorted(supported_paradigms)
    supported_labels = "、".join(
        supported_cn_labels.get(k, k) for k in supported_sorted
    )
    n = len(supported_sorted)
    return {
        "status": "unsupported",
        "paradigm_key": paradigm_key,
        "paradigm_label": paradigm_label,
        "supported_paradigms": supported_sorted,
        "message": (
            f"当前版本暂不支持「{paradigm_label}」范式分析。"
            f"v0.1 已实现 {n} 个范式: {supported_labels}。"
        ),
        "hint": (
            "请用 ask_clarification 告知用户当前不支持的范式名称, 并询问: "
            f"(a) 数据是否实际属于已支持的 {n} 个范式之一(用户用错名称); "
            "(b) 还是希望等版本更新后再分析。不要尝试跑流水线、不要伪装成相近范式、不要静默 fallback。"
        ),
    }

# 文件名中的范式简写模式 (regex → paradigm_key)
_FILENAME_PARADIGM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"forced.?swim|fst|porsolt", re.IGNORECASE), "forced_swim"),
    (re.compile(r"tail.?suspension|tst", re.IGNORECASE), "tail_suspension"),
    (re.compile(r"epm|plus.?maze|elevated.?plus", re.IGNORECASE), "epm"),
    (re.compile(r"open.?field|oft|旷场", re.IGNORECASE), "open_field"),
    (re.compile(r"zero.?maze|o.?maze", re.IGNORECASE), "zero_maze"),
    (re.compile(r"light.?dark|ldb|明暗", re.IGNORECASE), "light_dark_box"),
    (re.compile(r"y.?maze", re.IGNORECASE), "y_maze"),
    (re.compile(r"t.?maze", re.IGNORECASE), "t_maze"),
    (re.compile(r"barnes", re.IGNORECASE), "barnes_maze"),
    (re.compile(r"morris|水迷宫|mwm", re.IGNORECASE), "morris_water_maze"),
    (re.compile(r"novel.?object|新物体|新异物体", re.IGNORECASE), "novel_object"),
    (re.compile(r"sociability|社会交往|social", re.IGNORECASE), "sociability"),
    (re.compile(r"radial|八臂|8.?arm", re.IGNORECASE), "radial_arm_maze"),
    (re.compile(r"fear.?condition|条件恐惧|恐惧条件", re.IGNORECASE), "fear_conditioning"),
    (re.compile(r"active.?avoidance|主动回避", re.IGNORECASE), "active_avoidance"),
    (re.compile(r"shoal|鱼群|schooling", re.IGNORECASE), "shoaling"),
    (re.compile(r"3d.?swim", re.IGNORECASE), "3d_swimming"),
    (re.compile(r"强迫游泳", re.IGNORECASE), "forced_swim"),
    (re.compile(r"悬尾", re.IGNORECASE), "tail_suspension"),
]

# zone 列检测模式
_ZONE_COLUMN_PATTERN = re.compile(r"in zone\s*\(|in_zone", re.IGNORECASE)
_NOVOBJ_COLUMN_PATTERN = re.compile(r"nose within object zone|novel.?object", re.IGNORECASE)

# 疑似分析区归属列检测（窄白名单，2026-06-16 新增）。
# 用途：仅用于排除 NoZones（NoZones 定义=完全无任何区归属列），**不用于判定哪列是哪个区**
# ——哪列对哪个区仍由 column-confirmation skill 反问用户（守 feedback_oft_single_zone_must_ask_not_guess）。
# 命中真实场景：EPM 的 open/closed（dogfood 实证）、OFT 的 center/中心区/边缘区、
# Zero Maze 的 open/closed。这些是"列名非标准但确实划了区"的归属列。
# 排除 x_center/y_center（坐标列固定名，绝不会单叫 center；坐标列叫 *_center）。
# label 仅用于措辞展示（"疑似归属列 open、closed"），绝不在 filter 逻辑里做区归属推断。
_SUSPECT_ZONE_COLUMN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:^|[_\s])open(?:[_\s]|$)|open_?arms?", re.IGNORECASE), "开臂"),
    (re.compile(r"(?:^|[_\s])closed(?:[_\s]|$)|closed_?arms?", re.IGNORECASE), "闭臂"),
    (re.compile(r"(?:^|[_\s])cent(er|re)(?:[_\s]|$)", re.IGNORECASE), "中心区"),
    (re.compile(r"head.?dip", re.IGNORECASE), "head dip 区"),
    (re.compile(r"zone[_\s]*[a-z0-9]", re.IGNORECASE), "分析区"),
    (re.compile(r"开臂|闭臂|中心区|中央区|边缘区|周边区|外周区", re.IGNORECASE), "分析区"),
]
# 坐标列固定名（绝不可能是区归属列）——在疑似归属列检测时显式排除。
_COORD_COLUMN_PATTERN = re.compile(r"^[xy][_ ]?cent(er|re)$", re.IGNORECASE)

# 用户消息中的物种提示
_SUBJECT_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"大鼠|rat|sprague.?dawley|wistar|long.?evans", re.IGNORECASE), "rodent"),
    (re.compile(r"小鼠|mouse|mice|c57|balb", re.IGNORECASE), "rodent"),
    (re.compile(r"斑马鱼|zebrafish|danio", re.IGNORECASE), "fish"),
    (re.compile(r"果蝇|drosophila|昆虫|insect|fly", re.IGNORECASE), "insect"),
]


def _resolve_skills_ref_dir() -> Path:
    """Return host-side path to ethovision-paradigm-knowledge references/."""
    from deerflow.config import get_app_config

    skills_path = get_app_config().skills.get_skills_path()
    return skills_path / "custom" / "ethovision-paradigm-knowledge" / "references"


def _extract_paradigm_from_hints(user_message: str, uploaded_files: list[str]) -> str | None:
    """Extract paradigm_key from user message and filenames. Returns None if no match."""
    all_text = user_message
    for f in uploaded_files:
        all_text += " " + Path(f).name

    # Check Chinese hints first (more specific)
    for cn_hint, paradigm_key in _PARADIGM_CN_HINTS.items():
        if cn_hint.lower() in all_text.lower():
            return paradigm_key

    # Check filename patterns
    for pattern, paradigm_key in _FILENAME_PARADIGM_PATTERNS:
        if pattern.search(all_text):
            return paradigm_key

    return None


def _extract_subject(user_message: str) -> str | None:
    """Extract subject type from user message. Returns None if no match."""
    for pattern, subject in _SUBJECT_HINTS:
        if pattern.search(user_message):
            return subject
    return None


def _detect_zone_config(columns: list[str]) -> dict:
    """Detect zone configuration from data columns.

    Returns:
      {"has_zone_columns": bool,          # 标准 in_zone 列（精确）
       "has_novobj_columns": bool,        # novel object 列
       "has_suspect_zone_columns": bool,  # 疑似非标准归属列（窄白名单；仅排除 NoZones 用）
       "zone_columns": [...],
       "novobj_columns": [...],
       "suspect_columns": [...]}          # 疑似归属列的原始列名（仅措辞展示，不判区归属）

    疑似归属列仅在 ``has_zone_columns=False`` 时才扫——标准 in_zone 命中即无需疑似判定。
    """
    zone_cols = [c for c in columns if _ZONE_COLUMN_PATTERN.search(c)]
    novobj_cols = [c for c in columns if _NOVOBJ_COLUMN_PATTERN.search(c)]
    # 疑似归属列：非标准命名但可能是区归属（open/closed/center/中心区/...），排除坐标列。
    # 守 feedback_oft_single_zone_must_ask_not_guess：这里只标"疑似"，不判哪列是哪个区。
    suspect_cols: list[str] = []
    if not zone_cols:
        for c in columns:
            if _COORD_COLUMN_PATTERN.search(c):
                continue  # x_center/y_center 坐标列，绝非区归属
            if any(pat.search(c) for pat, _label in _SUSPECT_ZONE_COLUMN_PATTERNS):
                suspect_cols.append(c)
    return {
        "has_zone_columns": len(zone_cols) > 0,
        "has_novobj_columns": len(novobj_cols) > 0,
        "has_suspect_zone_columns": len(suspect_cols) > 0,
        "zone_columns": zone_cols,
        "novobj_columns": novobj_cols,
        "suspect_columns": suspect_cols,
    }


def _filter_candidates_by_zone(template_ids: list[str], zone_info: dict) -> list[str]:
    """Filter template candidates based on zone column evidence.

    - has_zone_columns（标准 in_zone）→ prefer AllZones/NovObjZones/FewZones/AFewZones variants
    - has_suspect_zone_columns（疑似非标准归属列，如 open/closed）→ 剔除 NoZones
      （NoZones 定义=完全无任何区归属列；疑似归属列存在即不满足该定义。守
      feedback_lead_inverts_fewzones_vs_nozones_by_column_name：有归属列绝不改判 NoZones）。
    - no zone columns at all（纯轨迹）→ prefer NoZones variants
    - has_novobj_columns → prefer NovObjZones variants
    """
    if not template_ids:
        return []

    has_zone = zone_info["has_zone_columns"]
    has_novobj = zone_info["has_novobj_columns"]
    has_suspect = zone_info.get("has_suspect_zone_columns", False)

    filtered = []
    for tid in template_ids:
        tid_lower = tid.lower()
        is_nozone = "nozone" in tid_lower or "notemplate" in tid_lower
        is_zone_variant = not is_nozone

        if has_novobj and "novobj" in tid_lower:
            filtered.append(tid)
        elif has_zone and is_zone_variant:
            filtered.append(tid)
        elif has_suspect and is_zone_variant:
            # 疑似归属列存在 → 划了区的变体，剔除 NoZones（定义层面，非猜测）。
            filtered.append(tid)
        elif not has_zone and not has_suspect and is_nozone:
            filtered.append(tid)
        elif not has_zone and not has_suspect and not has_novobj:
            # 纯轨迹、无任何疑似归属列 — keep everything, note in evidence
            filtered.append(tid)

    # If filtering eliminated everything, return original list (defer to lead)
    return filtered if filtered else list(template_ids)


def _read_markdown_section(path: Path) -> str | None:
    """Read a markdown file. Returns content or None if absent."""
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _extract_template_recommendations(by_experiment_md: str | None) -> list[str]:
    """Extract recommended template IDs from by-experiment markdown.

    Looks for lines like: ``- `PorsoltCylinder-NoZones` — **推荐首选**``
    """
    if not by_experiment_md:
        return []
    recs = []
    for match in re.finditer(r"`(PorsoltCylinder|PlusMaze|OpenField|ZeroMaze|MWM|BarnesMaze|Sociability|Radial-8-arm|T-Maze|Y-Maze|Cross Maze-Fish|WellPlate|PhenoTyper|AquariumTrack3D|DanioVision|FlightChamberTrack3D|NoTemplate|UgoBasile)[^`]*`", by_experiment_md):
        recs.append(match.group(0).strip("`"))
    return recs


def _build_clarification_question(
    paradigm_key: str,
    candidates: list[dict],
    evidence: dict,
) -> str:
    """Build a structured clarification question for the lead to ask the user."""
    if not candidates:
        return "无法确定您的实验范式。请确认您做的是什么实验？（如：强迫游泳、高架十字迷宫、旷场等）"

    paradigm_label = _SUPPORTED_PARADIGM_CN_LABELS.get(paradigm_key, paradigm_key)

    lines = ["我从您的数据看到："]
    if evidence.get("filename_hint"):
        lines.append(f"- 文件名含「{evidence['filename_hint']}」")
    if evidence.get("subject"):
        lines.append(f"- 实验对象：{evidence['subject']}")
    zone_info = evidence.get("zone_info", {})
    if zone_info.get("has_zone_columns"):
        lines.append("- 数据含标准 zone 列（区域标记）")
    elif zone_info.get("has_suspect_zone_columns"):
        # 非标准命名但疑似区归属列（open/closed/中心区 等）。如实呈现，明示非 NoZones，
        # 破除"无 zone 列"假陈述（2026-06-16 EPM dogfood：open/closed 被 _ZONE_COLUMN_PATTERN
        # 漏检 → 旧措辞误报"无 zone 列"→ lead 植入 NoZones 印象）。哪列对哪个区不在这里判，
        # 留给列语义对齐环节反问（守 feedback_oft_single_zone_must_ask_not_guess）。
        suspects = zone_info.get("suspect_columns") or []
        suspects_str = "、".join(suspects[:6]) if suspects else "若干非标准列"
        lines.append(
            f"- 未检测到标准 in_zone 列，但发现疑似分析区归属列：{suspects_str}。"
            "这些列是否为区归属列请在后续列语义对齐环节确认；若属实则属于划了区的变体"
            "（Few/AllZones），非 NoZones。"
        )
    else:
        lines.append("- 数据无 zone 列（仅坐标 + mobility state）")

    lines.append("")
    lines.append(f"实验类型：{paradigm_label}")
    lines.append("")
    lines.append("您用的是哪个 EV19 模板？")

    labels = "ABCDEFGHIJ"
    for i, c in enumerate(candidates):
        label = labels[i] if i < len(labels) else f"({i+1})"
        recommended = "（推荐）" if c.get("recommended") else ""
        why = c.get("why", "")
        suffix = f" — {why}" if why else ""
        lines.append(f"{label}. **{c['template_id']}**{recommended}{suffix}")

    first_rec = next((c for c in candidates if c.get("recommended")), candidates[0] if candidates else None)
    if first_rec:
        lines.append("")
        lines.append(f"如果不确定，选 A（{first_rec['template_id']}）。")

    return "\n".join(lines)


# ── 错误码体系 ──────────────────────────────────────────────────────

_ERROR_HINTS: dict[str, str] = {
    "no_files_provided": "uploaded_files 为空。把当前 <uploaded_files> 中所有数据文件传进来。",
    "workspace_missing": "thread_data.workspace_path 未设置——基础设施 bug。present_files 把错误信息呈现给用户。",
    "file_not_found": "数据文件不存在。用 ask_clarification 让用户重新上传。",
    "parse_failed": "数据文件解析失败。用 ask_clarification 让用户确认文件格式。",
    "format_unrecognized": "文件不是 EthoVision XT 导出格式。用 ask_clarification 让用户确认导出方式。",
}


def _write_template_candidates(workspace: str, data: dict) -> None:
    """Persist identify_ev19_template result so guardrail can enforce confirmation."""
    import json
    from pathlib import Path

    p = Path(workspace) / "template_candidates.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # Non-critical; guardrail will skip if file missing


def _error_result(code: str, message: str, failed_file: str | None = None) -> dict:
    hint = _ERROR_HINTS.get(code, "未知错误，请联系开发者。")
    result: dict = {"status": "error", "error_code": code, "message": message, "hint": hint}
    if failed_file:
        result["failed_file"] = failed_file
    return result


# ── 主工具函数 ──────────────────────────────────────────────────────


@tool("identify_ev19_template", parse_docstring=True)
def identify_ev19_template_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_files: list[str],
    user_message: str,
) -> dict:
    """识别用户实验对应的 EV19 模板变体。

    在调用 set_experiment_paradigm 之前必须先调此工具。它会自动完成：
    - 从文件名和用户消息中提取范式 hint（中文/英文）
    - 解析上传文件的列结构（判断 AllZones/NoZones/NovObjZones）
    - 查 62 变体白名单，交叉排除不匹配的候选项
    - 读取领域知识文件（by-experiment / by-template markdown）
    - 返回候选列表 + 证据 + 反问话术

    Args:
      uploaded_files: 用户上传文件的虚拟路径列表如 ["/mnt/user-data/uploads/file1.txt", ...]
      user_message: 用户的原始消息文本（用于提取范式关键词和物种 hint）

    Returns:
      status="ok" (候选唯一):
        {"status": "ok",
         "paradigm_key": "forced_swim",
         "ev19_template": "PorsoltCylinder-NoZones",
         "candidates": [{"template_id": "PorsoltCylinder-NoZones", "recommended": true, ...}],
         "evidence": {...},
         "domain_summary": "..."}

      status="ambiguous" (2-3 候选):
        {"status": "ambiguous",
         "paradigm_key": "forced_swim",
         "candidates": [{...}, {...}],
         "evidence": {...},
         "clarification_question": "您用的是哪个 EV19 模板？\\nA. ...\\nB. ..."}

      status="unknown" (无候选):
        {"status": "unknown",
         "evidence": {...},
         "clarification_question": "无法确定实验范式，请确认？"}

      status="unsupported" (v0.1 暂不支持该范式 — 鱼类 / MWM / Barnes 等未在
              SUPPORTED_PARADIGMS_V01 里的范式；当前清单见 message / supported_paradigms):
        {"status": "unsupported",
         "paradigm_key": "shoaling",
         "paradigm_label": "斑马鱼鱼群行为",
         "supported_paradigms": ["<sorted(SUPPORTED_PARADIGMS_V01)>"],
         "message": "当前版本暂不支持「斑马鱼鱼群行为」范式...",
         "hint": "请用 ask_clarification 告知用户..."}

      status="error":
        {"status": "error", "error_code": "...", "message": "...", "hint": "..."}
    """
    # Step 0: validate inputs
    if not uploaded_files:
        return _error_result("no_files_provided", "uploaded_files is empty")

    # Step 1: resolve workspace and skills references directory
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return _error_result("workspace_missing", "thread_data.workspace_path is not set")
    real_workspace = thread_data["workspace_path"]

    # Lazy import for circular dependency avoidance
    from deerflow.sandbox.tools import replace_virtual_path

    refs_dir = _resolve_skills_ref_dir()

    # Step 2: extract paradigm hints from user message + filenames
    paradigm_key = _extract_paradigm_from_hints(user_message, uploaded_files)
    subject_hint = _extract_subject(user_message)

    # Step 2.5: v0.1 范围公告 — 识别到不支持的范式 keyword 时直接 unsupported 返回,
    # 不进入后续 zone 解析 / 候选搜索流水线。
    # Why: 代码层只为 SUPPORTED_PARADIGMS_V01 里的范式实现了 catalog/metrics/scripts,
    # 其他 paradigm_key 即使识别成功也会在 code-executor 阶段炸。提前在这里告知用户更友好。
    if paradigm_key is not None:
        from ethoinsight.ev19_facts import SUPPORTED_PARADIGMS_V01

        if paradigm_key not in SUPPORTED_PARADIGMS_V01:
            paradigm_label = _UNSUPPORTED_PARADIGM_CN_LABELS.get(paradigm_key, paradigm_key)
            return _build_unsupported_result(
                paradigm_key=paradigm_key,
                paradigm_label=paradigm_label,
                supported_paradigms=SUPPORTED_PARADIGMS_V01,
                supported_cn_labels=_SUPPORTED_PARADIGM_CN_LABELS,
            )

    # build filename-derived hint for evidence
    filename_hint = None
    for f in uploaded_files:
        fname = Path(f).name
        for pattern, pkey in _FILENAME_PARADIGM_PATTERNS:
            if pattern.search(fname):
                filename_hint = pattern.search(fname).group(0)
                break
        if filename_hint:
            break

    # Step 3: parse first uploaded file → columns + zone detection
    first_file = uploaded_files[0]
    real_first = replace_virtual_path(first_file, thread_data)
    if not Path(real_first).exists():
        return _error_result("file_not_found", f"File not found: {first_file}", failed_file=first_file)

    # Validate EthoVision format
    from ethoinsight.parse._core import detect_ethovision, parse_header

    if not detect_ethovision(real_first):
        return _error_result("format_unrecognized", f"Not an EthoVision export: {first_file}", failed_file=first_file)

    try:
        header = parse_header(real_first)
    except Exception as e:
        logger.warning("parse_header failed for %s: %s", first_file, e)
        return _error_result("parse_failed", f"Failed to parse header: {e}", failed_file=first_file)

    columns = header.get("columns", [])
    zone_info = _detect_zone_config(columns)

    # Step 3.5: 遍历全部 uploaded_files 提取每个文件的分组字段（EV19 头自带 Treatment/Group/...）。
    # 让 lead 一次拿到全部分组，无需逐个 inspect_uploaded_file 试探边界。
    # 只读 header（parse_header ~0.09s/文件，calamine 已生效），不读 trajectory。
    # 性能：28 文件 ≈2.6s，可接受。若未来文件数 >100 再考虑并行（TODO）。
    from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields

    per_file_grouping: dict[str, dict[str, str]] = {}
    for f in uploaded_files:
        real_f = replace_virtual_path(f, thread_data)
        if not Path(real_f).exists():
            continue  # 缺文件不阻断模板识别；记空即可
        try:
            h = parse_header(real_f)
            gf = extract_grouping_fields(h.get("raw_metadata", {}) or {})
            if gf:
                per_file_grouping[Path(f).name] = gf  # 用文件名做 key（lead 按文件名对照分组）
        except Exception:
            continue  # 单文件解析失败不阻断（防御性，同 inspect 风格）

    # Step 4: look up candidates from ev19_facts
    from ethoinsight.ev19_facts import EV19_TEMPLATE_PARADIGM_MAP

    candidate_ids: list[str] = []
    if paradigm_key:
        candidate_ids = list(EV19_TEMPLATE_PARADIGM_MAP.get(paradigm_key, []))

    # If no candidates from paradigm, fall back to all templates matching filename patterns
    if not candidate_ids:
        # Try to find matching category from filename
        for pattern, pkey in _FILENAME_PARADIGM_PATTERNS:
            if pattern.search(user_message) or any(pattern.search(Path(f).name) for f in uploaded_files):
                candidate_ids = list(EV19_TEMPLATE_PARADIGM_MAP.get(pkey, []))
                if candidate_ids:
                    paradigm_key = pkey
                    break

    # Step 5: filter candidates by zone evidence
    filtered_ids = _filter_candidates_by_zone(candidate_ids, zone_info) if candidate_ids else []

    # Step 6: read domain knowledge files
    by_experiment_md = None
    if paradigm_key:
        by_experiment_path = refs_dir / "by-experiment" / f"{paradigm_key}.md"
        by_experiment_md = _read_markdown_section(by_experiment_path)

    # Extract recommended templates from experiment markdown
    experiment_recs = _extract_template_recommendations(by_experiment_md)

    # Step 7: read by-template files for each candidate to get "why" descriptions
    candidates: list[dict] = []
    target_ids = filtered_ids if filtered_ids else candidate_ids

    for tid in target_ids:
        from ethoinsight.ev19_facts import get_template_facts

        facts = get_template_facts(tid)
        category = facts.get("category", tid.split("-")[0]) if facts else tid.split("-")[0]

        # Read by-template markdown for variant details
        by_template_path = refs_dir / "by-template" / f"{category}.md"
        by_template_md = _read_markdown_section(by_template_path)

        # Extract the "why" for this specific variant
        why = ""
        if by_template_md and tid in by_template_md:
            # Try to extract the recommended scenario line
            rec_match = re.search(
                rf"## 变体：{re.escape(tid)}.*?### 🟡 推荐的实验场景\s*\n\s*(.+?)(?:\n|$)",
                by_template_md, re.DOTALL,
            )
            if rec_match:
                why = rec_match.group(1).strip()
            else:
                # Fallback: use the variant difference description
                diff_match = re.search(
                    rf"## 变体：{re.escape(tid)}.*?### 🟡 这个变体相对其他变体的核心差异\s*\n\s*(.+?)(?:\n|$)",
                    by_template_md, re.DOTALL,
                )
                if diff_match:
                    why = diff_match.group(1).strip()

        zone_config = facts.get("zone_config", "") if facts else ""
        is_recommended = tid in experiment_recs or (tid == target_ids[0] if target_ids else False)

        candidates.append({
            "template_id": tid,
            "category": category,
            "zone_config": zone_config,
            "recommended": is_recommended,
            "why": why,
        })

    # Step 8: build evidence
    # zone_info 直接整体引用 _detect_zone_config 的产物（6 字段），与落盘
    # template_candidates.json 的 zone_info 同源同构——内存 evidence 与落盘双写一致
    # （spec 2026-06-22 §6.4）：lead 当次用内存 evidence，evidence 被 summarize
    # 截断后用落盘，两者不会漂移。
    evidence = {
        "paradigm_key": paradigm_key,
        "filename_hint": filename_hint,
        "subject": subject_hint,
        "zone_info": zone_info,
        "columns": columns[:20],  # first 20 columns for reference
    }

    # Step 9: determine status and build response
    if not candidates:
        # Write ok status to overwrite any stale ambiguous state from a prior run.
        # zone_info 一并沉淀（spec 2026-06-22）：花 parse_header 成本检测出的列信号
        # 不只在内存 evidence 里，落盘 template_candidates.json 也要带，供 lead 后续
        # 带依据反问、guardrail、将来下游在 ToolMessage evidence 被截断后使用。
        _write_template_candidates(real_workspace, {
            "status": "unknown",
            "paradigm_key": paradigm_key,
            "zone_info": zone_info,
        })
        return {
            "status": "unknown",
            "evidence": evidence,
            "clarification_question": _build_clarification_question(paradigm_key or "unknown", [], evidence),
        }

    if len(candidates) == 1:
        # Unique candidate — can proceed directly
        # Write ok status to overwrite any stale ambiguous state from a prior run
        _write_template_candidates(real_workspace, {
            "status": "ok",
            "paradigm_key": paradigm_key,
            "ev19_template": candidates[0]["template_id"],
            "zone_info": zone_info,
        })
        domain_summary = ""
        if by_experiment_md:
            # Extract the one-liner definition
            def_match = re.search(r"## 🟡 一句话定义\s*\n\s*(.+?)(?:\n|$)", by_experiment_md)
            if def_match:
                domain_summary = def_match.group(1).strip()

        return {
            "status": "ok",
            "paradigm_key": paradigm_key,
            "ev19_template": candidates[0]["template_id"],
            "candidates": candidates,
            "evidence": evidence,
            "domain_summary": domain_summary,
            "per_file_grouping": per_file_grouping,
        }

    # 2-3 candidates — need user clarification
    # Write candidates to workspace so guardrail can enforce user-confirmation requirement
    _write_template_candidates(real_workspace, {
        "status": "ambiguous",
        "paradigm_key": paradigm_key,
        "candidates": candidates,
        "clarification_question": _build_clarification_question(paradigm_key or "unknown", candidates, evidence),
        "zone_info": zone_info,
    })

    return {
        "status": "ambiguous",
        "paradigm_key": paradigm_key,
        "candidates": candidates,
        "evidence": evidence,
        "clarification_question": _build_clarification_question(paradigm_key or "unknown", candidates, evidence),
        "per_file_grouping": per_file_grouping,
    }
