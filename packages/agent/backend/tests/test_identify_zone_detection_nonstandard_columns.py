"""Spec 2026-06-15 §7 第二条 — identify 工具把含归属列的数据误报为 NoZones 修复测试。

根因（已用真实 EPM 数据坐实，见 plan cheeky-singing-puddle.md）：
  - identify_ev19_template_tool.py:91 ``_ZONE_COLUMN_PATTERN`` 只匹配标准 EthoVision
    列名（``in zone(...)`` / ``in_zone``），**漏检 open/closed 等非标准归属列**。
  - → ``_detect_zone_config`` 返回 ``has_zone_columns=False``。
  - → ``_filter_candidates_by_zone`` 走 ``not has_zone`` 兜底，**保留全部候选含 NoZones**。
  - → ``_build_clarification_question`` 说"数据无 zone 列（仅坐标 + mobility state）"
    ——**假陈述**（数据明明有 open/closed 归属列）。
  - lead 在看到 prompt 铁律前就被 candidates+措辞植入 NoZones 印象 → 后续误判切模板。

修复（守 ``feedback_oft_single_zone_must_ask_not_guess``：只判定"有疑似归属列→非 NoZones"，
**不猜哪列是哪个区**，留给 column-confirmation skill）：
  - 新增窄白名单 ``_SUSPECT_ZONE_COLUMN_PATTERNS`` 检测疑似归属列（open/closed/
    open_arms/closed_arms/center/centre/中心区/边缘区/head_dip/zone_X，排除 x_center/y_center）。
  - ``_detect_zone_config`` 增 ``has_suspect_zone_columns``/``suspect_columns``。
  - ``_filter_candidates_by_zone``：有疑似归属列时剔除 NoZones 候选。
  - 措辞三态：有标准 zone 列 / 有疑似归属列(明示非 NoZones) / 纯轨迹(保留"无 zone 列")。

守 ``feedback_worktree_shares_main_venv_editable_link``：用 importlib 加载 worktree 源
（editable deerflow 指向主仓，直接 import 测主仓代码 = 假绿）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

# Load the identify tool module fresh from THIS worktree's source.
# editable-link 铁律：主仓 venv 的 deerflow editable 指向主仓源码，直接
# `from deerflow...` 会测主仓、worktree 改动假绿。必须 importlib 指向 worktree 文件。
_TOOL_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "harness"
    / "deerflow"
    / "tools"
    / "builtins"
    / "identify_ev19_template_tool.py"
)


def _load_identify_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.tools.builtins.identify_ev19_template_tool_real",
        _TOOL_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_MOD = _load_identify_module()
_detect_zone_config = _MOD._detect_zone_config
_filter_candidates_by_zone = _MOD._filter_candidates_by_zone
_build_clarification_question = _MOD._build_clarification_question
identify_ev19_template_tool = _MOD.identify_ev19_template_tool

# Module path strings for patching (patch on the worktree-loaded module object).
_SANDBOX_TOOLS = "deerflow.sandbox.tools"
_PARSE_CORE = "ethoinsight.parse._core"
_EV19_FACTS = "ethoinsight.ev19_facts"


# Real EPM column set from /home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28/
# (verified: 12 columns, open/closed are 0/1 互斥 归属列, result_1 恒 1).
_EPM_REAL_COLUMNS = [
    "trial_time",
    "recording_time",
    "x_center",
    "y_center",
    "area",
    "areachange",
    "elongation",
    "distance_moved",
    "velocity",
    "open",
    "closed",
    "result_1",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(workspace_path: str) -> MagicMock:
    runtime = MagicMock()
    runtime.state = {
        "thread_data": {
            "workspace_path": workspace_path,
            "uploads_path": "/mnt/user-data/uploads",
            "outputs_path": "/mnt/user-data/outputs",
        }
    }
    return runtime


def _header_with_columns(columns: list[str]) -> dict:
    return {
        "columns": columns,
        "raw_metadata": {},
        "experiment": "Test",
        "trial_name": "Trial 1",
        "arena": "Arena 1",
        "subject": "Subject 1",
    }


def _nozones_in(candidates: list[dict]) -> bool:
    return any("nozone" in c["template_id"].lower() for c in candidates)


# ---------------------------------------------------------------------------
# 1. RED anchor: real EPM columns (open/closed) → must NOT keep NoZones
# ---------------------------------------------------------------------------


class TestSuspectZoneDetection:
    def test_detect_suspect_zone_columns_on_real_epm(self):
        """锚点：真实 EPM 列（open/closed）→ has_suspect_zone_columns=True，suspect 含 open/closed。"""
        z = _detect_zone_config(_EPM_REAL_COLUMNS)
        assert z["has_zone_columns"] is False, "open/closed 不是标准 in_zone 列"
        assert z["has_suspect_zone_columns"] is True, (
            "open/closed 是疑似归属列，应检出（修复后绿；修复前无此字段→AttributeError/False 红）"
        )
        suspects = z.get("suspect_columns", [])
        assert "open" in suspects and "closed" in suspects

    def test_filter_removes_nozones_when_suspect_zone_present(self):
        """锚点：有疑似归属列 → NoZones 被剔除，zone 变体保留。"""
        zone_info = _detect_zone_config(_EPM_REAL_COLUMNS)
        candidates = ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]
        filtered = _filter_candidates_by_zone(candidates, zone_info)
        assert "PlusMaze-NoZones" not in filtered, (
            f"有疑似归属列时 NoZones 应被剔除，实际保留: {filtered}"
        )
        assert "PlusMaze-FewZones" in filtered
        assert "PlusMaze-AllZones" in filtered

    def test_clarification_does_not_claim_no_zone_on_real_epm(self):
        """锚点：真实 EPM 列 → 措辞不该说"数据无 zone 列"，应明示疑似归属列 + 非 NoZones。"""
        zone_info = _detect_zone_config(_EPM_REAL_COLUMNS)
        candidates = [
            {"template_id": "PlusMaze-AllZones", "recommended": True, "why": ""},
            {"template_id": "PlusMaze-FewZones", "recommended": False, "why": ""},
        ]
        evidence = {
            "filename_hint": "EPM",
            "subject": "rodent",
            "zone_info": {
                "has_zone_columns": zone_info["has_zone_columns"],
                "has_novobj_columns": zone_info["has_novobj_columns"],
                "has_suspect_zone_columns": zone_info.get("has_suspect_zone_columns", False),
                "suspect_columns": zone_info.get("suspect_columns", []),
            },
        }
        q = _build_clarification_question("epm", candidates, evidence)
        assert "数据无 zone 列" not in q, (
            f"有疑似归属列时措辞不该说'数据无 zone 列'（假陈述），实际: {q}"
        )
        assert ("疑似" in q or "归属列" in q), f"措辞应明示疑似归属列，实际: {q}"
        assert "NoZones" not in q or "非 NoZones" in q, "措辞应明示非 NoZones"


# ---------------------------------------------------------------------------
# 2. GUARD: pure trajectory (no suspect columns) → NoZones stays
# ---------------------------------------------------------------------------


_PURE_TRAJECTORY_COLUMNS = [
    "trial_time",
    "recording_time",
    "x_center",
    "y_center",
    "area",
    "areachange",
    "elongation",
    "distance_moved",
    "velocity",
]


class TestPureTrajectoryGuards:
    def test_pure_trajectory_has_no_suspect(self):
        """守护：纯轨迹列 → has_suspect_zone_columns=False（轨迹派生量不是归属列）。"""
        z = _detect_zone_config(_PURE_TRAJECTORY_COLUMNS)
        assert z["has_zone_columns"] is False
        assert z["has_suspect_zone_columns"] is False, (
            f"纯轨迹列不该检出疑似归属列，实际 suspect: {z.get('suspect_columns')}"
        )

    def test_pure_trajectory_keeps_nozones(self):
        """守护：纯轨迹列 → NoZones 保留在候选里（真实 NoZones 数据路径不破坏）。"""
        zone_info = _detect_zone_config(_PURE_TRAJECTORY_COLUMNS)
        candidates = ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]
        filtered = _filter_candidates_by_zone(candidates, zone_info)
        assert "PlusMaze-NoZones" in filtered, "纯轨迹数据 NoZones 必须保留"

    def test_trajectory_derived_columns_not_suspect(self):
        """边界：velocity/area/areachange/elongation/distance_moved/result_1/x_center/y_center
        都不是疑似归属列（轨迹派生量/坐标/常量，非区归属）。"""
        derived = [
            "velocity",
            "area",
            "areachange",
            "elongation",
            "distance_moved",
            "result_1",
            "x_center",
            "y_center",
        ]
        z = _detect_zone_config(derived)
        assert z["has_suspect_zone_columns"] is False, (
            f"轨迹派生列被误判为疑似归属列: {z.get('suspect_columns')}"
        )

    def test_clarification_says_no_zone_on_pure_trajectory(self):
        """守护：纯轨迹 → 措辞仍说"数据无 zone 列"（唯一保留该措辞的合法路径）。"""
        zone_info = _detect_zone_config(_PURE_TRAJECTORY_COLUMNS)
        evidence = {
            "zone_info": {
                "has_zone_columns": zone_info["has_zone_columns"],
                "has_novobj_columns": zone_info["has_novobj_columns"],
                "has_suspect_zone_columns": zone_info.get("has_suspect_zone_columns", False),
                "suspect_columns": zone_info.get("suspect_columns", []),
            }
        }
        q = _build_clarification_question("epm", [{"template_id": "PlusMaze-NoZones", "recommended": True, "why": ""}], evidence)
        assert "数据无 zone 列" in q


# ---------------------------------------------------------------------------
# 3. GUARD: standard in_zone column unchanged
# ---------------------------------------------------------------------------


class TestStandardInZoneUnchanged:
    def test_standard_inzone_still_removes_nozones(self):
        """守护：标准 in_zone 列命中 → NoZones 剔除（现有行为不回归）。"""
        z = _detect_zone_config(["trial_time", "x_center", "in_zone_open_arms", "in_zone_closed_arms"])
        assert z["has_zone_columns"] is True
        candidates = ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]
        filtered = _filter_candidates_by_zone(candidates, z)
        assert "PlusMaze-NoZones" not in filtered


# ---------------------------------------------------------------------------
# 4. EDGE: OFT real columns (center/边缘区)
# ---------------------------------------------------------------------------


class TestOFTSuspectColumns:
    def test_oft_center_edgezone_detected(self):
        """边界：OFT 真实列含 center/边缘区 → 疑似归属列检出 + NoZones 剔除。"""
        oft_cols = [
            "trial_time",
            "x_center",
            "y_center",
            "distance_moved",
            "velocity",
            "center",
            "边缘区",
        ]
        z = _detect_zone_config(oft_cols)
        assert z["has_suspect_zone_columns"] is True, (
            f"OFT center/边缘区 应检出疑似归属列，实际: {z.get('suspect_columns')}"
        )
        candidates = ["OpenFieldRectangle-AllZones", "OpenFieldRectangle-NoZones"]
        filtered = _filter_candidates_by_zone(candidates, z)
        assert not any("nozone" in c.lower() for c in filtered), (
            f"OFT 有归属列时 NoZones 应剔除，实际: {filtered}"
        )


# ---------------------------------------------------------------------------
# 5. EDGE: FST not affected (mobility_state/对照组 not suspect)
# ---------------------------------------------------------------------------


class TestFSTNotAffected:
    def test_fst_columns_not_suspect(self):
        """边界：FST 列含 mobility_state/对照组 → 不是疑似归属列（不命中白名单）。"""
        fst_cols = [
            "trial_time",
            "x_center",
            "y_center",
            "distance_moved",
            "velocity",
            "mobility_state",
        ]
        z = _detect_zone_config(fst_cols)
        assert z["has_suspect_zone_columns"] is False, (
            f"FST mobility_state 不该被误判为归属列: {z.get('suspect_columns')}"
        )

    def test_fst_notemplate_kept_when_no_suspect(self):
        """边界：FST/TST 候选 NoTemplate，无疑似归属列时保留（双兜底安全）。"""
        z = _detect_zone_config(["trial_time", "x_center", "y_center", "mobility_state"])
        candidates = ["NoTemplate"]
        filtered = _filter_candidates_by_zone(candidates, z)
        # NoTemplate 保留（FST/TST 本就走 NoTemplate，无 zone 概念）
        assert filtered == ["NoTemplate"] or "NoTemplate" in filtered


# ---------------------------------------------------------------------------
# 6. E2E: full identify call with mocked parse_header returning real EPM cols
# ---------------------------------------------------------------------------


class TestIdentifyE2ERealEPM:
    def test_identify_excludes_nozones_for_real_epm_columns(self, tmp_path):
        """端到端：identify 工具喂真实 EPM 列 → 返回 candidates 不含 NoZones。"""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)
        real_file = upload_path / "epm_trial1.xlsx"
        real_file.write_text("dummy")

        runtime = _make_runtime(workspace)
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        mock_map = {"epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]}

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", return_value=header), \
             patch.object(_MOD, "_resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch.object(_MOD, "_read_markdown_section", return_value=None), \
             patch.object(_MOD, "_extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", mock_map), \
             patch(f"{_EV19_FACTS}.get_template_facts", return_value={"category": "PlusMaze", "zone_config": ""}):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=["/mnt/user-data/uploads/epm_trial1.xlsx"],
                user_message="高架十字迷宫实验数据",
            )

        candidates = result.get("candidates", [])
        template_ids = [c["template_id"] for c in candidates] if candidates else []
        assert not any("nozone" in t.lower() for t in template_ids), (
            f"EPM 真实列（含 open/closed 归属列）→ NoZones 不该在候选，实际: {template_ids}"
        )
        # evidence.zone_info 透传 has_suspect_zone_columns
        zi = result.get("evidence", {}).get("zone_info", {})
        assert zi.get("has_suspect_zone_columns") is True, (
            f"evidence.zone_info 应透传 has_suspect_zone_columns=True，实际: {zi}"
        )
