"""Spec 2026-06-23-template-candidate-ab-order-deterministic — 候选列表确定性排序测试（红→绿）。

真 bug（ETHO-8，EPM 两轮 E2E 实证）：
  ``identify_ev19_template`` Step 7 遍历 ``target_ids`` 构造 ``candidates`` 列表，
  **遍历前没排序**。``target_ids`` 来自 ``EV19_TEMPLATE_PARADIGM_MAP`` 的 dict 值列表
  （或经 ``_filter_candidates_by_zone`` 过滤、保留原相对序），其顺序跟随 map 初始化
  时的填充序——不保证跨进程/跨调用稳定。A/B 字母标签在 ``_build_clarification_question``
  里按 ``candidates`` 索引位分配 → 候选序一变，A/B 就对调，且「推荐项」绑在
  ``target_ids[0]`` 也随之漂移。

  现象：同一份数据两轮 E2E，Run1 模板反问 A=AllZones/B=FewZones，Run2 反过来。

修法（spec §2.1）：Step 7 遍历前对 ``target_ids`` 做确定性排序——推荐项
（``experiment_recs`` 命中的）排最前（恒为 A），其余按 ``template_id`` 字典序。

守 ``feedback_worktree_shares_main_venv_editable_link``：用 importlib 加载 worktree
源码（主仓 venv 的 editable deerflow 指向主仓源码，直接 ``from deerflow...`` 会测
主仓代码 = 假绿）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Load the identify tool module fresh from THIS worktree's source (bypass
# editable link — see module docstring).
# ---------------------------------------------------------------------------
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
        "deerflow.tools.builtins.identify_ev19_template_tool_real_0623_ab",
        _TOOL_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_MOD = _load_identify_module()
identify_ev19_template_tool = _MOD.identify_ev19_template_tool

# Module path strings for patching.
_SANDBOX_TOOLS = "deerflow.sandbox.tools"
_PARSE_CORE = "ethoinsight.parse._core"
_EV19_FACTS = "ethoinsight.ev19_facts"

# Real EPM column set (open/closed are 0/1 互斥 归属列 → has_suspect_zone_columns
# path keeps AllZones/FewZones, drops NoZones). Same anchor as
# test_identify_ev19_zone_info_persisted.py.
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


def _run_identify(
    *,
    workspace: str,
    real_file: Path,
    header: dict,
    ev19_map: dict,
    user_message: str,
    experiment_recs: list[str] | None = None,
    get_template_facts_ret: dict | None = None,
) -> dict:
    """Invoke identify_ev19_template_tool with the heavy deps mocked out.

    Mirrors the patch envelope used by
    test_identify_ev19_zone_info_persisted.py / test_identify_zone_detection_nonstandard_columns.py.
    """
    runtime = _make_runtime(workspace)
    facts_ret = get_template_facts_ret or {"category": "PlusMaze", "zone_config": ""}
    recs = [] if experiment_recs is None else experiment_recs

    with (
        patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)),
        patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True),
        patch(f"{_PARSE_CORE}.parse_header", return_value=header),
        patch.object(_MOD, "_resolve_skills_ref_dir", return_value=real_file.parent / "skills_ref"),
        patch.object(_MOD, "_read_markdown_section", return_value=None),
        patch.object(_MOD, "_extract_template_recommendations", return_value=recs),
        patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", ev19_map),
        patch(f"{_EV19_FACTS}.get_template_facts", return_value=facts_ret),
        patch(
            f"{_EV19_FACTS}.SUPPORTED_PARADIGMS_V01",
            {"epm", "open_field", "forced_swim", "light_dark_box", "zero_maze"},
        ),
    ):
        return identify_ev19_template_tool.func(
            runtime=runtime,
            uploaded_files=[f"/mnt/user-data/uploads/{real_file.name}"],
            user_message=user_message,
        )


@pytest.fixture
def workspace_and_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    real_file = uploads / "trial01.xlsx"
    real_file.write_text("dummy")
    return str(workspace), real_file


def _candidate_ids(result: dict) -> list[str]:
    """Extract ordered template_id list from an identify result."""
    cands = result.get("candidates") or []
    return [c["template_id"] for c in cands]


# ---------------------------------------------------------------------------
# 1. test_candidate_order_is_deterministic_across_calls（核心，红→绿）
# ---------------------------------------------------------------------------


class TestCandidateOrderDeterministic:
    def test_candidate_order_is_deterministic_across_calls(self, workspace_and_file):
        """核心红线：把 ``EV19_TEMPLATE_PARADIGM_MAP["epm"]`` 的值列表故意打乱成两种
        顺序（模拟 dict 迭代序漂移），各调一次工具，断言两次返回的 ``candidates``
        的 ``template_id`` 顺序**完全一致**。

        改前红：``target_ids`` 直接跟随打乱的输入序 → 两次结果顺序不同（A/B 对调）。
        改后绿：Step 7 排序让推荐项在前、其余字典序 → 两次结果一致。

        EPM 真实列（open/closed）触发 ``has_suspect_zone_columns`` 分支，
        ``_filter_candidates_by_zone`` 剔除 NoZones、保留 AllZones/FewZones（≥2 →
        ambiguous 路径）。两次都用同一个 ``experiment_recs``（含 AllZones）保证
        推荐项稳定，只动 map 值列表的顺序这一自变量。
        """
        workspace, real_file = workspace_and_file
        # open/closed 都在列 → has_suspect_zone_columns → filter 保留 AllZones/FewZones
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        recs = ["PlusMaze-AllZones"]  # 稳定的推荐集合（by-experiment 来源是确定输入）

        # 两种「map 填充序」——模拟跨进程/跨调用的 dict 迭代序漂移
        map_order_a = {"epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]}
        map_order_b = {"epm": ["PlusMaze-FewZones", "PlusMaze-NoZones", "PlusMaze-AllZones"]}

        r1 = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=header,
            ev19_map=map_order_a,
            user_message="高架十字迷宫实验数据",
            experiment_recs=recs,
        )
        # 第二个 workspace 避免落盘 template_candidates.json 互相覆盖干扰
        workspace2 = str(Path(workspace).parent / "workspace2")
        Path(workspace2).mkdir(parents=True, exist_ok=True)
        r2 = _run_identify(
            workspace=workspace2,
            real_file=real_file,
            header=header,
            ev19_map=map_order_b,
            user_message="高架十字迷宫实验数据",
            experiment_recs=recs,
        )

        order1 = _candidate_ids(r1)
        order2 = _candidate_ids(r2)
        assert order1, "至少应有候选（EPM ambiguous 路径）"
        assert order1 == order2, (
            f"候选顺序必须跨调用确定（改 A/B 对调），实际：\n  map_order_a → {order1}\n  map_order_b → {order2}"
        )


# ---------------------------------------------------------------------------
# 2. test_recommended_candidate_is_first
# ---------------------------------------------------------------------------


class TestRecommendedFirst:
    def test_recommended_candidate_is_first(self, workspace_and_file):
        """构造 ``experiment_recs`` 含某个 template_id（AllZones），断言：
        ``candidates[0].template_id == 推荐项`` 且 ``candidates[0].recommended is True``。

        守 spec §2.2 / §5.3：推荐项恒为首位 → 字母 A。无论 map 值列表怎么排，
        推荐项都得被排到 index 0。
        """
        workspace, real_file = workspace_and_file
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        # map 里把推荐项 AllZones 故意放最后，验证排序能把它提到首位
        ev19_map = {"epm": ["PlusMaze-FewZones", "PlusMaze-NoZones", "PlusMaze-AllZones"]}
        recs = ["PlusMaze-AllZones"]

        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=header,
            ev19_map=ev19_map,
            user_message="高架十字迷宫实验数据",
            experiment_recs=recs,
        )

        cands = result["candidates"]
        assert cands, "应有候选"
        assert cands[0]["template_id"] == "PlusMaze-AllZones", (
            f"推荐项必须排首位（恒为 A），实际首位: {cands[0]['template_id']}"
        )
        assert cands[0]["recommended"] is True, "首位候选的 recommended 必须为 True"


# ---------------------------------------------------------------------------
# 3. test_non_recommended_sorted_lexicographically
# ---------------------------------------------------------------------------


class TestNonRecommendedLexicographic:
    def test_non_recommended_sorted_lexicographically(self, workspace_and_file):
        """无 ``experiment_recs``（或推荐项之外的候选）时，候选按 ``template_id``
        字典序排列（排序退化为纯字典序，稳定）。

        守 spec §2.2 注 + §6.2：别引入语义排序，字典序即可。
        """
        workspace, real_file = workspace_and_file
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        # map 乱序：FewZones 在 AllZones 前，验证过滤后（剔除 NoZones）的候选会被
        # 字典序重排成 [AllZones, FewZones]
        ev19_map = {"epm": ["PlusMaze-FewZones", "PlusMaze-NoZones", "PlusMaze-AllZones"]}
        # 无推荐 → 排序退化为字典序
        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=header,
            ev19_map=ev19_map,
            user_message="高架十字迷宫实验数据",
            experiment_recs=[],
        )

        order = _candidate_ids(result)
        # NoZones 被 _filter_candidates_by_zone 剔除（has_suspect 分支），剩 AllZones/FewZones
        assert order == ["PlusMaze-AllZones", "PlusMaze-FewZones"], (
            f"无推荐时候选应按 template_id 字典序，实际: {order}"
        )


# ---------------------------------------------------------------------------
# 4. test_ab_label_stable_in_clarification_question
# ---------------------------------------------------------------------------


class TestAbLabelStable:
    def test_ab_label_stable_in_clarification_question(self, workspace_and_file):
        """调工具取 ``clarification_question`` 文本，断言：字母 A 对应的 template_id
        跨两次（打乱 map 输入）调用一致（守 spec §5.2：A/B 字母↔模板映射稳定）。

        ``_build_clarification_question`` 按 candidates 索引位分配字母 A/B/…，并写
        「如果不确定，选 A（<template_id>）。」。解析该行即可钉死 A↔模板。
        """
        import re

        workspace, real_file = workspace_and_file
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        recs = ["PlusMaze-AllZones"]

        map_order_a = {"epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]}
        map_order_b = {"epm": ["PlusMaze-FewZones", "PlusMaze-NoZones", "PlusMaze-AllZones"]}

        def _a_template(result: dict) -> str:
            q = result["clarification_question"]
            # 「如果不确定，选 A（<template_id>）。」钉死 A 对应的模板
            m = re.search(r"选 A（([^）]+)）", q)
            assert m, f"clarification_question 应含「选 A（<template>）」行，实际:\n{q}"
            return m.group(1)

        r1 = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=header,
            ev19_map=map_order_a,
            user_message="高架十字迷宫实验数据",
            experiment_recs=recs,
        )
        workspace2 = str(Path(workspace).parent / "workspace2b")
        Path(workspace2).mkdir(parents=True, exist_ok=True)
        r2 = _run_identify(
            workspace=workspace2,
            real_file=real_file,
            header=header,
            ev19_map=map_order_b,
            user_message="高架十字迷宫实验数据",
            experiment_recs=recs,
        )

        a1 = _a_template(r1)
        a2 = _a_template(r2)
        assert a1 == a2, (
            f"字母 A 对应的模板必须跨调用稳定，实际：map_a → A={a1}，map_b → A={a2}"
        )
        # 推荐项是 AllZones → A 必对应 AllZones（推荐项恒为 A）
        assert a1 == "PlusMaze-AllZones", f"A 应对应推荐项 AllZones，实际: {a1}"


# ---------------------------------------------------------------------------
# 5. test_single_candidate_unaffected
# ---------------------------------------------------------------------------


class TestSingleCandidateUnaffected:
    def test_single_candidate_unaffected(self, workspace_and_file):
        """单候选场景，排序对单元素无副作用：仍是那一个、status=ok、recommended=True
        （守 spec §5.5：不改 status 判定、不改候选集合内容，只改顺序）。
        """
        workspace, real_file = workspace_and_file
        header = _header_with_columns(_EPM_REAL_COLUMNS)
        ev19_map = {"epm": ["PlusMaze-AllZones"]}

        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=header,
            ev19_map=ev19_map,
            user_message="高架十字迷宫实验数据",
            experiment_recs=[],
        )

        assert result["status"] == "ok", f"单候选应为 ok，实际: {result['status']}"
        order = _candidate_ids(result)
        assert order == ["PlusMaze-AllZones"], f"单候选应原样保留，实际: {order}"
