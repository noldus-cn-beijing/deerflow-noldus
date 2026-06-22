"""Spec 2026-06-22-identify-zone-info-persist — zone_info 落盘沉淀测试（红→绿）。

真 bug（dogfood thread ``3a41e483`` 实证，见 spec §1）：
  ``identify_ev19_template`` 花 ``parse_header`` 成本在 Step 3
  ``_detect_zone_config`` 检测出的 ``zone_info``（含
  ``suspect_columns=["open","closed"]``），**只在当次返回的内存 ``evidence`` 里**，
  落盘 ``template_candidates.json`` 时被三条路径（unknown L609 / ok L619 /
  ambiguous L643）剥掉。后果：
    ① lead 在 unknown 后为带依据反问，得自己在 thinking 里读 evidence 推断
       ``open/closed → EPM`` → 烧 turn；evidence 被 summarize 截断后还得重读文件。
    ② 检测是一次性的、不沉淀；guardrail / 将来下游要用拿不到，得重检测。

修法（spec §2）：三条 ``_write_template_candidates`` 路径的 ``data`` 都加
``"zone_info": zone_info``。工具**仍返回 unknown**（守 lead prompt L480「不猜范式」），
只是把检测成本沉淀。

守 ``feedback_worktree_shares_main_venv_editable_link``：用 importlib 加载 worktree
源码（主仓 venv 的 editable deerflow 指向主仓源码，直接 ``from deerflow...`` 会测
主仓代码 = 假绿）。
"""

from __future__ import annotations

import importlib.util
import json
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
        "deerflow.tools.builtins.identify_ev19_template_tool_real_0622",
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

# Real EPM column set (open/closed are 0/1 互斥 归属列). Same anchor as
# test_identify_zone_detection_nonstandard_columns.py.
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


def _read_template_candidates(workspace: str) -> dict | None:
    p = Path(workspace) / "template_candidates.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _run_identify(
    *,
    workspace: str,
    real_file: Path,
    header: dict,
    ev19_map: dict,
    user_message: str,
    get_template_facts_ret: dict | None = None,
) -> dict:
    """Invoke identify_ev19_template_tool with the heavy deps mocked out.

    Mirrors the patch envelope used by
    test_identify_zone_detection_nonstandard_columns.py::TestIdentifyE2ERealEPM.
    """
    runtime = _make_runtime(workspace)
    facts_ret = get_template_facts_ret or {"category": "PlusMaze", "zone_config": ""}

    with (
        patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)),
        patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True),
        patch(f"{_PARSE_CORE}.parse_header", return_value=header),
        patch.object(_MOD, "_resolve_skills_ref_dir", return_value=real_file.parent / "skills_ref"),
        patch.object(_MOD, "_read_markdown_section", return_value=None),
        patch.object(_MOD, "_extract_template_recommendations", return_value=[]),
        patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", ev19_map),
        patch(f"{_EV19_FACTS}.get_template_facts", return_value=facts_ret),
        patch(f"{_EV19_FACTS}.SUPPORTED_PARADIGMS_V01", {"epm", "open_field", "forced_swim", "light_dark_box", "zero_maze"}),
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


# ---------------------------------------------------------------------------
# 1. RED anchor (reproduce dogfood): unknown path persists zone_info
# ---------------------------------------------------------------------------


class TestUnknownPersistsZoneInfo:
    def test_unknown_persists_zone_info(self, workspace_and_file):
        """红线（复现 dogfood）：unknown 路径落盘 template_candidates.json
        必须含 ``zone_info.suspect_columns == ["open","closed"]``。

        改动前红：unknown 路径只写 ``{"status": "unknown", "paradigm_key": None}``，
        落盘文件无 ``zone_info`` 键 → ``KeyError``/``None``。
        """
        workspace, real_file = workspace_and_file
        # No paradigm keyword in filename/message AND empty ev19_map → no candidates
        # → status="unknown" path (L607-614).
        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=_header_with_columns(_EPM_REAL_COLUMNS),
            ev19_map={},  # no candidate mapping for any paradigm
            user_message="这是我的行为学实验数据",  # no paradigm keyword
        )

        assert result["status"] == "unknown", f"预期 unknown 路径，实际: {result['status']}"

        tc = _read_template_candidates(workspace)
        assert tc is not None, "template_candidates.json 必须落盘"
        assert "zone_info" in tc, (
            f"unknown 路径落盘必须带 zone_info（spec §2.1），实际落盘键: {list(tc.keys())}"
        )
        suspect = tc["zone_info"].get("suspect_columns")
        assert suspect == ["open", "closed"], (
            f"落盘 zone_info.suspect_columns 应含 open/closed，实际: {suspect}"
        )


# ---------------------------------------------------------------------------
# 2. ok + ambiguous paths also persist zone_info
# ---------------------------------------------------------------------------


class TestOkAmbiguousPersistZoneInfo:
    def test_ok_path_persists_zone_info(self, workspace_and_file):
        """ok 路径（单一候选）落盘也含 zone_info。"""
        workspace, real_file = workspace_and_file
        # epm → single candidate → filter keeps it → len==1 → ok path (L616-639).
        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=_header_with_columns(_EPM_REAL_COLUMNS),
            ev19_map={"epm": ["PlusMaze-AllZones"]},
            user_message="高架十字迷宫实验数据",  # epm keyword
        )

        assert result["status"] == "ok", f"预期 ok 路径，实际: {result['status']}"

        tc = _read_template_candidates(workspace)
        assert tc is not None and "zone_info" in tc, (
            f"ok 路径落盘必须带 zone_info，实际键: {list((tc or {}).keys())}"
        )
        assert tc["zone_info"].get("suspect_columns") == ["open", "closed"]

    def test_ambiguous_path_persists_zone_info(self, workspace_and_file):
        """ambiguous 路径（2-3 候选）落盘也含 zone_info。"""
        workspace, real_file = workspace_and_file
        # epm → multiple zone variants → filter keeps AllZones + FewZones (>=2)
        # → ambiguous path (L641-657).
        result = _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=_header_with_columns(_EPM_REAL_COLUMNS),
            ev19_map={"epm": ["PlusMaze-AllZones", "PlusMaze-FewZones", "PlusMaze-NoZones"]},
            user_message="高架十字迷宫实验数据",
        )

        assert result["status"] == "ambiguous", f"预期 ambiguous 路径，实际: {result['status']}"

        tc = _read_template_candidates(workspace)
        assert tc is not None and "zone_info" in tc, (
            f"ambiguous 路径落盘必须带 zone_info，实际键: {list((tc or {}).keys())}"
        )
        assert tc["zone_info"].get("suspect_columns") == ["open", "closed"]


# ---------------------------------------------------------------------------
# 3. JSON-serializable + full 6 fields
# ---------------------------------------------------------------------------


class TestZoneInfoJsonSerializable:
    def test_zone_info_is_valid_json_with_all_fields(self, workspace_and_file):
        """落盘的 zone_info 是合法 JSON 且含 _detect_zone_config 的全部 6 字段
        （has_zone_columns / has_novobj_columns / has_suspect_zone_columns /
        zone_columns / novobj_columns / suspect_columns），可被后续读取方反序列化。

        守 spec §5.1 + §6.4「内存与落盘双写一致」：落盘的 zone_info 与变量同源。
        """
        workspace, real_file = workspace_and_file
        _run_identify(
            workspace=workspace,
            real_file=real_file,
            header=_header_with_columns(_EPM_REAL_COLUMNS),
            ev19_map={},
            user_message="这是我的行为学实验数据",
        )

        raw = (Path(workspace) / "template_candidates.json").read_text(encoding="utf-8")
        # Whole file is valid JSON (json.loads round-trip).
        parsed = json.loads(raw)

        zi = parsed["zone_info"]
        expected_keys = {
            "has_zone_columns",
            "has_novobj_columns",
            "has_suspect_zone_columns",
            "zone_columns",
            "novobj_columns",
            "suspect_columns",
        }
        assert expected_keys.issubset(zi.keys()), (
            f"zone_info 落盘应含全部 6 字段，缺: {expected_keys - set(zi.keys())}"
        )
        # zone_info 字段类型正确（守反序列化方可消费）
        assert isinstance(zi["has_zone_columns"], bool)
        assert isinstance(zi["has_suspect_zone_columns"], bool)
        assert isinstance(zi["suspect_columns"], list)
        assert isinstance(zi["zone_columns"], list)


# ---------------------------------------------------------------------------
# 4. Static prompt contract: lead prompt has column-evidence rule
# ---------------------------------------------------------------------------

_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "agents" / "lead_agent" / "prompt.py"
)

_REAL_PROMPT: ModuleType | None = None


def _get_real_prompt() -> ModuleType:
    global _REAL_PROMPT
    if _REAL_PROMPT is not None:
        return _REAL_PROMPT
    spec = importlib.util.spec_from_file_location(
        "deerflow.agents.lead_agent.prompt_real_0622",
        _PROMPT_FILE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        pytest.skip("Could not load real prompt module")
    _REAL_PROMPT = module
    return _REAL_PROMPT


@pytest.fixture(autouse=False)
def _ensure_prompt_loaded():
    _get_real_prompt()


def test_lead_prompt_has_column_evidence_rule():
    """spec §2.2 静态契约：lead prompt 含「范式反问带列依据」措辞 +
    「open/closed 同时支撑 EPM 和 Zero Maze」措辞（钉死列信号→多范式事实，
    防 lead 猜成单一范式）。
    """
    prompt = _get_real_prompt()
    text = prompt.SYSTEM_PROMPT_TEMPLATE
    # 核心指令：反问带列依据、把列信号支撑的所有范式都列为选项
    assert "范式反问带列依据" in text or "带列依据" in text, (
        "lead prompt 必须含「范式反问带列依据」铁律措辞（spec §2.2）"
    )
    # 钉死 open/closed 同见 EPM + Zero Maze 的事实（不许猜成单一范式）
    assert ("Zero Maze" in text or "ZeroMaze" in text or "零迷宫" in text), (
        "lead prompt 必须明示 open/closed 同时支撑 EPM 和 Zero Maze（spec §2.2 例）"
    )
    # 必须把「多个范式列为选项」的正面指令写进去
    assert "所有范式都列为选项" in text or "把列信号支撑的所有范式" in text or "都列为选项" in text


# ---------------------------------------------------------------------------
# 5. Forward-compat: guardrail ignores extra zone_info field
# ---------------------------------------------------------------------------


def test_guardrail_ignores_extra_zone_info_field(tmp_path):
    """spec §5.5 + §3.3 向前兼容：template_candidates.json 含 zone_info 字段
    不破坏 Ev19TemplateGuardrailProvider（它只读 status/candidates）。
    """
    from deerflow.guardrails.ev19_template_provider import Ev19TemplateGuardrailProvider
    from deerflow.guardrails.provider import GuardrailRequest

    # workspace 有 experiment-context.json（ev19_template 已设）+ 含 zone_info 的
    # template_candidates.json（status=ambiguous + 多余 zone_info 字段）。
    ctx = {"paradigm": "epm", "ev19_template": "PlusMaze-AllZones"}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
    tc = {
        "status": "ambiguous",
        "paradigm_key": "epm",
        "candidates": [{"template_id": "PlusMaze-AllZones"}, {"template_id": "PlusMaze-FewZones"}],
        "zone_info": {
            "has_zone_columns": False,
            "has_novobj_columns": False,
            "has_suspect_zone_columns": True,
            "zone_columns": [],
            "novobj_columns": [],
            "suspect_columns": ["open", "closed"],
        },
    }
    (tmp_path / "template_candidates.json").write_text(json.dumps(tc, ensure_ascii=False), encoding="utf-8")

    provider = Ev19TemplateGuardrailProvider(workspace_resolver=lambda: str(tmp_path))

    # set_experiment_paradigm 调用带 ambiguous 模板里存在的 ev19_template 但未确认 →
    # 应正常返回 deny（template_not_confirmed），证明它读到了 status=candidates 而没被
    # 多余的 zone_info 字段弄崩。
    req = GuardrailRequest(
        tool_name="set_experiment_paradigm",
        tool_input={"ev19_template": "PlusMaze-AllZones"},
        agent_id=None,
        timestamp="2026-06-22T00:00:00Z",
    )
    decision = provider.evaluate(req)
    # 关键断言：provider 正常运行（没因 zone_info 抛异常），且因 ambiguous 未确认而 deny。
    assert decision.allow is False, (
        "guardrail 应拒绝未确认的 ambiguous 模板，证明它正确解析了含 zone_info 的文件"
    )
    assert any(r.code == "ethoinsight.template_not_confirmed" for r in decision.reasons)

    # 也验 task(code-executor) 在 ev19_template 已设时放行（不读 zone_info 也不坏）。
    req2 = GuardrailRequest(
        tool_name="task",
        tool_input={"subagent_type": "code-executor", "prompt": "..."},
        agent_id=None,
        timestamp="2026-06-22T00:00:00Z",
    )
    decision2 = provider.evaluate(req2)
    assert decision2.allow is True, "ev19_template 已设时 task(code-executor) 应放行"
