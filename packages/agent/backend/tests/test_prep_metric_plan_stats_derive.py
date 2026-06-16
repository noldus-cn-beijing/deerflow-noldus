"""#6a 层 B red 锚点（2026-06-16 EPM dogfood）：prep_metric_plan 端到端 —
groups=7control+21treatment → plan.statistics.skip_reason is None（统计不被误跳过）。

修复前（#6a bug）：prep 调 resolve_metrics 传 groups_file 但漏传 n_per_group/n_groups
→ stats gate `n_per_group>=2 and n_groups>=2` 用 None 评 False → skip_reason 含
'n_per_group=None' → run_metric_plan Step7 跳过统计 → handoff statistics={} 毒化 data-analyst。

本测试锚定 prep 侧显式派生组计数（spec §1.2③）：prep 写完 groups.json 后从 groups dict
派生 n_groups/n_per_group 传入 resolve_metrics。

importlib 加载 worktree 源：worktree 共享主仓 backend venv，editable deerflow 指向主仓，
直接 `from deerflow.tools.builtins.prep_metric_plan_tool import prep_metric_plan_tool` 测主仓
代码（worktree 改动不生效=假绿）。守 feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from langchain.tools import ToolRuntime

# ---------------------------------------------------------------------------
# Load the REAL prep_metric_plan_tool source from this worktree
# ---------------------------------------------------------------------------
_PREP_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "tools" / "builtins" / "prep_metric_plan_tool.py"
)

_PREP_MODULE: ModuleType | None = None


def _get_prep_module() -> ModuleType:
    global _PREP_MODULE
    if _PREP_MODULE is not None:
        return _PREP_MODULE
    assert _PREP_FILE.exists(), f"prep_metric_plan_tool.py not found at {_PREP_FILE}"
    spec = importlib.util.spec_from_file_location("prep_metric_plan_tool_worktree_6a", _PREP_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not load worktree prep_metric_plan_tool.py: {e}")
    _PREP_MODULE = module
    return module


# ---------------------------------------------------------------------------
# Helpers (mirror test_prep_metric_plan_tool.py style)
# ---------------------------------------------------------------------------

EPM_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "in zone Open arms 1 / Center-point",
    "in zone Open arms 2 / Center-point",
    "in zone Closed arms 1 / Center-point",
    "in zone Closed arms 2 / Center-point",
    "in zone Center-point / Center-point",
]


def _write_ethovision_file(path: str, columns: list[str]) -> None:
    """写一个 UTF-16 LE EthoVision 轨迹文件（parse_header 能识别的最小头）。"""
    header_lines = 36
    lines: list[str] = [f'"Number of header lines:";"{header_lines}"']
    metadata = [
        ("Experiment", "Mock EPM"),
        ("Trial name", "Trial 1"),
        ("Subject", "Subject 1"),
        ("Start time", "2026-01-01 00:00:00"),
        ("Trial duration", "300"),
        ("Arena name", "Arena 1"),
        ("Number of Subjects", "1"),
    ]
    for k, v in metadata:
        lines.append(f'"{k}";"{v}"')
    while len(lines) < header_lines - 2:
        lines.append('""')
    lines.append('"' + '";"'.join(columns) + '"')
    lines.append('"' + '";"'.join(["s"] * len(columns)) + '"')
    lines.append(";".join(["-1.0"] * len(columns)))
    content = "\r\n".join(lines) + "\r\n"
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")
        f.write(content.encode("utf-16-le"))


def _runtime_with_paths(workspace: Path, uploads: Path) -> ToolRuntime:
    return ToolRuntime(
        state={
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(uploads),
            }
        },
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


# ---------------------------------------------------------------------------
# 层 B red 锚点
# ---------------------------------------------------------------------------


def test_prep_metric_plan_stats_not_skipped_with_valid_groups(tmp_path: Path) -> None:
    """red 锚点：prep(groups=7control+21treatment) → plan.statistics.skip_reason is None。

    修复前（#6a bug）：prep 漏传 n_per_group/n_groups → skip_reason 含 'n_per_group=None'。
    """
    prep_tool = _get_prep_module().prep_metric_plan_tool

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    # 28 个 EPM 文件：前 7 control，后 21 treatment（dogfood 实测分组）
    uploaded_files: list[str] = []
    groups: dict[str, str] = {}
    for i in range(28):
        fname = f"s{i:02d}.txt"
        _write_ethovision_file(str(uploads / fname), EPM_COLUMNS)
        vpath = f"/mnt/user-data/uploads/{fname}"
        uploaded_files.append(vpath)
        groups[vpath] = "control" if i < 7 else "treatment"

    runtime = _runtime_with_paths(workspace, uploads)
    result = prep_tool.invoke({
        "uploaded_files": uploaded_files,
        "paradigm": "epm",
        "groups": groups,
        "runtime": runtime,
    })

    assert result["status"] == "ok", f"prep 失败：{result}"
    plan_path = workspace / "plan_metrics.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text())

    # 核心断言：统计不被误跳过
    stats = plan.get("statistics")
    assert stats is not None, "EPM plan 应有 statistics 段"
    skip_reason = stats.get("skip_reason")
    assert skip_reason is None, (
        f"#6a bug：统计被误跳过（应为 None）。skip_reason={skip_reason!r}"
    )


def test_prep_metric_plan_single_group_skips_correctly(tmp_path: Path) -> None:
    """守护：单组（28 个全 control）→ 正确 skip（不误伤单组 skip 语义）。"""
    prep_tool = _get_prep_module().prep_metric_plan_tool

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    uploaded_files: list[str] = []
    groups: dict[str, str] = {}
    for i in range(5):  # 单组 5 个
        fname = f"s{i:02d}.txt"
        _write_ethovision_file(str(uploads / fname), EPM_COLUMNS)
        vpath = f"/mnt/user-data/uploads/{fname}"
        uploaded_files.append(vpath)
        groups[vpath] = "control"

    runtime = _runtime_with_paths(workspace, uploads)
    result = prep_tool.invoke({
        "uploaded_files": uploaded_files,
        "paradigm": "epm",
        "groups": groups,
        "runtime": runtime,
    })

    assert result["status"] == "ok", f"prep 失败：{result}"
    plan = json.loads((workspace / "plan_metrics.json").read_text())
    stats = plan.get("statistics")
    assert stats is not None
    # n_groups=1 < 2 → 正确 skip（单组无组间比较）
    assert stats.get("skip_reason") is not None, "单组应被正确 skip"


def test_prep_metric_plan_no_groups_skips_correctly(tmp_path: Path) -> None:
    """守护：无 groups（单组/未收集分组）→ 正确 skip。"""
    prep_tool = _get_prep_module().prep_metric_plan_tool

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    _write_ethovision_file(str(uploads / "s00.txt"), EPM_COLUMNS)
    runtime = _runtime_with_paths(workspace, uploads)
    result = prep_tool.invoke({
        "uploaded_files": ["/mnt/user-data/uploads/s00.txt"],
        "paradigm": "epm",
        "groups": None,  # 无分组
        "runtime": runtime,
    })

    assert result["status"] == "ok", f"prep 失败：{result}"
    plan = json.loads((workspace / "plan_metrics.json").read_text())
    stats = plan.get("statistics")
    assert stats is not None
    assert stats.get("skip_reason") is not None  # 无分组计数 None → skip


def test_prep_metric_plan_writes_groups_json(tmp_path: Path) -> None:
    """附带守护：传 groups → groups.json 落盘（下游 code-executor 据此分组聚合）。"""
    prep_tool = _get_prep_module().prep_metric_plan_tool

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    uploaded_files: list[str] = []
    groups: dict[str, str] = {}
    for i in range(6):
        fname = f"s{i:02d}.txt"
        _write_ethovision_file(str(uploads / fname), EPM_COLUMNS)
        vpath = f"/mnt/user-data/uploads/{fname}"
        uploaded_files.append(vpath)
        groups[vpath] = "control" if i < 3 else "treatment"

    runtime = _runtime_with_paths(workspace, uploads)
    result = prep_tool.invoke({
        "uploaded_files": uploaded_files,
        "paradigm": "epm",
        "groups": groups,
        "runtime": runtime,
    })

    assert result["status"] == "ok", f"prep 失败：{result}"
    groups_json = workspace / "groups.json"
    assert groups_json.exists(), "groups.json 应落盘"
    data = json.loads(groups_json.read_text())
    assert len(data) == 6
    assert set(data.values()) == {"control", "treatment"}
