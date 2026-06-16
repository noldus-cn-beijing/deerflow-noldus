"""#6a red 锚点（2026-06-16 EPM dogfood）：统计被误跳过的根因——
prep_metric_plan 调 resolve_metrics 时漏传 n_per_group/n_groups（传了 None），
stats gate `n_per_group>=2 and n_groups>=2` 用 None 评估 → False → plan.statistics.skip_reason
被写死 → run_metric_plan Step7 跳过统计 → handoff statistics={}。

治本：让 resolve_metrics 在拿到 groups_file 时自读文件派生组计数，
使"有分组却 statistics={}"结构上不可能（计数从"调用方必传入参"降级为
"在拿得到 groups 内容的唯一权威点自动派生的内部量"）。

三层 red 锚点：
- _derive_group_counts 纯函数：两种 groups.json 形状都能派生 (n_per_group=min组, n_groups=组数)。
- resolve_metrics 只传 groups_file（不传 n）→ 自派生 → statistics 不被 skip。
- 单组守护：1 组 → 正确 skip（不误伤）。

importlib 加载 worktree 源：worktree 共享主仓 venv，editable ethoinsight 指向主仓源，
直接 `from ethoinsight...import` 测主仓代码（worktree 改动不生效=假绿）。
守 feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL resolve.py source from this worktree (bypass editable link → main repo)
# ---------------------------------------------------------------------------
_RESOLVE_FILE = (
    Path(__file__).resolve().parents[1]
    / "ethoinsight" / "catalog" / "resolve.py"
)

_REAL_RESOLVE: ModuleType | None = None


def _get_resolve() -> ModuleType:
    global _REAL_RESOLVE
    if _REAL_RESOLVE is not None:
        return _REAL_RESOLVE

    assert _RESOLVE_FILE.exists(), f"resolve.py not found at {_RESOLVE_FILE}"
    # 注册一个真实模块名，让 resolve.py 内部的 `from ethoinsight...import` 能命中
    # 已安装的 editable ethoinsight（数据层 load_catalog 等走主仓，catalog YAML 内容一致）。
    spec = importlib.util.spec_from_file_location(
        "ethoinsight.catalog.resolve_worktree_6a",
        _RESOLVE_FILE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:  # pragma: no cover - 加载失败说明环境异常
        pytest.skip(f"Could not load worktree resolve.py: {e}")
    _REAL_RESOLVE = module
    return _REAL_RESOLVE


VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"
VIRTUAL_UPLOADS = "/mnt/user-data/uploads"

EPM_MIN_COLUMNS = [
    "in_zone_open_arms_center",
    "in_zone_closed_arms_center",
]


def _write_groups_subject_map(workspace: Path, mapping: dict[str, str]) -> str:
    """写 groups.json（{subject_path: group_name} 正向映射，prep_metric_plan 写的形状）→ 返回物理路径。"""
    gf = workspace / "groups.json"
    gf.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return str(gf)


# ---------------------------------------------------------------------------
# 纯函数：_derive_group_counts
# ---------------------------------------------------------------------------


def test_derive_group_counts_from_subject_map(tmp_path: Path) -> None:
    """{subject: group} 形状：Counter(values) → (n_per_group=min组, n_groups=组数)。"""
    f = _write_groups_subject_map(
        tmp_path,
        {"a.xlsx": "control", "b.xlsx": "control", "c.xlsx": "treatment"},
    )
    # 最小组 = treatment(1)，组数 = 2
    assert _get_resolve()._derive_group_counts(f) == (1, 2)


def test_derive_group_counts_balanced_groups(tmp_path: Path) -> None:
    """control=7 / treatment=21 → (7, 2)（dogfood 实测分组）。"""
    mapping = {f"s{i:02d}.xlsx": ("control" if i < 7 else "treatment") for i in range(28)}
    f = _write_groups_subject_map(tmp_path, mapping)
    assert _get_resolve()._derive_group_counts(f) == (7, 2)


def test_derive_group_counts_from_grouplist_shape(tmp_path: Path) -> None:
    """{group: [subjects]} 反向形状：{g: len(lst)} → 派生计数。"""
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"control": ["a", "b"], "treatment": ["c"]}), encoding="utf-8")
    assert _get_resolve()._derive_group_counts(str(f)) == (1, 2)


def test_derive_group_counts_none_when_no_groups_file() -> None:
    """groups_file=None（单组/无分组）→ (None, None)，gate 正确 skip。"""
    assert _get_resolve()._derive_group_counts(None) == (None, None)


def test_derive_group_counts_fail_safe_on_bad_json(tmp_path: Path) -> None:
    """读不到/坏 JSON/空 → (None, None)，fail-safe 不阻断（与现状等价）。"""
    resolve = _get_resolve()
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert resolve._derive_group_counts(str(bad)) == (None, None)

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert resolve._derive_group_counts(str(empty)) == (None, None)

    nonexistent = tmp_path / "missing.json"
    assert resolve._derive_group_counts(str(nonexistent)) == (None, None)


def test_derive_group_counts_mixed_shape_is_none(tmp_path: Path) -> None:
    """值类型混合（非全 str 也非全 list）→ (None, None)（保守不猜）。"""
    f = tmp_path / "mixed.json"
    f.write_text(json.dumps({"control": "a", "treatment": ["b", "c"]}), encoding="utf-8")
    assert _get_resolve()._derive_group_counts(str(f)) == (None, None)


# ---------------------------------------------------------------------------
# resolve_metrics 自派生：只传 groups_file（不传 n）→ 统计不被 skip
# ---------------------------------------------------------------------------


def test_resolve_self_derives_counts_so_stats_not_skipped(tmp_path: Path) -> None:
    """red 锚点：只传 groups_file（不传 n）→ resolve 自派生 → statistics 不被 skip。

    修复前：resolve 不自派生 → None → skip_reason 含 'n_per_group=None'。
    """
    resolve = _get_resolve()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gf = _write_groups_subject_map(
        workspace,
        {f"{VIRTUAL_UPLOADS}/s{i:02d}.txt": ("control" if i < 7 else "treatment") for i in range(28)},
    )
    raw_files = [f"{VIRTUAL_UPLOADS}/s{i:02d}.txt" for i in range(28)]
    pm = resolve.resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=str(workspace),
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        groups_file=gf,  # 关键：传 groups_file
        # 关键：不传 n_per_group / n_groups（复现 prep_metric_plan #6a bug 的入参形态）
    )
    assert pm.statistics is not None, "EPM 有 statistics_default，plan.statistics 不应为 None"
    assert pm.statistics.skip_reason is None, (
        f"统计被误 skip（#6a bug）：{pm.statistics.skip_reason}"
    )


def test_resolve_explicit_counts_override_derived(tmp_path: Path) -> None:
    """守护向后兼容：显式传 n_per_group/n_groups 时仍优先用显式值（CLI 覆盖兜底）。"""
    resolve = _get_resolve()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gf = _write_groups_subject_map(
        workspace,
        {f"{VIRTUAL_UPLOADS}/s{i:02d}.txt": "control" for i in range(5)},  # 真实 1 组 5 个
    )
    raw_files = [f"{VIRTUAL_UPLOADS}/s{i:02d}.txt" for i in range(5)]
    pm = resolve.resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=str(workspace),
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        groups_file=gf,
        n_per_group=3,
        n_groups=2,  # 显式覆盖（与文件 1 组矛盾）→ 显式优先
    )
    assert pm.statistics is not None
    assert pm.statistics.skip_reason is None  # n_per_group=3>=2 and n_groups=2>=2 → 不 skip


def test_resolve_single_group_still_skips(tmp_path: Path) -> None:
    """守护：单组（n_groups 派生=1）→ 正确 skip（不误伤单组 skip 语义）。"""
    resolve = _get_resolve()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gf = _write_groups_subject_map(
        workspace,
        {f"{VIRTUAL_UPLOADS}/s{i:02d}.txt": "control" for i in range(7)},  # 1 组 7 个
    )
    raw_files = [f"{VIRTUAL_UPLOADS}/s{i:02d}.txt" for i in range(7)]
    pm = resolve.resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=str(workspace),
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        groups_file=gf,
    )
    assert pm.statistics is not None
    # n_groups=1 < 2 → 正确 skip（正常单组无组间比较场景）
    assert pm.statistics.skip_reason is not None, "单组应被正确 skip"


def test_resolve_no_groups_file_still_skips(tmp_path: Path) -> None:
    """守护：无 groups_file（单组/无分组）→ 计数 None → 正确 skip。"""
    resolve = _get_resolve()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    raw_files = [f"{VIRTUAL_UPLOADS}/s{i:02d}.txt" for i in range(5)]
    pm = resolve.resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=str(workspace),
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        groups_file=None,  # 无分组文件
    )
    assert pm.statistics is not None
    assert pm.statistics.skip_reason is not None  # None 计数 → skip


# ---------------------------------------------------------------------------
# 层 C 兜底信号：groups_file 指向坏 JSON → skip_reason 含"组计数不可得"响亮信号
# ---------------------------------------------------------------------------


def test_resolve_unreadable_groups_file_emits_loud_signal(tmp_path: Path) -> None:
    """层 C：groups_file 非空但派生不出计数（坏 JSON）→ skip_reason 写明"组计数不可得"
    （响亮、可 grep 的信号），区别于正常的"n 真的不足"skip。"""
    resolve = _get_resolve()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = workspace / "groups.json"
    bad.write_text("{not valid json", encoding="utf-8")
    raw_files = [f"{VIRTUAL_UPLOADS}/s{i:02d}.txt" for i in range(28)]
    pm = resolve.resolve_metrics(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=str(workspace),
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
        groups_file=str(bad),
    )
    assert pm.statistics is not None
    assert pm.statistics.skip_reason is not None
    # 响亮信号：写明是"组计数不可得"（读取/格式问题），非"n 真的不足"
    assert "组计数不可得" in pm.statistics.skip_reason or "不可得" in pm.statistics.skip_reason, (
        f"坏 groups.json 应触发响亮 skip 信号，得到：{pm.statistics.skip_reason}"
    )
