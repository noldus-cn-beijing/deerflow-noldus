"""验证 metric_plan.json 的 output 路径必须是虚拟路径（不能含 host 物理路径）。

Issue #6 from thread 5046a6e6 dogfood:
plan.json 的 output 字段写了 /home/wangqiuyang/.../workspace/m_*.json 物理路径，
打穿了 sandbox 抽象。后续 subagent 看到这条 plan 时，本来应该用 /mnt/...
虚拟路径，但 lead 直接 cat plan.json 后把物理路径暴露给模型。
"""

from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import resolve


VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"
VIRTUAL_UPLOADS = "/mnt/user-data/uploads"


# EPM default_metrics 要求列 `in_zone_open_arms_*` (glob)。
# 测试用 1 个匹配的列名让 resolve 真跑通完整流程。
EPM_MIN_COLUMNS = [
    "in_zone_open_arms_center",
    "in_zone_closed_arms_center",
]


def test_resolve_outputs_only_virtual_paths(tmp_path: Path) -> None:
    """传入 virtual_workspace_dir 后，output 必须使用虚拟路径而非物理路径。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [f"{VIRTUAL_UPLOADS}/dummy.txt"]

    plan = resolve(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        ev19_template="PlusMaze-FewZones",
        virtual_workspace_dir=VIRTUAL_WORKSPACE,
    )

    for m in plan.metrics:
        assert m.output.startswith(VIRTUAL_WORKSPACE), (
            f"metric {m.id} output 不是虚拟路径: {m.output}"
        )
        assert "/home/" not in m.output, (
            f"metric {m.id} output 含 host 物理路径前缀: {m.output}"
        )
        assert physical_workspace not in m.output, (
            f"metric {m.id} output 含物理 workspace 路径: {m.output}"
        )


def test_resolve_outputs_use_virtual_workspace_with_explicit_kwarg(tmp_path: Path) -> None:
    """resolve 收到 virtual_workspace_dir 后必须用它而非物理 workspace_dir。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)
    virtual_workspace = "/mnt/user-data/workspace"

    raw_files = [f"{VIRTUAL_UPLOADS}/dummy.txt"]

    plan = resolve(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,
        ev19_template="PlusMaze-FewZones",
        virtual_workspace_dir=virtual_workspace,
    )

    for m in plan.metrics:
        assert m.output.startswith(virtual_workspace), m.output
