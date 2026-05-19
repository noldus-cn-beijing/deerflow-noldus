"""验证 catalog.resolve CLI 在 sandbox bash 替换后仍能产出虚拟路径。

G5 回归（thread 8ff3be6d dogfood）：
  lead 调 `python -m ethoinsight.catalog.resolve --virtual-workspace-dir /mnt/user-data/workspace ...`
  sandbox replace_virtual_paths_in_command 把 /mnt/user-data/workspace 翻译成物理路径
  CLI 收到的实际参数值已经是物理路径
  plan.json output 最终仍是物理路径，sandbox 抽象被打穿

本测试模拟 sandbox 替换后的命令行参数，并通过环境变量传 DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
（与 sandbox/tools.py:_build_path_env 注入的 key 完全一致）。CLI 必须从 env var 反推虚拟路径
作为兜底，覆盖被替换过的物理参数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ethoinsight.catalog.cli import main as cli_main


VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"


def _setup_inputs(tmp_path: Path) -> tuple[str, str, str, str, str]:
    """Create columns/raw_files/groups/output paths in tmp_path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    columns_file = workspace / "columns.json"
    columns_file.write_text(json.dumps({
        "columns": ["in_zone_open_arms_center", "in_zone_closed_arms_center"]
    }), encoding="utf-8")

    raw_files_json = workspace / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/mnt/user-data/uploads/dummy.txt"]), encoding="utf-8")

    groups_file = workspace / "groups.json"
    groups_file.write_text(json.dumps({}), encoding="utf-8")

    output = workspace / "metric_plan.json"

    return str(workspace), str(columns_file), str(raw_files_json), str(groups_file), str(output)


def test_cli_uses_env_var_when_virtual_workspace_dir_arg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟 sandbox 替换：lead 不传 --virtual-workspace-dir，env var 提供物理路径，
    CLI 从 env var key 反推虚拟路径用作 output。"""
    physical_workspace, columns_file, raw_files_json, groups_file, output = _setup_inputs(tmp_path)

    # sandbox 注入：env key = DEERFLOW_PATH_<虚拟路径转大写下划线>
    # 虚拟路径 /mnt/user-data/workspace → key=DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
    # value 是物理路径（被替换过的真实磁盘路径）
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", physical_workspace)

    exit_code = cli_main([
        "--paradigm", "epm",
        "--columns-file", columns_file,
        "--raw-files-json", raw_files_json,
        "--workspace-dir", physical_workspace,  # 已被 sandbox 替换为物理
        "--groups-file", groups_file,
        "--output", output,
        "--ev19-template", "PlusMaze-FewZones",
        # 故意不传 --virtual-workspace-dir，模拟 lead 已被引导停止传它
    ])
    assert exit_code == 0, "CLI 应成功退出"

    plan = json.loads(Path(output).read_text(encoding="utf-8"))
    for m in plan["metrics"]:
        assert m["output"].startswith(VIRTUAL_WORKSPACE), (
            f"metric {m['id']} output 不是虚拟路径: {m['output']}"
        )
        assert "/home/" not in m["output"], (
            f"metric {m['id']} output 含 host 物理路径: {m['output']}"
        )
        assert physical_workspace not in m["output"], (
            f"metric {m['id']} output 含物理 workspace 路径: {m['output']}"
        )


def test_cli_uses_env_var_when_virtual_workspace_dir_arg_replaced_to_physical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟 sandbox 替换：lead 传了 --virtual-workspace-dir /mnt/user-data/workspace，
    sandbox 把它替换成物理路径，env var 仍是虚拟路径解码的来源——env var 必须优先于
    被替换过的 --virtual-workspace-dir 参数。"""
    physical_workspace, columns_file, raw_files_json, groups_file, output = _setup_inputs(tmp_path)

    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", physical_workspace)

    exit_code = cli_main([
        "--paradigm", "epm",
        "--columns-file", columns_file,
        "--raw-files-json", raw_files_json,
        "--workspace-dir", physical_workspace,
        "--virtual-workspace-dir", physical_workspace,  # 模拟 sandbox 已替换
        "--groups-file", groups_file,
        "--output", output,
        "--ev19-template", "PlusMaze-FewZones",
    ])
    assert exit_code == 0

    plan = json.loads(Path(output).read_text(encoding="utf-8"))
    for m in plan["metrics"]:
        assert m["output"].startswith(VIRTUAL_WORKSPACE), (
            f"metric {m['id']} output 不是虚拟路径: {m['output']}"
        )


def test_cli_uses_explicit_virtual_workspace_dir_when_no_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """直接命令行调试（无 sandbox 包装）：env var 不存在，--virtual-workspace-dir
    若显式传入则使用它；保持非 sandbox 场景的兼容。"""
    physical_workspace, columns_file, raw_files_json, groups_file, output = _setup_inputs(tmp_path)

    # 确保 env var 不存在（防止其他测试污染）
    monkeypatch.delenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", raising=False)

    exit_code = cli_main([
        "--paradigm", "epm",
        "--columns-file", columns_file,
        "--raw-files-json", raw_files_json,
        "--workspace-dir", physical_workspace,
        "--virtual-workspace-dir", VIRTUAL_WORKSPACE,  # 显式传虚拟字符串（未被替换）
        "--groups-file", groups_file,
        "--output", output,
        "--ev19-template", "PlusMaze-FewZones",
    ])
    assert exit_code == 0

    plan = json.loads(Path(output).read_text(encoding="utf-8"))
    for m in plan["metrics"]:
        assert m["output"].startswith(VIRTUAL_WORKSPACE), (
            f"metric {m['id']} output 不是虚拟路径: {m['output']}"
        )


def test_cli_falls_back_to_workspace_dir_when_no_env_and_no_virtual_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两个 fallback 都缺失（直接调试场景 + 未传虚拟参数）：
    退化为 --workspace-dir 物理路径（保持现有兼容行为，不报错）。"""
    physical_workspace, columns_file, raw_files_json, groups_file, output = _setup_inputs(tmp_path)

    monkeypatch.delenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", raising=False)

    exit_code = cli_main([
        "--paradigm", "epm",
        "--columns-file", columns_file,
        "--raw-files-json", raw_files_json,
        "--workspace-dir", physical_workspace,
        "--groups-file", groups_file,
        "--output", output,
        "--ev19-template", "PlusMaze-FewZones",
    ])
    assert exit_code == 0

    plan = json.loads(Path(output).read_text(encoding="utf-8"))
    # 兜底用物理路径——不是修复目标，但确保非 sandbox 场景不崩
    for m in plan["metrics"]:
        assert m["output"].startswith(physical_workspace), m["output"]
