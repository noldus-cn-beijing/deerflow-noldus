"""2026-06-15 spec #2: compute 脚本经 I/O sink resolve /mnt 虚拟路径。

背景：``run_metric_plan``（S4）改成进程内 ProcessPoolExecutor 跑 compute 脚本，
无 bash sandbox 的 mount。脚本内 ``parse_trajectory(args.input)`` +
``save_output_json(args.output)`` 拿到原样 ``/mnt/...`` 虚拟路径 → FileNotFoundError。

修复：在 I/O 边界（``_parse_path_and_sheet`` 入口 + ``save_output_json`` 入口）
resolve ``/mnt`` 虚拟路径，与 ``read_inputs_json``/``read_groups_json`` 同一处方对称。
``resolve_sandbox_path`` 是 fail-safe 幂等的（真实路径原样返回、/mnt 匹配不到 env 也
原样返回），所以对现有喂真实路径/bash-mounted 路径的调用方零行为变化。

red 锚点：修复前 ``parse_trajectory('/mnt/user-data/uploads/x.xlsx')`` +
设 ``DEERFLOW_PATH_MNT_USER_DATA_UPLOADS`` 直接 FileNotFoundError（脚本拿原样
/mnt 路径）；修复后经 ``_parse_path_and_sheet`` resolve 能读到真实文件。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import save_output_json

# 真实 EV19 xlsx fixture（多空格文件名，用 glob 取，避免手敲）。
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_EPM_XLSX_GLOB = list((_FIXTURES_DIR).glob("原始数据-Elevated Plus Maze*.xlsx"))


def _real_epm_xlsx() -> Path:
    assert _EPM_XLSX_GLOB, "missing EPM xlsx fixture under tests/fixtures/"
    return _EPM_XLSX_GLOB[0]


# /mnt/user-data/uploads → DEERFLOW_PATH_MNT_USER_DATA_UPLOADS
# /mnt/user-data/workspace → DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
# (env key 规则见 scripts/_cli.py:_sandbox_env_key_for_prefix)


def test_parse_trajectory_resolves_mnt_path(tmp_path, monkeypatch):
    """parse_trajectory 经 _parse_path_and_sheet resolve /mnt uploads 路径能读到真实文件。"""
    real_uploads = tmp_path / "real_uploads"
    real_uploads.mkdir()
    real_file = real_uploads / "x.xlsx"
    real_file.write_bytes(_real_epm_xlsx().read_bytes())

    # /mnt/user-data/uploads/x.xlsx → real_uploads/x.xlsx
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(real_uploads))

    # 修复前：FileNotFoundError（脚本拿原样 /mnt/user-data/uploads/x.xlsx）。
    # 修复后：经 _parse_path_and_sheet resolve 成 real_uploads/x.xlsx 能读。
    df = parse_trajectory("/mnt/user-data/uploads/x.xlsx")
    assert len(df) > 0


def test_save_output_json_resolves_mnt_path(tmp_path, monkeypatch):
    """save_output_json 经入口 resolve /mnt workspace 路径写到真实目录。"""
    real_ws = tmp_path / "real_workspace"
    real_ws.mkdir()
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real_ws))

    # 修复前：写到不存在的 /mnt/user-data/workspace/m_test.json → parent mkdir 在
    # /mnt 下失败 / 文件落不到真实目录。修复后：resolve 成 real_ws/m_test.json。
    save_output_json("/mnt/user-data/workspace/m_test.json", {"value": 1.0})
    assert (real_ws / "m_test.json").exists()


def test_resolve_is_idempotent_and_failsafe(tmp_path):
    """守护：真实路径原样、/mnt 无 env 原样（不引入新失败）——对现有调用方零影响。"""
    # 真实路径，无 env：照常写（不因 resolve 改行为）
    out = tmp_path / "m.json"
    save_output_json(out, {"value": 2.0})
    assert out.exists()

    # /mnt 路径但无 DEERFLOW_PATH_* env → fail-safe 原样返回，不会抛也不会误解析。
    # 这里只验证 resolve_sandbox_path 的契约本身（save_output_json 会因 /mnt 真不存在
    # 而写失败，所以单独测 resolve 契约更干净）。
    from ethoinsight.scripts._cli import resolve_sandbox_path

    assert str(resolve_sandbox_path("/mnt/user-data/workspace/no_env.json")) == \
        "/mnt/user-data/workspace/no_env.json"
    # 真实路径原样
    real = tmp_path / "real.txt"
    assert resolve_sandbox_path(str(real)) == real


def test_parse_trajectory_xlsx_with_sheet_suffix_resolves(tmp_path, monkeypatch):
    """守护：path::sheet 形式也要先 resolve 再切 sheet（:: 不被 resolve 误处理）。

    验证路径部分被正确 resolve（文件能被打开），sheet 分离后由 pandas 处理。
    用不存在的 sheet 名 → 应报 "Worksheet not found"（路径层 OK），而非
    FileNotFoundError（路径未 resolve 的特征）——这区分证明 resolve 生效。
    """
    real_uploads = tmp_path / "real_uploads"
    real_uploads.mkdir()
    real_file = real_uploads / "x.xlsx"
    real_file.write_bytes(_real_epm_xlsx().read_bytes())

    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(real_uploads))

    # /mnt 虚拟路径 + ::NopeSheet：resolve 只作用路径部分，sheet 分离保留。
    # 若路径未被 resolve，会抛 FileNotFoundError（/mnt/user-data/...不存在）。
    # 路径被 resolve 后，pandas 打开文件成功，但找不到 sheet → ValueError sheet。
    with pytest.raises(ValueError, match="[Ww]orksheet"):
        parse_trajectory("/mnt/user-data/uploads/x.xlsx::NopeSheet")
