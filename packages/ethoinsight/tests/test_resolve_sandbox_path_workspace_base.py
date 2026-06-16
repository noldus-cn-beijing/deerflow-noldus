"""2026-06-16 spec 层 A red 锚点（虚拟路径解析故障族根治）：

机制 B（ethoinsight ``resolve_sandbox_path``）依赖 ``DEERFLOW_PATH_*`` env 才能解析
``/mnt/...`` 虚拟路径。当 ethoinsight 函数在 **harness 进程内被直接调用**（不经沙箱子
进程、不设 env）时，``resolve_sandbox_path`` 静默原样返回 ``/mnt`` 路径 → 读不到 →
下游静默退化（统计跳过 / 校验误报 / handoff 残缺）。#5 / #6a 是该族的两个已点修实例。

根治：``resolve_sandbox_path`` 接受可选 ``workspace_base``——无 env 时用真实 workspace
物理路径兜底解析 workspace 前缀的虚拟路径。这让"进程内调 ethoinsight 读 workspace /mnt
文件"不再依赖调用方记得设 env。

本测试用 importlib 加载 worktree 源：worktree 共享主仓 venv，editable ethoinsight 指向
主仓源，直接 ``from ethoinsight...import`` 测主仓代码（worktree 改动不生效=假绿）。
守 feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL _cli.py source from this worktree (bypass editable link → main repo)
# ---------------------------------------------------------------------------
_CLI_FILE = (
    Path(__file__).resolve().parents[1] / "ethoinsight" / "scripts" / "_cli.py"
)

_REAL_CLI: ModuleType | None = None


def _get_cli() -> ModuleType:
    global _REAL_CLI
    if _REAL_CLI is not None:
        return _REAL_CLI

    assert _CLI_FILE.exists(), f"_cli.py not found at {_CLI_FILE}"
    spec = importlib.util.spec_from_file_location(
        "ethoinsight.scripts._cli_worktree_vpr", _CLI_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:  # pragma: no cover - 加载失败说明环境异常
        pytest.skip(f"Could not load worktree _cli.py: {e}")
    _REAL_CLI = module
    return _REAL_CLI


ENV_KEY_WORKSPACE = "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE"


class TestResolveSandboxPathWorkspaceBase:
    """改动①：workspace_base 兜底 + env 优先 + fail-safe 守护。"""

    def test_env_priority_workspace_base_ignored_when_env_set(
        self, tmp_path, monkeypatch
    ):
        """env 设了 → 用 env（workspace_base 不参与，沙箱子进程路径不变）。"""
        cli = _get_cli()
        real = tmp_path / "ws_env"
        real.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real))

        decoy = tmp_path / "ws_base"
        decoy.mkdir()

        got = cli.resolve_sandbox_path(
            "/mnt/user-data/workspace/g.json", workspace_base=str(decoy)
        )
        assert got == real / "g.json"

    def test_workspace_base_fallback_when_no_env(self, tmp_path, monkeypatch):
        """red 锚点：无 env + 有 workspace_base → 兜底解析。

        修复前：无 env → 原样返回 /mnt（读不到，下游静默退化）。
        修复后：用 workspace_base 拼后缀解析成真实 workspace 物理路径。
        """
        cli = _get_cli()
        monkeypatch.delenv(ENV_KEY_WORKSPACE, raising=False)
        real = tmp_path / "ws"
        real.mkdir()

        got = cli.resolve_sandbox_path(
            "/mnt/user-data/workspace/g.json", workspace_base=str(real)
        )
        assert got == real / "g.json"

    def test_workspace_base_fallback_exact_prefix(self, tmp_path, monkeypatch):
        """无 env + workspace_base + 精确 workspace 前缀（无后缀）→ 返回 base 本身。"""
        cli = _get_cli()
        monkeypatch.delenv(ENV_KEY_WORKSPACE, raising=False)
        real = tmp_path / "ws"
        real.mkdir()

        got = cli.resolve_sandbox_path(
            "/mnt/user-data/workspace", workspace_base=str(real)
        )
        assert got == real

    def test_workspace_base_only_for_workspace_prefix(self, tmp_path, monkeypatch):
        """守护：workspace_base 只兜底 workspace 前缀，不泛化到 uploads/outputs。

        无 env + workspace_base + /mnt/uploads 路径 → 仍 fail-safe 原样返回
        （uploads 兜底由 prep 已做的 replace_virtual_path 在传入前解决，不在此兜底）。
        """
        cli = _get_cli()
        for k in list(__import__("os").environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)
        real = tmp_path / "ws"
        real.mkdir()

        got = cli.resolve_sandbox_path(
            "/mnt/user-data/uploads/sub.txt", workspace_base=str(real)
        )
        assert str(got) == "/mnt/user-data/uploads/sub.txt"

    def test_no_env_no_base_passthrough(self, monkeypatch):
        """守护：无 env 无 workspace_base → 原样返回（非沙箱/测试合法路径，不引入新失败）。"""
        cli = _get_cli()
        monkeypatch.delenv(ENV_KEY_WORKSPACE, raising=False)

        got = cli.resolve_sandbox_path("/mnt/user-data/workspace/g.json")
        assert str(got) == "/mnt/user-data/workspace/g.json"

    def test_real_path_unchanged(self, tmp_path):
        """守护：真实路径原样返回（fail-safe 幂等），workspace_base 不影响真实路径。"""
        cli = _get_cli()
        f = tmp_path / "x.json"
        assert cli.resolve_sandbox_path(str(f)) == f
        # workspace_base 对真实路径无影响
        assert cli.resolve_sandbox_path(str(f), workspace_base=str(tmp_path)) == f

    def test_backward_compat_no_workspace_base_kwarg(self, tmp_path, monkeypatch):
        """守护：不传 workspace_base（旧行为）零变化——env 路径与 fail-safe 都不变。"""
        cli = _get_cli()
        real = tmp_path / "ws"
        real.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real))
        # 旧调用形态（仅 path，无 kwargs）
        assert cli.resolve_sandbox_path("/mnt/user-data/workspace/g.json") == real / "g.json"


class TestEnvPathUnchangedOnWorktreeSource:
    """改动①红线：env 优先路径字节不变——在 worktree 源上复测 env 路径族行为
    （沙箱子进程/进程池 worker 行为不能变）。这些场景主仓测试已覆盖，此处用 importlib
    在 worktree 源上独立坐实，防止改动①的 workspace_base 分支误伤 env 路径。"""

    def test_env_longest_prefix_wins(self, tmp_path, monkeypatch):
        """workspace env（长前缀）优先于 user-data env（短前缀）。"""
        cli = _get_cli()
        for k in list(__import__("os").environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)
        real_ws = tmp_path / "ws"
        real_ud = tmp_path / "ud"
        real_ws.mkdir()
        real_ud.mkdir()
        monkeypatch.setenv(ENV_KEY_WORKSPACE, str(real_ws))
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA", str(real_ud))

        got = cli.resolve_sandbox_path("/mnt/user-data/workspace/m.json")
        assert got == real_ws / "m.json"
        assert str(got) != str(real_ud / "workspace" / "m.json")

    def test_env_fallback_to_shorter_prefix(self, tmp_path, monkeypatch):
        """workspace 无 env → 退化到 user-data env（workspace_base 不该抢占 env 路径）。"""
        cli = _get_cli()
        for k in list(__import__("os").environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)
        real_ud = tmp_path / "ud"
        real_ud.mkdir()
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA", str(real_ud))
        # workspace env 未设；给一个 workspace_base，证 env（user-data）仍优先于 base
        decoy = tmp_path / "decoy_ws"
        decoy.mkdir()

        got = cli.resolve_sandbox_path(
            "/mnt/user-data/workspace/m.json", workspace_base=str(decoy)
        )
        # user-data env 命中（短前缀匹配），workspace_base 不参与
        assert got == real_ud / "workspace" / "m.json"
