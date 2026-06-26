"""Task 3 (P1) 常驻守护测试：禁止散落的裸 /mnt 字符串拼接（spec 2026-06-26 §任务3）。

历史路径类 bug 大量来自工具/脚本各自拼 ``/mnt/...`` 字符串或假设脚本自己 resolve
（铁律 11、FileNotFoundError 家族）。DeerFlow 已有 ``Paths.resolve_virtual_path`` +
ethoinsight ``_cli.resolve_sandbox_path`` 解析单点。本测试把「别手拼 /mnt 当主机路径」
从惯例升级为 CI 可抓的约束。

判据（精确区分「合法虚拟路径契约」与「违规主机路径拼接」）：

- **违规**：在主机侧把 ``/mnt/...`` 当真实文件系统路径直接操作——
  ``Path("/mnt/...")``、``open("/mnt/...")``、``os.*("/mnt/...")``、
  ``shutil.*("/mnt/...")``。这些绕过解析单点，在主机上必 FileNotFoundError
  或读到错位文件。
- **合法**：把 ``/mnt/...`` 当**虚拟路径契约**产出/传递——
  default 参数（``plan_path: str = "/mnt/user-data/workspace/..."``）、
  docstring 示例、handoff/状态 JSON 里写给 agent 看的虚拟路径、
  ``f"/mnt/shared/{name}"`` 这类 placeholder 还原（还原后仍是虚拟路径，由
  read_file 经 resolve_virtual_path 解析）。这些不是主机路径操作。

实现走 ripgrep / grep 扫关键源目录，断言违规模式集合为空。已知合法引用进白名单
（按行 match），新增合法点需显式登记——这正是「守」的力度。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# 关键源目录：工具 / sandbox / 持久化 / runtime。paths.py 是解析单点本体，不扫。
_BACKEND = Path(__file__).resolve().parents[1]
_HARNESS = _BACKEND / "packages" / "harness" / "deerflow"

_SCAN_DIRS = [
    _HARNESS / "tools" / "builtins",
    _HARNESS / "sandbox",
    _HARNESS / "subagents",
    _HARNESS / "agents",
    _HARNESS / "runtime",
    _BACKEND / "app" / "gateway" / "routers",
]

# 主机侧文件系统操作 + 裸 /mnt 字面量的组合 = 违规。捕获：
#   Path("/mnt/..."), open("/mnt/..."), os.path.*("/mnt/..."),
#   os.makedirs("/mnt/..."), shutil.*("/mnt/..."), .exists("/mnt/...") 等。
# 用 word-boundary 的函数名前缀 + 紧跟的 /mnt 字面量。
_VIOLATION_RE = re.compile(
    r"""
    (?:
        \bPath\s*\(\s*["']                  # Path("/mnt/..."
      | \bopen\s*\(\s*["']                  # open("/mnt/..."
      | \bos\.[a-z_.]+\s*\(\s*["']          # os.path.exists("/mnt/..."
      | \bshutil\.[a-z_]+\s*\(\s*["']       # shutil.rmtree("/mnt/..."
      | \.read_text\s*\(\s*\)\s*$           # 已在 Path(...) 之外的形态由 Path 模式覆盖
    )
    /mnt/
    """,
    re.VERBOSE,
)

# 已知合法引用（按 相对路径:行号 或 相对路径:内容片段 match）。这些是把 /mnt
# 当虚拟路径契约产出/传递，非主机路径操作。新增合法点须在此登记。
_ALLOWLIST_PATTERNS = [
    # 文档字符串 / 注释里给 agent 看的虚拟路径示例。
    r"#.*/mnt/",
    r'""".*/mnt/',
    r"^\s*>>>.*",  # doctest 示例
]


def _iter_py_files(root: Path):
    if not root.exists():
        return
    # ripgrep 不可用时退化到 walk；优先 rg（快、自动跳过 .venv）。
    for p in root.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        yield p


def _find_violations() -> list[str]:
    violations: list[str] = []
    for d in _SCAN_DIRS:
        for py in _iter_py_files(d):
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not _VIOLATION_RE.search(line):
                    continue
                if any(re.search(p, line) for p in _ALLOWLIST_PATTERNS):
                    continue
                rel = py.relative_to(_BACKEND)
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def test_no_raw_mnt_host_path_concatenation() -> None:
    """关键源目录不得出现把 /mnt 当主机路径直接操作的新增代码。

    违规示例（必须改走 resolve_virtual_path / resolve_sandbox_path）：
        Path("/mnt/user-data/outputs/x.png").read_text()   # ❌ 主机上不存在
        open("/mnt/user-data/workspace/plan.json")         # ❌
        os.path.exists("/mnt/shared/foo")                  # ❌

    合法：虚拟路径当契约产出（default 参数 / docstring / JSON 字段）不在此列。
    若新增的是合法虚拟路径产出，登记到 _ALLOWLIST_PATTERNS 并在此说明理由。
    """
    violations = _find_violations()
    if violations:
        msgs = "\n".join(violations)
        raise AssertionError(
            "发现裸 /mnt 主机路径操作（spec 2026-06-26 §任务3 守护）。"
            "主机侧读 /mnt 必 FileNotFoundError 或读错位文件——改走解析单点"
            "Paths.resolve_virtual_path 或 ethoinsight _cli.resolve_sandbox_path：\n" + msgs
        )


def test_resolve_single_point_exists() -> None:
    """解析单点本体存在性回归：两个 resolve 函数必须仍可用。

    守护测试本身依赖单点存在；若上游 sync 重命名/移走了 resolve_virtual_path，
    这里先红，提示重接单点（而非把守护测试改成扫别的）。
    """
    from deerflow.config.paths import Paths  # noqa: F401  — 单点本体须可导入

    assert hasattr(Paths, "resolve_virtual_path"), "Paths.resolve_virtual_path 单点缺失"

    # ethoinsight 解析单点（惰性，可能未装）——只在能 import 时校验。
    try:
        from ethoinsight.scripts._cli import resolve_sandbox_path  # noqa: F401
    except ImportError:
        pass  # ethoinsight 未装时跳过（harness 必须能纯导入，守 CLAUDE.md 铁律）


if __name__ == "__main__":
    # 手动跑：python tests/test_no_raw_mnt_path.py → 列出违规。
    v = _find_violations()
    print("\n".join(v) if v else "no violations")
    sys.exit(1 if v else 0)
