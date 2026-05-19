# G5 catalog 虚拟路径回归修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `metric_plan.json` 的 `output` 字段在生产环境（lead → bash → sandbox → CLI）下仍然写入 host 物理路径的回归——commit `2eb1532a` 修了 Python 层，但 sandbox 层把 `--virtual-workspace-dir "/mnt/user-data/workspace"` 参数值翻译成了物理路径，让 Python 层根本没机会用上虚拟路径。

**Architecture:** **改用环境变量传递虚拟路径，而不是命令行参数**。Sandbox 已经在执行 bash 时注入 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` env var（[`sandbox/tools.py:507-520`](../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py)），值是物理路径但 **key 名稳定地编码了虚拟路径** —— 反推一下就拿到虚拟路径字符串。CLI 在 `--virtual-workspace-dir` 未传时，从 env var 反推出虚拟路径作为 fallback；这样即便 lead 传了被替换过的物理路径，env var 这条带外通道仍然能拿到正确的虚拟路径。

**Tech Stack:** Python 3.10+ (ethoinsight)、Python 3.12+ (agent backend)、pytest、ruff (line length 240)、deerflow `_build_path_env` 注入机制（已存在）

**Context links（实现时必读）：**
- 诊断材料：`docs/problems/2026-05-14-G5-catalog-virtual-path-regression.md`（解释为什么 commit 2eb1532a 单测过但生产挂）
- spec：`docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` §10.3.1（本 plan 同步修补其论证缺陷）
- 现有 CLI：`packages/ethoinsight/ethoinsight/catalog/cli.py`（要改）
- 现有 resolve：`packages/ethoinsight/ethoinsight/catalog/resolve.py`（不改，已正确）
- 现有单测：`packages/ethoinsight/tests/test_catalog_resolve_paths.py`（保留 + 加新测）
- Sandbox env var 注入：`packages/agent/backend/packages/harness/deerflow/sandbox/tools.py:507-520`
- Lead prompt 例子：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:1112-1117`
- Skill 文档：`packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md:46-62` + `references/resolve-cli.md:1-11`
- 项目约束：`CLAUDE.md`（TDD 强制 / 中文 commit / ruff line length 240）

**Scope（明确不做）：**
- 不动 `resolve.py` 的 Python 函数逻辑——它正确，只是上层喂错参数
- 不改 sandbox `replace_virtual_paths_in_command`（spec §10.3 论证不动 sandbox，本 plan 用 env-var 旁路而非修 sandbox）
- 不修 G4（阶段播报失效）和 G1（subagent 透传违规话术）——独立 plan
- 不删 `--virtual-workspace-dir` 参数——保留向后兼容（直接命令行调试 / 单测时仍可用），仅改"未传时 fallback 优先级"
- 不动其他 ethoinsight CLI（`parse.dump_headers` 等）—— 它们只用 input/output 路径作文件操作目标、不作为 metadata 写入 plan.json，不受影响

**前置假设（执行前用 `git log -1` 验证）：**
- 当前在 `dev` 分支，工作目录 `/home/wangqiuyang/noldus-insight/`
- dev 已和 origin 同步（最新 commit 是 `docs(problems): G5 catalog 路径回归根因诊断`）
- backend / ethoinsight 测试在本基线全绿（执行前先各跑一次 `make test` / `pytest tests/` 确认）

---

## File Structure（决策已锁定，照此实施）

**修改 2 个文件**（生产代码）：
- `packages/ethoinsight/ethoinsight/catalog/cli.py` — 加 env-var fallback 逻辑（~15 行新代码 + docstring 更新）
- `packages/ethoinsight/ethoinsight/catalog/resolve.py` — **不改函数逻辑**，只更新 docstring 说明 env-var 来源（~5 行 docstring）

**新建 1 个文件**（测试代码）：
- `packages/ethoinsight/tests/test_catalog_resolve_env_var.py` — 覆盖 env-var fallback 三种场景（~80 行）

**修改 1 个文件**（既有测试）：
- `packages/ethoinsight/tests/test_catalog_resolve_paths.py` — 不删，保留作为 Python 层不变契约

**修改 2 个文件**（文档 / prompt）：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:1112-1117` — Step 0.5 说明改用 env-var fallback，**lead 不再需要传 `--virtual-workspace-dir`**（保留兼容、可以不传）
- `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md:46-62` + `references/resolve-cli.md:1-11` — 示例命令删除 `--virtual-workspace-dir /mnt/user-data/workspace`，文档说明该参数现以 env var 为 fallback

**修改 1 个文件**（spec 补论证）：
- `docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` §10.3.1 — 补论证 catalog CLI 路径仍受 sandbox replace 影响，env-var 是已落地的修复模式

**不改的关键文件**（防止 scope creep）：
- `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` — sandbox 不动
- `packages/ethoinsight/ethoinsight/catalog/resolve.py` 函数体 — 函数对了，只动 docstring
- `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py` — 上一 plan 产物，本 plan 不动
- 任何 prompt 里"角色边界 / 失败处理"段落 — 不动
- `packages/ethoinsight/tests/test_catalog_resolve_paths.py` — 保留作为 Python 层契约

---

### Task 1: 写失败测试 — 覆盖 sandbox bash 替换层后 CLI 仍能输出虚拟路径（TDD 先红）

**Files:**
- Create: `packages/ethoinsight/tests/test_catalog_resolve_env_var.py`

**为什么先写测试**：诊断材料指出根因正是"单测 mock 了 sandbox 替换这一层、bug 藏在 mock 之下"。本 task 写的测试**必须**模拟 sandbox 替换效果，否则修复又是空修。

- [ ] **Step 1: 写完整测试文件**

完整内容：

```python
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


def _setup_inputs(tmp_path: Path) -> tuple[str, str, str, str]:
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
```

- [ ] **Step 2: 运行测试，确认前 3 个测试失败、第 4 个通过**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_catalog_resolve_env_var.py -v
```

Expected:
- `test_cli_uses_env_var_when_virtual_workspace_dir_arg_missing` FAIL（CLI 现在没读 env var，会用 workspace_dir 物理路径）
- `test_cli_uses_env_var_when_virtual_workspace_dir_arg_replaced_to_physical` FAIL（同上）
- `test_cli_uses_explicit_virtual_workspace_dir_when_no_env_var` PASS（这是现有行为）
- `test_cli_falls_back_to_workspace_dir_when_no_env_and_no_virtual_arg` PASS（这是现有兜底行为）

- [ ] **Step 3: Commit 失败测试**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/tests/test_catalog_resolve_env_var.py
git commit -m "$(cat <<'EOF'
test(catalog): 加 env-var fallback 失败测试（先红后绿，TDD）

覆盖 G5 回归根因——sandbox bash 替换把 --virtual-workspace-dir 参数值翻译成物理路径，
让 commit 2eb1532a 的 Python 层修复无法生效。本测试通过 monkeypatch 注入 sandbox 同款
env var DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 来模拟 sandbox 行为，证明 CLI 必须从
env var key 反推虚拟路径才能修复回归。

4 个场景：env var 唯一来源 / env var 优先于被替换的参数 / 直接调试无 env / 都缺失兜底。
前 2 个红、后 2 个绿。
EOF
)"
```

---

### Task 2: 加 env-var fallback 逻辑到 CLI（绿）

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/cli.py` (docstring + parser default + main 调用)

- [ ] **Step 1: 改写 cli.py 完整文件**

用 Edit 工具，**把整个文件替换**为以下内容（覆盖现有 133 行，新版 ~160 行）：

```python
"""Catalog CLI entry — python -m ethoinsight.catalog.resolve.

参数:
  --paradigm PARADIGM            必填
  --columns-file PATH            必填（dump_headers 产物）
  --raw-files-json PATH          必填（指向 JSON 数组）
  --groups-file PATH             可选
  --workspace-dir PATH           必填（物理路径，用于脚本执行）
  --virtual-workspace-dir PATH   可选（plan.json output 虚拟路径）
                                  fallback 顺序：
                                    1. 显式传入的 --virtual-workspace-dir 值（如果不是物理路径）
                                    2. 环境变量 DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 的 key 反推
                                       （sandbox 注入的标准 env var，详见 deerflow.sandbox.tools._build_path_env）
                                    3. 兜底使用 --workspace-dir（非 sandbox 调试场景）
  --include METRIC_ID            可重复
  --exclude METRIC_ID            可重复
  --n-per-group INT              可选
  --n-groups INT                 可选
  --ev19-template TEMPLATE_ID    可选
  --output PATH                  必填，写 plan.json 路径

行为:
  成功: exit 0 + 写 plan.json
  失败: exit 1 + stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}

G5 回归修复（thread 8ff3be6d）:
  sandbox replace_virtual_paths_in_command 会把命令字符串里所有 /mnt/user-data/...
  字面量翻译成物理路径——包括 --virtual-workspace-dir 参数的值。修复方案是改用
  sandbox 已经注入的 env var 作为虚拟路径来源（env var 的 key 名稳定编码虚拟路径，
  即便 value 是物理路径，key 本身仍可反推回虚拟字符串）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve


# 与 deerflow.sandbox.tools._build_path_env 完全一致的命名规则：
# 虚拟路径 /mnt/user-data/workspace → env key DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
_WORKSPACE_VIRTUAL_PATH = "/mnt/user-data/workspace"
_WORKSPACE_ENV_KEY = "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE"


def _resolve_virtual_workspace_dir(arg_value: str | None, workspace_dir: str) -> str:
    """决定 plan.json output 字段用的"虚拟"路径，三级 fallback：

    1. 如果 arg_value 已经是虚拟路径前缀（/mnt/user-data/...），用 arg_value
       （兼容直接命令行调试时显式传入虚拟字符串的场景）
    2. 否则如果 env var DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 存在，
       用 _WORKSPACE_VIRTUAL_PATH（env key 反推的稳定虚拟字符串）
    3. 否则兜底用 workspace_dir（非 sandbox 调试场景，行为同 commit 2eb1532a 之前）

    为什么 env var 的 VALUE 不直接当虚拟路径用：
      sandbox 注入的 env var value 是物理路径（用于 Python 脚本运行时读真实文件），
      不是虚拟路径。但 env var 的 KEY 是稳定的——它的存在本身就证明"sandbox 抽象在生效"，
      此时我们可以用代码中硬编码的虚拟路径常量 _WORKSPACE_VIRTUAL_PATH 作为返回值。
    """
    # 1) 显式传入的虚拟路径优先（直接调试场景）
    if arg_value and arg_value.startswith(_WORKSPACE_VIRTUAL_PATH):
        return arg_value

    # 2) sandbox 模式：env var 存在 → 用硬编码虚拟路径
    if _WORKSPACE_ENV_KEY in os.environ:
        return _WORKSPACE_VIRTUAL_PATH

    # 3) 非 sandbox 调试场景：兜底物理 workspace_dir
    return arg_value or workspace_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.resolve")
    p.add_argument("--paradigm", required=True)
    p.add_argument("--columns-file", required=True)
    p.add_argument("--raw-files-json", required=True)
    p.add_argument("--workspace-dir", required=True)
    p.add_argument("--virtual-workspace-dir", default=None)
    p.add_argument("--groups-file", default=None)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--n-per-group", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--ev19-template", default=None)
    p.add_argument("--output", required=True)
    return p


def _emit_error(code: str, message: str, details: dict | None = None) -> int:
    payload = {"code": code, "message": message, "details": details or {}}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Read columns.json
    try:
        columns_data = json.loads(Path(args.columns_file).read_text(encoding="utf-8"))
        columns = columns_data.get("columns", [])
        if not isinstance(columns, list):
            return _emit_error(
                "schema_violation",
                f"columns-file does not contain a 'columns' list: {args.columns_file}",
                {"path": args.columns_file},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read columns-file: {e}",
            {"path": args.columns_file},
        )

    # Read raw-files-json
    try:
        raw_files = json.loads(Path(args.raw_files_json).read_text(encoding="utf-8"))
        if not isinstance(raw_files, list) or not all(
            isinstance(p, str) for p in raw_files
        ):
            return _emit_error(
                "schema_violation",
                f"raw-files-json must be a JSON array of strings: {args.raw_files_json}",
                {"path": args.raw_files_json},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read raw-files-json: {e}",
            {"path": args.raw_files_json},
        )

    # 三级 fallback 决定 plan.json output 用的虚拟路径
    virtual_workspace_dir = _resolve_virtual_workspace_dir(
        args.virtual_workspace_dir, args.workspace_dir
    )

    try:
        plan = resolve(
            paradigm=args.paradigm,
            columns=columns,
            raw_files=raw_files,
            workspace_dir=args.workspace_dir,
            include=args.include,
            exclude=args.exclude,
            n_per_group=args.n_per_group,
            n_groups=args.n_groups,
            groups_file=args.groups_file,
            columns_file=args.columns_file,
            ev19_template=args.ev19_template,
            virtual_workspace_dir=virtual_workspace_dir,
        )
    except ResolveError as e:
        return _emit_error(e.code, str(e), e.details)
    except Exception as e:
        return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

    # Write plan.json
    plan_dict = plan_to_dict(plan)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = (
        f"Plan written to {args.output}: paradigm={plan.paradigm}, "
        f"metrics={len(plan.metrics)}, charts={len(plan.charts)}, "
        f"skipped={len(plan.skipped)}, statistics="
        f"{'skip' if (plan.statistics and plan.statistics.skip_reason) else 'run' if plan.statistics else 'none'}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**关键变更**：
- 新增 `_resolve_virtual_workspace_dir(arg_value, workspace_dir)` 函数（三级 fallback）
- 新增 `_WORKSPACE_VIRTUAL_PATH` / `_WORKSPACE_ENV_KEY` 常量
- `main()` 调 `_resolve_virtual_workspace_dir` 决定 virtual_workspace_dir，不再直接用 `args.virtual_workspace_dir or args.workspace_dir`
- docstring 解释新 fallback 顺序 + G5 修复来源

- [ ] **Step 2: 跑新单测，确认全 4 个绿**

Run:
```bash
cd /home/wangqiuyang/noldus-insight
pytest packages/ethoinsight/tests/test_catalog_resolve_env_var.py -v
```

Expected: 4/4 PASS。

- [ ] **Step 3: 跑既有单测（test_catalog_resolve_paths.py），确认不回归**

Run:
```bash
cd /home/wangqiuyang/noldus-insight
pytest packages/ethoinsight/tests/test_catalog_resolve_paths.py -v
```

Expected: 2/2 PASS（既有 Python 层契约保持）。

- [ ] **Step 4: 跑 ethoinsight 全量单测**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/
```

Expected: 全绿（除已知 9 个预存 parse 失败，与本 plan 无关）。

- [ ] **Step 5: Commit CLI 修复**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/cli.py
git commit -m "$(cat <<'EOF'
fix(catalog): G5 修复 — CLI 改用 env var 兜底虚拟路径

thread 8ff3be6d dogfood 暴露 commit 2eb1532a 的修复在生产环境失效——
sandbox replace_virtual_paths_in_command 把 --virtual-workspace-dir 参数值翻译成
物理路径，让 Python 层 virtual_workspace_dir or workspace_dir 兜底没机会用上。

修复方案：CLI 加 _resolve_virtual_workspace_dir() 三级 fallback：
  1. 显式 --virtual-workspace-dir 是虚拟路径前缀 → 用它（直接调试场景）
  2. env var DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 存在 → 用硬编码 /mnt/user-data/workspace
     （sandbox 模式：env var key 反推虚拟路径，绕过被替换的参数值）
  3. 兜底用 --workspace-dir（非 sandbox 调试场景）

未改 resolve.py 函数逻辑（它本身是对的），只动 CLI 入口的参数解析。
不改 sandbox（spec §10.3 论证不动 sandbox，用 env-var 旁路替代）。
EOF
)"
```

---

### Task 3: 同步 lead prompt + skill 文档（删示例中的 --virtual-workspace-dir）

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md:46-62`
- Modify: `packages/agent/skills/custom/ethoinsight-metric-catalog/references/resolve-cli.md:11`

**为什么改文档而不是改 prompt**：lead 的 catalog.resolve 调用例子在 skill 里（SKILL.md `Step 2: 生成执行计划`），不在 lead_agent/prompt.py 主体。skill 是渐进披露注入的——改 skill 例子让 lead 不再传 `--virtual-workspace-dir`，让 lead 命令更短、更不容易被 sandbox 误判。**lead_agent/prompt.py:1112-1117 段不动**——它只说"bash catalog.resolve 生成 plan.json"，没给具体参数。

- [ ] **Step 1: 改 skill 主文档**

用 Edit 工具修改 `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`：

old_string：
```
```bash
python -m ethoinsight.catalog.resolve \
    --paradigm <epm|oft|fst|...> \
    --columns-file /mnt/user-data/workspace/columns.json \
    --raw-files-json /mnt/user-data/workspace/raw_files.json \
    --workspace-dir /mnt/user-data/workspace \
    --virtual-workspace-dir /mnt/user-data/workspace \
    --groups-file /mnt/user-data/workspace/groups.json \
    --output /mnt/user-data/workspace/metric_plan.json \
    [--include METRIC_ID]* \
    [--exclude METRIC_ID]* \
    [--n-per-group N] \
    [--n-groups N] \
    [--ev19-template TEMPLATE_ID]
```
```

new_string：
```
```bash
python -m ethoinsight.catalog.resolve \
    --paradigm <epm|oft|fst|...> \
    --columns-file /mnt/user-data/workspace/columns.json \
    --raw-files-json /mnt/user-data/workspace/raw_files.json \
    --workspace-dir /mnt/user-data/workspace \
    --groups-file /mnt/user-data/workspace/groups.json \
    --output /mnt/user-data/workspace/metric_plan.json \
    [--include METRIC_ID]* \
    [--exclude METRIC_ID]* \
    [--n-per-group N] \
    [--n-groups N] \
    [--ev19-template TEMPLATE_ID]
```

> 注：早期版本要求显式传 `--virtual-workspace-dir /mnt/user-data/workspace`。
> 现已改用 sandbox 注入的 env var（`DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE`）作为兜底，
> lead 无需再传该参数——sandbox 会自动确保 plan.json output 字段是虚拟路径。
> 该参数仍保留以兼容直接命令行调试（无 sandbox env var 的场景）。
```

- [ ] **Step 2: 改 skill 参数文档（references/resolve-cli.md）**

用 Edit 工具修改 `packages/agent/skills/custom/ethoinsight-metric-catalog/references/resolve-cli.md`：

old_string：
```
| `--virtual-workspace-dir` | path | plan.json output 字段使用的虚拟路径（面向 downstream subagent），未提供时兜底用 `--workspace-dir` |
```

new_string：
```
| `--virtual-workspace-dir` | path | plan.json output 字段使用的虚拟路径。**通常无需在 sandbox 中显式传入**——CLI 优先读 env var `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` 自动确定。显式传入只用于直接命令行调试（无 sandbox 环境）。fallback 顺序：(1) 显式 `--virtual-workspace-dir` 是虚拟路径 → 用它 (2) env var 存在 → 用硬编码 `/mnt/user-data/workspace` (3) 兜底物理 `--workspace-dir` |
```

- [ ] **Step 3: 重启服务跑一次 skill load，确认 SKILL.md 仍能解析**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
make dev  # 后台跑
# 等服务起来
until curl -sf --max-time 2 http://localhost:2026/ -o /dev/null 2>/dev/null; do sleep 2; done && echo "ready"
# 看 langgraph.log 有没有 skill parser 报错
grep -E "Invalid YAML|skill.*error" packages/agent/logs/langgraph.log | head -5
make stop
```

Expected: skill parser 不报错（如有 "Invalid YAML front-matter" 类报错且涉及 ethoinsight-metric-catalog，回退改动）。

- [ ] **Step 4: Commit skill 文档改动**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md \
        packages/agent/skills/custom/ethoinsight-metric-catalog/references/resolve-cli.md
git commit -m "$(cat <<'EOF'
docs(skill): G5 修复 — catalog.resolve 调用示例删除 --virtual-workspace-dir

随 cli.py 改用 env var fallback（fix commit 上一条），lead 无需再传
--virtual-workspace-dir 参数。skill 主文档示例命令删除该参数 + 加注释说明
sandbox env var 机制；references/resolve-cli.md 参数表更新 fallback 顺序。

参数本身保留以兼容直接命令行调试场景。
EOF
)"
```

---

### Task 4: dogfood 复测 G5

**Files:**
- 不创建 / 不修改任何文件（纯测试 + 观察）

**目的**：用 thread b0d3a611 + 8ff3be6d 同款入口跑一遍，验证 metric_plan.json 的 output 字段终于是虚拟路径。

- [ ] **Step 1: 起服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
make dev  # 后台跑
until curl -sf --max-time 2 http://localhost:2026/ -o /dev/null 2>/dev/null; do sleep 2; done && echo "ready"
```

如果 gateway 30s 起不来，参照 `docs/handoffs/2026-05/2026-05-14-e2e-test-checklist.md` §1.4 手动起备选。

- [ ] **Step 2: 浏览器复测（用 Playwright MCP）**

1. `browser_navigate http://localhost:2026`
2. 登录（如需要）
3. 新建 thread
4. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`
5. 发首条消息：`请分析这个 EPM 单只数据`
6. 跟着流程走到 lead 调用 `python -m ethoinsight.catalog.resolve`（看 langgraph.log SandboxAudit）
7. **关键观察点**：等 catalog.resolve 跑完，立即抓 metric_plan.json

- [ ] **Step 3: 抓 metric_plan.json 验证**

```bash
# 找到最新 thread_id
THREAD_ID=$(ls -t packages/agent/backend/.deer-flow/users/*/threads/ 2>/dev/null | head -1)
echo "Testing thread: $THREAD_ID"

# 看 output 字段是否虚拟路径
cat packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/metric_plan.json \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
all_outputs = [m.get('output','') for m in d.get('metrics', [])] + [c.get('output','') for c in d.get('charts', [])]
print('sample outputs:')
for o in all_outputs[:3]:
    print(' ', o)
all_virtual = all(o.startswith('/mnt/user-data/') for o in all_outputs if o)
print(f'all virtual: {all_virtual}')
assert all_virtual, 'G5 修复未生效，仍有物理路径'
print('G5 FIX VERIFIED ✓')
"
```

Expected: 输出 `G5 FIX VERIFIED ✓` 且 `all virtual: True`。如果 assertion 失败 → 修复无效，看 cli.py 实现是否真的接到 env var。

- [ ] **Step 4: 看 SandboxAudit 验证 env var 注入**

```bash
grep "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE" packages/agent/logs/langgraph.log | head -3
```

Expected: 看到 env var 被注入到 bash 命令（虽然 sandbox audit log 通常不打 env var，主要看是否有相关日志线索）。这一步如果 grep 不到也不算失败——env var 注入是 sandbox 内部行为，可能不打日志。Step 3 的 assertion 才是决定性证据。

- [ ] **Step 5: 停服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent && make stop
```

- [ ] **Step 6: 记录 dogfood 结果到验证 handoff**

用 Edit 工具，在 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 末尾追加：

```markdown
## G5 修复复测（2026-05-14 / YYYY-MM-DD）

修复 commit hash: <填 Task 2 Step 5 的 commit hash>

修复后 thread: <新 dogfood thread_id>

metric_plan.json output 字段抽样：
```
<贴 Step 3 cat 输出的 sample outputs 前 3 条>
```

判定：✅ G5 修复成功（output 字段全部 /mnt/user-data/workspace/ 前缀）/ ❌ 修复无效

Task 7 dogfood-followup-handoff 表格 G5 行原"实测"列从 "False" 改为 "True"，结论 ❌ 改为 ✅。
```

同步把上面 Batch A/B 表格的 G5 那行实测改成 ✅。

- [ ] **Step 7: Commit 复测记录**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md
git commit -m "$(cat <<'EOF'
docs(dogfood): G5 catalog 虚拟路径修复复测通过

新 thread <UUID> 复测：metric_plan.json output 字段全部 /mnt/user-data/workspace/ 前缀，
不再泄漏 host 物理路径。Batch A/B 检查表 G5 行从 ❌ 改为 ✅，10/11 通过（G1/G4 仍待
独立修复，与本 G5 修复无关）。
EOF
)"
```

---

### Task 5: 补 spec §10.3.1 论证缺陷

**Files:**
- Modify: `docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` §10.3.1

- [ ] **Step 1: 改 spec §10.3.1**

用 Edit 工具，在 `docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` 中：

old_string（找到 §10.3.1 的"为何不修"段，精确锁定）：
```
**为何不修**：在 §5.5 `LeadAgentExecutionBoundaryProvider` 落地后，lead 不能 `write_file *.py`，code-executor 又只能 `python -m ethoinsight.scripts.*`（预置模块自己读 `DEERFLOW_PATH_*` env var）——**两条都堵了之后这个 bug 在生产路径上无法触发**。
```

new_string：
```
**为何不修**：在 §5.5 `LeadAgentExecutionBoundaryProvider` 落地后，lead 不能 `write_file *.py`，code-executor 又只能 `python -m ethoinsight.scripts.*`（预置模块自己读 `DEERFLOW_PATH_*` env var）——**两条都堵了之后这个 bug 在生产路径上无法触发**。

**论证缺陷修正（thread 8ff3be6d dogfood 暴露）**：上述判断**不完整**。lead 调 `python -m ethoinsight.catalog.resolve` 等 ethoinsight CLI 仍然合法走 sandbox replace 通道——如果 CLI 接受"虚拟路径作为 metadata 字符串"的参数（例如 `--virtual-workspace-dir /mnt/user-data/workspace` 想让该字符串被写进 plan.json 而非作为文件操作目标），sandbox replace 会把该字符串翻译成物理路径，CLI 收到时语义已丢失。

**已采取的修复模式（G5 修复 plan）**：CLI 改用 env var `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` 反推虚拟路径，绕过被替换的命令行参数值。这是 §10.3.1 修复哲学（"用 env var 旁路 sandbox 替换"）的**早期落地**——本 spec 阶段未来如有同类需求（CLI 接受虚拟路径作 metadata），应复用同一模式：在 CLI 端用 env var key 的存在性判断"是否在 sandbox 模式下"，并用代码硬编码的虚拟路径常量作为 fallback 返回值。

**仍然不动 sandbox 本身**：本论证缺陷的修正不改变"不修 sandbox 路径替换逻辑"的结论——只是说明根因 B 的处置不能只靠"lead 路径堵死"，还需要在所有"虚拟路径作 metadata"的 CLI 接受口加 env var fallback。
```

- [ ] **Step 2: Commit spec 补论证**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md
git commit -m "$(cat <<'EOF'
docs(spec): §10.3.1 补论证缺陷 — sandbox 路径替换在 CLI metadata 场景仍漏

原 spec §10.3.1 论证 "lead 路径堵死后 sandbox 路径不对称问题不可触发"。
thread 8ff3be6d dogfood 证伪——lead 调 ethoinsight.catalog.resolve CLI 时
sandbox 仍然把 --virtual-workspace-dir 参数值替换成物理路径，让 CLI 输出的
plan.json 仍泄漏物理路径。

补充论证：CLI 接受"虚拟路径作 metadata"的参数时，sandbox replace 仍漏。
已采取的修复模式（G5 修复 plan）：CLI 改用 env var DEERFLOW_PATH_* 旁路
sandbox 替换，绕开命令行参数被改写的问题。该模式可复用到未来同类 CLI。

结论：仍不动 sandbox，但 §10.3.1 的"不可触发"判断在 CLI metadata 场景需要
显式启用 env var fallback 才成立。
EOF
)"
```

---

### Task 6: 收尾验证 + push

- [ ] **Step 1: 整体单测全绿（ethoinsight + backend）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/

cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿，除已知预存失败（与本 plan 无关）。

- [ ] **Step 2: ruff lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint

cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
ruff check . 2>&1 | tail -5 || true
```

Expected: 无错误。

- [ ] **Step 3: commit 自查**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline origin/dev..dev
```

Expected 看到本 plan 增加的 5 个 commit（按时间序）：

1. `test(catalog): 加 env-var fallback 失败测试（先红后绿，TDD）`
2. `fix(catalog): G5 修复 — CLI 改用 env var 兜底虚拟路径`
3. `docs(skill): G5 修复 — catalog.resolve 调用示例删除 --virtual-workspace-dir`
4. `docs(dogfood): G5 catalog 虚拟路径修复复测通过`
5. `docs(spec): §10.3.1 补论证缺陷 — sandbox 路径替换在 CLI metadata 场景仍漏`

- [ ] **Step 4: Push**

```bash
cd /home/wangqiuyang/noldus-insight
git push origin dev 2>&1 | tail -5
```

Expected: `<旧hash>..<新hash>  dev -> dev`，5 个 commit 入 origin。

- [ ] **Step 5: 回报用户**

把以下信息汇报给用户：

```
# G5 修复完成

## 5 个 Commit（已 push）
<贴 Step 3 的 commit 列表>

## Dogfood 复测结果
- thread: <UUID>
- metric_plan.json output 字段：✅ 全 /mnt/user-data/workspace/ 前缀
- Batch A/B 表格 G5 行：从 ❌ 改为 ✅
- 总通过率：10/11（G1/G4 仍待独立 plan 修复，与本 G5 无关）

## Spec 更新
- §10.3.1 补论证：sandbox 替换在 CLI metadata 场景仍漏，env var 旁路是修复模式

## 遗留
- G4（阶段播报失效）独立修复
- G1（subagent 透传违规话术）等 spec 阶段 3 落地宪法 read 机制
```

---

## 不要做的事（防止越权）

- ❌ **不要改 sandbox/tools.py** — spec §10.3 已论证不动 sandbox，本 plan 用 env-var 旁路替代
- ❌ **不要改 resolve.py 函数体** — 它本身正确，只动 cli.py 入口
- ❌ **不要删 `--virtual-workspace-dir` 参数** — 保留兼容直接调试场景
- ❌ **不要改 LeadAgentExecutionBoundaryProvider** — 上一 plan 产物，本 plan 不动
- ❌ **不要修 G1 (subagent 违规话术) / G4 (阶段播报失效)** — 独立 plan
- ❌ **不要 force push** — `git push origin dev` 直接推（已和 origin 同步）
- ❌ **不要用 `--no-verify` 跳过 pre-commit hook**
- ❌ **不要碰这 3 个无关文件**：
  - `docs/specs/llm-finetuning-strategy.md`
  - `docs/plans/2026-05-13-base-model-decision-memo.md`
  - `packages/agent/frontend/src/app/page.tsx`

---

## 实施完成后的状态

- 改 2 个 ethoinsight 文件（cli.py 实现 + 1 个新测试文件）
- 改 2 个 skill 文档（SKILL.md 例子 + resolve-cli.md 参数表）
- 改 1 个 spec 段（§10.3.1 补论证）
- 改 1 个 handoff 文档（dogfood 复测记录）
- 5 个 commit 入 origin
- thread 8ff3be6d 同款故障（G5 物理路径泄漏）不可复现
- backend / ethoinsight 单测全绿
- 单测和生产路径都覆盖（前者保护 Python 层契约、后者保护 sandbox 替换层 fallback）

如出现意料外的回归或失败，**停下来**联系用户而不是自己改 scope。
