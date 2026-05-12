# Plan A：脚本即指标地基 + EPM 验证 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地"脚本即指标"架构地基（scripts 包骨架 + 通用脚本 + EPM 范式全量脚本 + 脚本调用白名单 Guard + code-executor prompt 改造 + EPM skill md 重写），并通过合成数据 e2e 测试。EPM 跑通后作为剩余 6 范式（Plan B）的范本。

**Architecture:** 每个指标 / 图表 / 统计 = 独立可执行脚本（`python -m ethoinsight.scripts.<paradigm>.<name>`）。code-executor 不写代码，只看 `by-paradigm/<paradigm>.md`「决策手册」选脚本编排。`ScriptInvocationOnlyProvider` 白名单拦 bash 命令，强制只能调脚本或做文件操作。详见 [docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md](../specs/2026-05-12-script-per-metric-architecture-design.md)。

**Tech Stack:** Python 3.10+（ethoinsight 库），Python 3.12+（agent backend），pytest（脚本单测 + e2e），deerflow `GuardrailProvider` Protocol（agent backend 已有基建）。

---

## 文件结构地图

**新建文件：**

| 路径 | 责任 |
|---|---|
| `packages/ethoinsight/ethoinsight/scripts/__init__.py` | 标识 scripts 为 Python 子包 |
| `packages/ethoinsight/ethoinsight/scripts/_cli.py` | 统一 CLI helper：`parse_args()`、`read_inputs_json()`、`read_groups_json()`、`emit_result()`、`save_output()` 等 |
| `packages/ethoinsight/ethoinsight/scripts/_common/__init__.py` | 跨范式通用脚本模块 |
| `packages/ethoinsight/ethoinsight/scripts/_common/compute_distance_moved.py` | 距离移动总和 |
| `packages/ethoinsight/ethoinsight/scripts/_common/compute_velocity_stats.py` | 速度统计描述（mean/std/max/min/median） |
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py` | 轨迹图 |
| `packages/ethoinsight/ethoinsight/scripts/epm/__init__.py` | EPM 脚本子包标识 |
| `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time_ratio.py` | 开臂时间占比 |
| `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_count.py` | 开臂进入次数 |
| `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_ratio.py` | 开臂进入比 |
| `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time.py` | 开臂时间（秒） |
| `packages/ethoinsight/ethoinsight/scripts/epm/compute_total_entry_count.py` | 总进臂次数 |
| `packages/ethoinsight/ethoinsight/scripts/epm/plot_box_open_arm.py` | 开臂时间组间箱线图 |
| `packages/ethoinsight/ethoinsight/scripts/epm/run_groupwise_stats.py` | EPM 全指标分组统计检验 |
| `packages/ethoinsight/tests/scripts/__init__.py` | 测试包标识 |
| `packages/ethoinsight/tests/scripts/conftest.py` | pytest fixture（合成 EPM DataFrame，复用 `test_metrics_epm.py:_make_epm_df` 思路）+ 临时 EthoVision 文件 fixture |
| `packages/ethoinsight/tests/scripts/test_common_scripts.py` | `_common/*` 脚本子进程测试 |
| `packages/ethoinsight/tests/scripts/test_epm_scripts.py` | `epm/*` 脚本子进程测试 |
| `packages/ethoinsight/tests/scripts/test_epm_e2e.py` | EPM 多脚本编排 e2e 测试 |
| `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py` | 白名单 Guardrail |
| `packages/agent/backend/tests/test_script_invocation_only_provider.py` | Guardrail 单测 |

**修改文件：**

| 路径 | 范围 |
|---|---|
| `packages/agent/skills/custom/ethoinsight-code/SKILL.md` | 顶部 workflow 描述：从"按 md 写胶水脚本"改为"按 md 选脚本编排" |
| `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md` | 全文重写为「脚本清单 + 决策手册」 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | `system_prompt` workflow 段重写为 3 步硬路径（read md → 选脚本 → bash 调脚本） |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | `_create_agent()` 中追加 `GuardrailMiddleware(ScriptInvocationOnlyProvider())` |

**不动文件**（spec §6.3）：

- `ethoinsight/metrics/*.py`、`charts.py`、`statistics.py`、`parse.py` —— 现有库代码，脚本只是包装层
- `code-executor` 工具集（bash + read_file + write_file + ls + str_replace）
- 其他 5 个 subagent 的 prompt
- lead-agent prompt

---

## 任务列表

### Task 1：CLI Helper 模块 + 包骨架

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/__init__.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/_cli.py`
- Test: `packages/ethoinsight/tests/scripts/__init__.py`（空）
- Test: `packages/ethoinsight/tests/scripts/test_cli_helper.py`

- [ ] **Step 1: 创建 scripts 包根 __init__.py**

文件 `packages/ethoinsight/ethoinsight/scripts/__init__.py`：

```python
"""脚本即指标：每个指标 / 图表 / 统计 = 独立可执行脚本。

调用方式: ``python -m ethoinsight.scripts.<paradigm>.<script_name> ...``

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md
"""
```

- [ ] **Step 2: 创建测试包标识**

文件 `packages/ethoinsight/tests/scripts/__init__.py`：

```python
```（空文件）

- [ ] **Step 3: 写 CLI helper 的失败测试**

文件 `packages/ethoinsight/tests/scripts/test_cli_helper.py`：

```python
"""Tests for ethoinsight.scripts._cli helper module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEmitResult:
    def test_emit_result_prints_marker_with_payload(self, capsys):
        from ethoinsight.scripts._cli import emit_result

        emit_result({"metric": "open_arm_time_ratio", "value": 0.35})

        captured = capsys.readouterr()
        assert "[result]" in captured.out
        # The JSON payload must follow the marker on the same line
        line = next(l for l in captured.out.splitlines() if l.startswith("[result]"))
        payload = json.loads(line[len("[result] "):])
        assert payload == {"metric": "open_arm_time_ratio", "value": 0.35}


class TestSaveOutputJson:
    def test_save_output_writes_json_atomically(self, tmp_path: Path):
        from ethoinsight.scripts._cli import save_output_json

        out_path = tmp_path / "out.json"
        save_output_json(out_path, {"value": 0.5})

        assert out_path.exists()
        assert json.loads(out_path.read_text()) == {"value": 0.5}

    def test_save_output_creates_parent_dirs(self, tmp_path: Path):
        from ethoinsight.scripts._cli import save_output_json

        out_path = tmp_path / "deep/nested/out.json"
        save_output_json(out_path, {"value": 1})

        assert out_path.exists()


class TestReadInputsJson:
    def test_read_inputs_json_returns_list_of_paths(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_inputs_json

        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps(["/tmp/a.txt", "/tmp/b.txt"]))

        result = read_inputs_json(inputs_file)
        assert result == ["/tmp/a.txt", "/tmp/b.txt"]

    def test_read_inputs_json_rejects_non_array(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_inputs_json

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "array"}))

        with pytest.raises(ValueError, match="must be a JSON array"):
            read_inputs_json(bad)


class TestReadGroupsJson:
    def test_read_groups_json_returns_dict(self, tmp_path: Path):
        from ethoinsight.scripts._cli import read_groups_json

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps({"control": ["s1"], "treatment": ["s2"]}))

        result = read_groups_json(groups_file)
        assert result == {"control": ["s1"], "treatment": ["s2"]}
```

- [ ] **Step 4: 运行测试验证失败**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_cli_helper.py -v`
Expected: 失败，原因 `ModuleNotFoundError: ethoinsight.scripts._cli`

- [ ] **Step 5: 实现 _cli.py**

文件 `packages/ethoinsight/ethoinsight/scripts/_cli.py`：

```python
"""统一的脚本 CLI helper。

所有 ethoinsight.scripts.* 下的脚本通过本模块统一参数解析、I/O 接口。

接口约定见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §3.2
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ============================================================================
# Stdout marker - subagent uses this to extract per-script result
# ============================================================================

RESULT_MARKER = "[result]"


def emit_result(payload: dict[str, Any]) -> None:
    """Print a `[result] {json}` line to stdout for subagent extraction."""
    print(f"{RESULT_MARKER} {json.dumps(payload, ensure_ascii=False)}")


# ============================================================================
# File I/O
# ============================================================================


def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically (temp file + rename), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def read_inputs_json(path: str | Path) -> list[str]:
    """Read a JSON file containing a list of input file paths.

    Format: ``["/path/to/subject1.txt", "/path/to/subject2.txt", ...]``
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of file paths, got {type(data).__name__}")
    return [str(item) for item in data]


def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    """Read a JSON file containing the groups mapping.

    Format: ``{"group_name": ["subject_name_1", "subject_name_2"], ...}``
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object mapping group names to subject lists")
    return {k: list(v) for k, v in data.items()}


# ============================================================================
# Standard argparse builders for the three script types
# ============================================================================


def make_compute_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``compute_*.py``: --input <single trajectory> --output <metric.json>."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--input", required=True, help="Path to a single EthoVision trajectory file")
    ap.add_argument("--output", required=True, help="Path to write the metric JSON")
    return ap


def make_plot_parser(description: str, *, supports_groups: bool = False) -> argparse.ArgumentParser:
    """Argparse for ``plot_*.py``.

    Single-file plots use ``--input``; aggregated plots use ``--inputs`` + optional ``--groups``.
    """
    ap = argparse.ArgumentParser(description=description)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single trajectory file (single-file plots)")
    group.add_argument("--inputs", help="Path to a JSON file containing a list of trajectory file paths")
    if supports_groups:
        ap.add_argument("--groups", help="Path to a JSON file mapping group_name -> [subject_name, ...]")
    ap.add_argument("--output", required=True, help="Path to write the PNG plot")
    return ap


def make_stats_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``run_*_stats.py``: --inputs --groups --output."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--inputs", required=True, help="Path to a JSON file containing a list of trajectory file paths")
    ap.add_argument("--groups", required=True, help="Path to a JSON file mapping group_name -> [subject_name, ...]")
    ap.add_argument("--output", required=True, help="Path to write the stats JSON")
    return ap
```

- [ ] **Step 6: 运行测试验证通过**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_cli_helper.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 验证全量已有测试未被破坏**

Run: `cd packages/ethoinsight && python -m pytest tests/ -x --ignore=tests/scripts -q`
Expected: 现有 170+ 测试全绿（spec §6.3 保证不动现有库代码）

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/__init__.py \
        packages/ethoinsight/ethoinsight/scripts/_cli.py \
        packages/ethoinsight/tests/scripts/__init__.py \
        packages/ethoinsight/tests/scripts/test_cli_helper.py
git commit -m "feat(scripts): CLI helper + scripts 包骨架（Plan A T1）

统一脚本即指标的参数解析与 I/O 接口：
- emit_result() 输出 [result] {json} 行供 subagent 抓取
- save_output_json() 原子写
- read_inputs_json() / read_groups_json() JSON 文件读取
- make_{compute,plot,stats}_parser() 三类标准 argparse builder

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §3.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2：测试 conftest（合成数据 + 临时 EthoVision 文件 fixture）

**Files:**
- Create: `packages/ethoinsight/tests/scripts/conftest.py`

- [ ] **Step 1: 写 conftest（合成 EthoVision 文件 fixture）**

文件 `packages/ethoinsight/tests/scripts/conftest.py`：

```python
"""Shared pytest fixtures for scripts subprocess tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Synthetic EPM DataFrame builder (lifted from tests/test_metrics_epm.py)
# ============================================================================


def _make_epm_df(
    n_frames: int = 100,
    *,
    open_arm_cols: list[str] | None = None,
    closed_arm_cols: list[str] | None = None,
    center_cols: list[str] | None = None,
    open_arm_pattern: list[int] | None = None,
    closed_arm_pattern: list[int] | None = None,
    center_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if open_arm_cols is None:
        open_arm_cols = ["in_zone_open_arm_1"]
    if closed_arm_cols is None:
        closed_arm_cols = ["in_zone_closed_arm_1"]
    if center_cols is None:
        center_cols = ["in_zone_center-point"]

    df = pd.DataFrame({
        "trial_time": np.arange(n_frames, dtype=float) * 0.04,
        "x_center": rng.uniform(100, 500, n_frames),
        "y_center": rng.uniform(100, 500, n_frames),
        "distance_moved": rng.uniform(0, 5, n_frames),
        "velocity": rng.uniform(0, 20, n_frames),
    })

    if open_arm_pattern is None:
        # Alternating: 20 frames in, 20 out, repeat
        pat = ([1] * 20 + [0] * 20) * (n_frames // 40 + 1)
        open_arm_pattern = pat[:n_frames]
    if closed_arm_pattern is None:
        # Inverse of open arm by default
        closed_arm_pattern = [1 - v for v in open_arm_pattern]
    if center_pattern is None:
        center_pattern = [0] * n_frames

    for col in open_arm_cols:
        df[col] = open_arm_pattern
    for col in closed_arm_cols:
        df[col] = closed_arm_pattern
    for col in center_cols:
        df[col] = center_pattern

    return df


# ============================================================================
# EthoVision trajectory file fixture
# ============================================================================


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    """Write `df` as a minimal EthoVision-style trajectory file (UTF-16-LE BOM, semicolon-delimited).

    Format matches what ``ethoinsight.parse.parse_trajectory()`` expects:
    - First line: ``"标题行数";"<N>"`` where N = number of header lines
    - Header lines include "对象名称" pointing to `subject`
    - Then a unit row, then the data rows
    """
    columns = list(df.columns)
    n_header_lines = 5  # title-count + 3 metadata + column-names = 5 (units = 6th, then data)

    lines: list[str] = []
    # Line 1: header count
    lines.append(f'"标题行数";"{n_header_lines}"')
    # Line 2-3: metadata (placeholder)
    lines.append(f'"对象名称";"{subject}"')
    lines.append('"试验名称";"Trial 1"')
    lines.append('"竞技场名称";"Arena 1"')
    # Line 5: column names
    lines.append(";".join(f'"{c}"' for c in columns))
    # Line 6: units (placeholder)
    lines.append(";".join(['""'] * len(columns)))
    # Line 7+: data rows
    for _, row in df.iterrows():
        values = []
        for v in row.values:
            if pd.isna(v):
                values.append('"-"')
            else:
                values.append(f'"{v}"')
        lines.append(";".join(values))

    content = "\n".join(lines) + "\n"
    # Prepend BOM and write as UTF-16-LE
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")  # UTF-16-LE BOM
        f.write(content.encode("utf-16-le"))


@pytest.fixture
def epm_trajectory_file(tmp_path: Path) -> Path:
    """Write a synthetic EPM trajectory file with mixed open/closed arm occupancy."""
    df = _make_epm_df(n_frames=200)
    path = tmp_path / "epm_subject_1.txt"
    _df_to_ethovision_file(df, path, subject="Subject 1")
    return path


@pytest.fixture
def epm_trajectory_files(tmp_path: Path) -> list[Path]:
    """Write 6 synthetic EPM trajectory files (3 control, 3 treatment) with diverging patterns."""
    files = []
    for i in range(1, 7):
        # control: more open arm time; treatment: less
        if i <= 3:
            pattern = ([1] * 30 + [0] * 10) * 10  # 75% open arm
        else:
            pattern = ([1] * 5 + [0] * 35) * 10   # 12.5% open arm
        df = _make_epm_df(n_frames=400, open_arm_pattern=pattern[:400])
        path = tmp_path / f"epm_subject_{i}.txt"
        _df_to_ethovision_file(df, path, subject=f"Subject {i}")
        files.append(path)
    return files


@pytest.fixture
def make_epm_df():
    """Expose `_make_epm_df` directly for tests that don't need a file."""
    return _make_epm_df
```

- [ ] **Step 2: 烟测 fixture（验证 parse 真能读这种合成文件）**

文件 `packages/ethoinsight/tests/scripts/test_conftest_smoke.py`（**烟测专用，验证 fixture 自身正确**）：

```python
"""Smoke test for conftest fixtures: verifies synthetic EthoVision files parse correctly."""

from __future__ import annotations

from pathlib import Path

from ethoinsight.parse import parse_trajectory


def test_epm_trajectory_file_parses(epm_trajectory_file: Path):
    df = parse_trajectory(str(epm_trajectory_file))
    assert "in_zone_open_arm_1" in df.columns
    assert "trial_time" in df.columns
    assert len(df) > 0


def test_epm_trajectory_files_have_expected_count(epm_trajectory_files: list[Path]):
    assert len(epm_trajectory_files) == 6
    for p in epm_trajectory_files:
        df = parse_trajectory(str(p))
        assert len(df) > 0
```

- [ ] **Step 3: 跑烟测**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_conftest_smoke.py -v`
Expected: PASS。如果 parse 解析失败，说明 `_df_to_ethovision_file` 格式与 `parse_trajectory` 不匹配，需对照 `parse.py:140-200` 调整 header 格式。

- [ ] **Step 4: Commit**

```bash
git add packages/ethoinsight/tests/scripts/conftest.py \
        packages/ethoinsight/tests/scripts/test_conftest_smoke.py
git commit -m "test(scripts): conftest fixture + 合成 EthoVision 文件烟测（Plan A T2）

复用 test_metrics_epm.py 的 _make_epm_df 思路，加 _df_to_ethovision_file
helper 把 DataFrame 序列化成 UTF-16-LE BOM 的 EthoVision 格式，供脚本
subprocess 测试用。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3：第一个 compute 脚本（compute_open_arm_time_ratio）—— 范式确立

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/__init__.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time_ratio.py`
- Test: `packages/ethoinsight/tests/scripts/test_epm_scripts.py`（新建，本任务只加 1 个测试类）

> **本任务非常重要 ——** 它确立了"script template"。后续所有 compute_*.py 脚本都照这个 template 写。Reviewer 重点 review 这个脚本的接口、I/O 规范、错误处理是否符合 spec §3.2。

- [ ] **Step 1: 创建 epm 子包标识**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/__init__.py`：

```python
"""EPM (Elevated Plus Maze) 范式脚本。"""
```

- [ ] **Step 2: 写脚本的失败测试**

文件 `packages/ethoinsight/tests/scripts/test_epm_scripts.py`：

```python
"""Subprocess-level tests for ethoinsight.scripts.epm.*."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run `python -m <module> <args>` and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestComputeOpenArmTimeRatio:
    def test_happy_path_writes_json_and_emits_result(
        self, epm_trajectory_file: Path, tmp_path: Path
    ):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert isinstance(payload["value"], float)
        assert 0.0 <= payload["value"] <= 1.0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] "):])
        assert result_payload["metric"] == "open_arm_time_ratio"

    def test_missing_input_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_file_without_open_arm_columns_returns_value_none(
        self, tmp_path: Path, make_epm_df
    ):
        from tests.scripts.conftest import _df_to_ethovision_file

        # Build a df without any open-arm zone columns
        df = make_epm_df(open_arm_cols=["in_zone_closed_arm_1"])  # only closed arm
        df = df.drop(columns=["in_zone_closed_arm_1"])  # remove all zone cols
        df["x_center"] = df["x_center"]  # keep position cols
        path = tmp_path / "no_open_arm.txt"
        _df_to_ethovision_file(df, path)

        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(path), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert payload["value"] is None
```

- [ ] **Step 3: 跑测试验证失败**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py::TestComputeOpenArmTimeRatio -v`
Expected: 失败，`ModuleNotFoundError: ethoinsight.scripts.epm.compute_open_arm_time_ratio`

- [ ] **Step 4: 实现脚本**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time_ratio.py`：

```python
"""EPM: 开臂时间占比 (open arm time ratio)。

CLI: python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "open_arm_time_ratio", "value": <float or null>}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.epm import compute_open_arm_time_ratio
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    save_output_json,
)


METRIC_NAME = "open_arm_time_ratio"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    value = compute_open_arm_time_ratio(df)

    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 跑测试验证通过**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py::TestComputeOpenArmTimeRatio -v`
Expected: 全部 PASS（3 个测试）

- [ ] **Step 6: 手动 e2e 烟测**

Run:
```bash
cd packages/ethoinsight
python -m pytest tests/scripts/test_conftest_smoke.py -v  # 确认 fixture 仍 OK
# 然后用 Python REPL 跑一次：
python -c "
import tempfile, json
from pathlib import Path
import subprocess, sys
# Reuse fixture machinery
from tests.scripts.conftest import _make_epm_df, _df_to_ethovision_file
tmp = Path(tempfile.mkdtemp())
df = _make_epm_df(n_frames=200)
input_path = tmp / 'test.txt'
_df_to_ethovision_file(df, input_path)
output_path = tmp / 'out.json'
r = subprocess.run([sys.executable, '-m', 'ethoinsight.scripts.epm.compute_open_arm_time_ratio',
                    '--input', str(input_path), '--output', str(output_path)],
                   capture_output=True, text=True)
print('stdout:', r.stdout)
print('stderr:', r.stderr)
print('result file:', json.loads(output_path.read_text()))
"
```

Expected: stdout 含 `[result] {"metric": "open_arm_time_ratio", "value": 0.5}` 类似输出，result file 含同样内容。

- [ ] **Step 7: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/epm/__init__.py \
        packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time_ratio.py \
        packages/ethoinsight/tests/scripts/test_epm_scripts.py
git commit -m "feat(scripts): EPM compute_open_arm_time_ratio 脚本（Plan A T3）

第一个 compute_*.py 脚本，确立 script template：
- argparse 通过 _cli.make_compute_parser
- parse → 调 metrics.epm.* → save_output_json → emit_result
- 子进程测试覆盖 happy path + 缺参 + 无相关列三种场景

后续所有 compute_*.py 照此模板写。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4：EPM 剩余 4 个 compute 脚本

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_count.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_ratio.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/compute_total_entry_count.py`
- Test: `packages/ethoinsight/tests/scripts/test_epm_scripts.py`（追加测试类）

> 4 个脚本同结构同模板，4 次重复 TDD 循环。

- [ ] **Step 1: 写 4 个脚本的失败测试（追加到 test_epm_scripts.py）**

在 `test_epm_scripts.py` 末尾追加：

```python
class TestComputeOpenArmEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestComputeOpenArmEntryRatio:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_ratio"
        # ratio is None or float in [0, 1]
        assert payload["value"] is None or (
            isinstance(payload["value"], float) and 0.0 <= payload["value"] <= 1.0
        )


class TestComputeOpenArmTime:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeTotalEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_total_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "total_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py -v`
Expected: 4 个新测试失败（前 1 个 ratio 测试已 PASS）。

- [ ] **Step 3: 实现 compute_open_arm_entry_count.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_count.py`：

```python
"""EPM: 开臂进入次数 (open arm entry count)。

CLI: python -m ethoinsight.scripts.epm.compute_open_arm_entry_count \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "open_arm_entry_count", "value": <int or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.epm import compute_open_arm_entry_count
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "open_arm_entry_count"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_open_arm_entry_count(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 实现 compute_open_arm_entry_ratio.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_entry_ratio.py`：

```python
"""EPM: 开臂进入次数占总进臂次数比例 (open arm entry ratio)。

CLI: python -m ethoinsight.scripts.epm.compute_open_arm_entry_ratio \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "open_arm_entry_ratio", "value": <float or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.epm import compute_open_arm_entry_ratio
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "open_arm_entry_ratio"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_open_arm_entry_ratio(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 实现 compute_open_arm_time.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/compute_open_arm_time.py`：

```python
"""EPM: 开臂总停留时间（秒） (open arm time)。

CLI: python -m ethoinsight.scripts.epm.compute_open_arm_time \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "open_arm_time", "value": <float or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.epm import compute_open_arm_time
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "open_arm_time"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_open_arm_time(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 实现 compute_total_entry_count.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/compute_total_entry_count.py`：

```python
"""EPM: 总进臂次数（开臂 + 闭臂） (total entry count)。

CLI: python -m ethoinsight.scripts.epm.compute_total_entry_count \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "total_entry_count", "value": <int or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.epm import compute_total_entry_count
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "total_entry_count"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_total_entry_count(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: 跑测试验证全部通过**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py -v`
Expected: 7 个测试全 PASS（原 3 个 + 新 4 个）

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/epm/compute_*.py \
        packages/ethoinsight/tests/scripts/test_epm_scripts.py
git commit -m "feat(scripts): EPM 剩余 4 个 compute 脚本（Plan A T4）

照 T3 template 添加：
- compute_open_arm_entry_count
- compute_open_arm_entry_ratio
- compute_open_arm_time
- compute_total_entry_count

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5：通用脚本 _common（distance / velocity / trajectory）

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/_common/__init__.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/_common/compute_distance_moved.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/_common/compute_velocity_stats.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py`
- Test: `packages/ethoinsight/tests/scripts/test_common_scripts.py`

- [ ] **Step 1: 创建 _common 子包标识**

文件 `packages/ethoinsight/ethoinsight/scripts/_common/__init__.py`：

```python
"""跨范式通用脚本（distance / velocity / trajectory 等任范式可用的指标）。"""
```

- [ ] **Step 2: 写 3 个脚本的失败测试**

文件 `packages/ethoinsight/tests/scripts/test_common_scripts.py`：

```python
"""Tests for ethoinsight.scripts._common.*."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, check=False,
    )


class TestComputeDistanceMoved:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts._common.compute_distance_moved",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "distance_moved"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeVelocityStats:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts._common.compute_velocity_stats",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "velocity_stats"
        # value is a dict with mean/std/max/min/median
        v = payload["value"]
        assert v is None or set(v.keys()) >= {"mean", "std", "max", "min", "median"}


class TestPlotTrajectory:
    def test_single_input_produces_png(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "trajectory.png"
        result = _run_script(
            "ethoinsight.scripts._common.plot_trajectory",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # PNG 不应为空

    def test_multiple_inputs_produces_png(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))
        out_path = tmp_path / "trajectory_all.png"
        result = _run_script(
            "ethoinsight.scripts._common.plot_trajectory",
            ["--inputs", str(inputs_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
```

- [ ] **Step 3: 跑测试验证失败**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_common_scripts.py -v`
Expected: 全部失败（脚本未实现）。

- [ ] **Step 4: 实现 compute_distance_moved.py**

文件 `packages/ethoinsight/ethoinsight/scripts/_common/compute_distance_moved.py`：

```python
"""通用：总移动距离 (distance moved, sum of per-frame distance)。

CLI: python -m ethoinsight.scripts._common.compute_distance_moved \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "distance_moved", "value": <float or null>}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_distance_moved
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "distance_moved"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_distance_moved(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 实现 compute_velocity_stats.py**

文件 `packages/ethoinsight/ethoinsight/scripts/_common/compute_velocity_stats.py`：

```python
"""通用：速度统计描述 (mean / std / max / min / median)。

CLI: python -m ethoinsight.scripts._common.compute_velocity_stats \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "velocity_stats", "value": {"mean", "std", "max", "min", "median"} or null}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics._common import compute_velocity_stats
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_compute_parser, save_output_json


METRIC_NAME = "velocity_stats"


def main(argv: list[str] | None = None) -> int:
    args = make_compute_parser(description=__doc__).parse_args(argv)
    df = parse_trajectory(args.input)
    value = compute_velocity_stats(df)
    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 实现 plot_trajectory.py**

文件 `packages/ethoinsight/ethoinsight/scripts/_common/plot_trajectory.py`：

```python
"""通用：轨迹图 (X/Y plot of position over time)。

CLI:
  单文件:  python -m ethoinsight.scripts._common.plot_trajectory \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts._common.plot_trajectory \
             --inputs <inputs.json> --output <png>

输出: PNG 图像文件。

inputs.json 格式: ["/path/to/file1.txt", "/path/to/file2.txt", ...]
"""

from __future__ import annotations

import sys

import pandas as pd

from ethoinsight.charts import trajectory_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)

    if args.input:
        df = parse_trajectory(args.input)
    else:
        # Multi-file aggregated trajectory plot
        paths = read_inputs_json(args.inputs)
        dfs = []
        for p in paths:
            sub_df = parse_trajectory(p)
            subject_attr = sub_df.attrs.get("subject", p)
            sub_df = sub_df.assign(subject=subject_attr)
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)

    output_path = trajectory_plot(df, color_by="subject", output_path=args.output)
    emit_result({"plot": "trajectory", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: 跑测试验证通过**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_common_scripts.py -v`
Expected: 4 个测试全 PASS

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/_common/ \
        packages/ethoinsight/tests/scripts/test_common_scripts.py
git commit -m "feat(scripts): _common 通用脚本（distance/velocity/trajectory）（Plan A T5）

跨范式通用脚本，任何范式可调：
- compute_distance_moved
- compute_velocity_stats
- plot_trajectory（支持 --input 单文件 或 --inputs JSON 多文件）

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6：EPM plot + stats 脚本

**Files:**
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/plot_box_open_arm.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/epm/run_groupwise_stats.py`
- Test: `packages/ethoinsight/tests/scripts/test_epm_scripts.py`（追加）

- [ ] **Step 1: 写 plot + stats 失败测试（追加到 test_epm_scripts.py）**

在 `test_epm_scripts.py` 末尾追加：

```python
class TestPlotBoxOpenArm:
    def test_plot_with_groups(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

        # First 3 = control, last 3 = treatment
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps({
            "control": ["Subject 1", "Subject 2", "Subject 3"],
            "treatment": ["Subject 4", "Subject 5", "Subject 6"],
        }))

        out_path = tmp_path / "box.png"
        result = _run_script(
            "ethoinsight.scripts.epm.plot_box_open_arm",
            ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000


class TestRunGroupwiseStats:
    def test_stats_with_two_groups(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps({
            "control": ["Subject 1", "Subject 2", "Subject 3"],
            "treatment": ["Subject 4", "Subject 5", "Subject 6"],
        }))

        out_path = tmp_path / "stats.json"
        result = _run_script(
            "ethoinsight.scripts.epm.run_groupwise_stats",
            ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert "comparisons" in payload
        assert "alpha" in payload
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py::TestPlotBoxOpenArm tests/scripts/test_epm_scripts.py::TestRunGroupwiseStats -v`
Expected: 失败（脚本未实现）。

- [ ] **Step 3: 实现 plot_box_open_arm.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/plot_box_open_arm.py`：

```python
"""EPM: 开臂时间组间对比箱线图。

CLI: python -m ethoinsight.scripts.epm.plot_box_open_arm \
       --inputs <inputs.json> --groups <groups.json> --output <png>

inputs.json: ["/path/to/file1.txt", ...]
groups.json: {"control": ["Subject 1", ...], "treatment": ["Subject 4", ...]}

输出: PNG 图像。
"""

from __future__ import annotations

import sys

from ethoinsight.charts import box_plot
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    make_plot_parser,
    read_groups_json,
    read_inputs_json,
)


METRICS_TO_PLOT = ["open_arm_time_ratio", "open_arm_entry_count"]


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)
    if not args.inputs:
        print("error: plot_box_open_arm requires --inputs (multi-file)", file=sys.stderr)
        return 2

    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups) if args.groups else None

    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(parsed, paradigm="epm", groups=groups)

    output_path = box_plot(metrics, metrics_to_plot=METRICS_TO_PLOT, output_path=args.output)
    emit_result({"plot": "box_open_arm", "path": output_path, "metrics": METRICS_TO_PLOT})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 实现 run_groupwise_stats.py**

文件 `packages/ethoinsight/ethoinsight/scripts/epm/run_groupwise_stats.py`：

```python
"""EPM: 分组统计检验（Shapiro-Wilk 决策树自动选 t-test / Mann-Whitney）。

CLI: python -m ethoinsight.scripts.epm.run_groupwise_stats \
       --inputs <inputs.json> --groups <groups.json> --output <stats.json>

输出 JSON:
  {"paradigm": "epm",
   "comparisons": {metric: [{"group1", "group2", "p_value", ...}]},
   "summary": str, "alpha": float, "correction": str}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    make_stats_parser,
    read_groups_json,
    read_inputs_json,
    save_output_json,
)
from ethoinsight.statistics import compare_groups


METRICS_TO_TEST = [
    "open_arm_time_ratio",
    "open_arm_entry_count",
    "open_arm_entry_ratio",
    "open_arm_time",
    "total_entry_count",
]


def main(argv: list[str] | None = None) -> int:
    args = make_stats_parser(description=__doc__).parse_args(argv)
    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups)

    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(parsed, paradigm="epm", groups=groups)
    stats = compare_groups(metrics, metrics_to_test=METRICS_TO_TEST)

    payload = {"paradigm": "epm", **stats}
    save_output_json(args.output, payload)
    emit_result({"stats": "epm_groupwise", "n_metrics": len(METRICS_TO_TEST),
                 "summary": stats.get("summary", "")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: 跑测试验证通过**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_scripts.py -v`
Expected: 9 个测试全 PASS

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/scripts/epm/plot_box_open_arm.py \
        packages/ethoinsight/ethoinsight/scripts/epm/run_groupwise_stats.py \
        packages/ethoinsight/tests/scripts/test_epm_scripts.py
git commit -m "feat(scripts): EPM plot + stats 脚本（Plan A T6）

- plot_box_open_arm: 开臂时间组间对比箱线图
- run_groupwise_stats: 5 指标全 Shapiro-Wilk 决策树统计检验

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7：EPM e2e 编排测试（模拟 subagent 调度多脚本）

**Files:**
- Test: `packages/ethoinsight/tests/scripts/test_epm_e2e.py`

> 模拟 code-executor subagent 真实工作流：读 6 个 subject 文件 → 跑核心指标脚本 → 跑组间箱线图 → 跑统计 → 汇总 handoff JSON。

- [ ] **Step 1: 写 e2e 测试**

文件 `packages/ethoinsight/tests/scripts/test_epm_e2e.py`：

```python
"""End-to-end test: simulate code-executor orchestrating multiple EPM scripts.

This mirrors what the subagent will actually do in production:
1. Receive a list of trajectory files + group assignment from lead
2. Run each requested compute_*.py script per subject
3. Run plot_box_open_arm with groups
4. Run run_groupwise_stats with groups
5. Aggregate everything into handoff_code_executor.json

The test does NOT exercise the agent itself — it exercises the script
interface contract that the agent's prompt will rely on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, check=False,
    )


def test_epm_full_orchestration(epm_trajectory_files: list[Path], tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outputs = workspace / "outputs"
    outputs.mkdir()

    # Subagent would write these JSON files before calling scripts
    inputs_file = workspace / "inputs.json"
    inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

    groups_file = workspace / "groups.json"
    groups_file.write_text(json.dumps({
        "control": ["Subject 1", "Subject 2", "Subject 3"],
        "treatment": ["Subject 4", "Subject 5", "Subject 6"],
    }))

    # ----- Step 1: per-subject compute scripts (simulate subagent loop) -----
    per_subject: dict[str, dict] = {}
    for traj_file in epm_trajectory_files:
        subject_results: dict[str, object] = {}
        for metric_module in [
            "compute_open_arm_time_ratio",
            "compute_open_arm_entry_count",
            "compute_open_arm_entry_ratio",
            "compute_open_arm_time",
            "compute_total_entry_count",
        ]:
            out_path = outputs / f"{traj_file.stem}__{metric_module}.json"
            result = _run(
                f"ethoinsight.scripts.epm.{metric_module}",
                ["--input", str(traj_file), "--output", str(out_path)],
            )
            assert result.returncode == 0, f"{metric_module} failed: {result.stderr}"
            payload = json.loads(out_path.read_text())
            subject_results[payload["metric"]] = payload["value"]
        per_subject[traj_file.stem] = subject_results

    # ----- Step 2: group-level box plot -----
    box_path = outputs / "epm_box.png"
    result = _run(
        "ethoinsight.scripts.epm.plot_box_open_arm",
        ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(box_path)],
    )
    assert result.returncode == 0, f"box plot failed: {result.stderr}"
    assert box_path.exists()

    # ----- Step 3: groupwise stats -----
    stats_path = outputs / "epm_stats.json"
    result = _run(
        "ethoinsight.scripts.epm.run_groupwise_stats",
        ["--inputs", str(inputs_file), "--groups", str(groups_file), "--output", str(stats_path)],
    )
    assert result.returncode == 0, f"stats failed: {result.stderr}"
    stats = json.loads(stats_path.read_text())

    # ----- Step 4: aggregate into handoff JSON (subagent does this) -----
    handoff = {
        "paradigm": "epm",
        "per_subject": per_subject,
        "charts": [str(box_path)],
        "statistics": stats,
    }
    handoff_path = workspace / "handoff_code_executor.json"
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2))

    # ----- Assertions: handoff is well-formed -----
    assert handoff["paradigm"] == "epm"
    assert len(handoff["per_subject"]) == 6
    # Every subject has all 5 metrics
    for subject, metrics in handoff["per_subject"].items():
        assert set(metrics.keys()) == {
            "open_arm_time_ratio",
            "open_arm_entry_count",
            "open_arm_entry_ratio",
            "open_arm_time",
            "total_entry_count",
        }
    assert "comparisons" in handoff["statistics"]


def test_epm_single_subject_descriptive(epm_trajectory_file: Path, tmp_path: Path):
    """n=1 single-subject scenario: skip stats / group plots, only run compute_* + plot_trajectory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outputs = workspace / "outputs"
    outputs.mkdir()

    # Compute all 5 EPM metrics for the single subject
    metrics_result: dict[str, object] = {}
    for metric_module in [
        "compute_open_arm_time_ratio",
        "compute_open_arm_entry_count",
        "compute_open_arm_entry_ratio",
        "compute_open_arm_time",
        "compute_total_entry_count",
    ]:
        out_path = outputs / f"{metric_module}.json"
        result = _run(
            f"ethoinsight.scripts.epm.{metric_module}",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"{metric_module}: {result.stderr}"
        payload = json.loads(out_path.read_text())
        metrics_result[payload["metric"]] = payload["value"]

    # Trajectory plot
    traj_path = outputs / "trajectory.png"
    result = _run(
        "ethoinsight.scripts._common.plot_trajectory",
        ["--input", str(epm_trajectory_file), "--output", str(traj_path)],
    )
    assert result.returncode == 0, f"trajectory: {result.stderr}"
    assert traj_path.exists()

    # No stats, no group plots → handoff has no `statistics` / no group charts
    handoff = {
        "paradigm": "epm",
        "per_subject": {"Subject 1": metrics_result},
        "charts": [str(traj_path)],
    }
    assert len(handoff["per_subject"]) == 1
    assert "statistics" not in handoff
```

- [ ] **Step 2: 跑 e2e 测试**

Run: `cd packages/ethoinsight && python -m pytest tests/scripts/test_epm_e2e.py -v`
Expected: 2 个 e2e 测试 PASS

- [ ] **Step 3: 全量回归**

Run: `cd packages/ethoinsight && python -m pytest tests/ -q`
Expected: 现有 170+ 测试 + scripts 新增 ~13 测试 全绿

- [ ] **Step 4: Commit**

```bash
git add packages/ethoinsight/tests/scripts/test_epm_e2e.py
git commit -m "test(scripts): EPM 端到端编排测试（Plan A T7）

模拟 code-executor subagent 真实工作流：
- happy path: 6 subjects × 5 metrics + box plot + groupwise stats
- single-subject path: n=1 描述性 + trajectory plot, 无 stats

验证脚本接口契约稳定，agent prompt 可放心依赖。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8：ScriptInvocationOnlyProvider Guardrail + 单测

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`
- Test: `packages/agent/backend/tests/test_script_invocation_only_provider.py`

- [ ] **Step 1: 写 Guardrail 失败测试**

文件 `packages/agent/backend/tests/test_script_invocation_only_provider.py`：

```python
"""Tests for ScriptInvocationOnlyProvider."""

from __future__ import annotations

import pytest

from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def provider():
    from deerflow.guardrails.script_invocation_only_provider import (
        ScriptInvocationOnlyProvider,
    )
    return ScriptInvocationOnlyProvider()


def _req(tool_name: str, command: str = "", agent_id: str = "subagent:code-executor") -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input={"command": command} if command else {},
        agent_id=agent_id,
    )


class TestNonBashAlwaysAllowed:
    def test_read_file_allowed(self, provider):
        decision = provider.evaluate(_req("read_file"))
        assert decision.allow

    def test_write_file_allowed(self, provider):
        decision = provider.evaluate(_req("write_file"))
        assert decision.allow

    def test_ls_allowed(self, provider):
        decision = provider.evaluate(_req("ls"))
        assert decision.allow


class TestNonCodeExecutorAgentNotGated:
    def test_lead_agent_bash_allowed(self, provider):
        # lead agent has no agent_id passport set by GuardrailMiddleware in our config
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'", agent_id=""))
        assert decision.allow

    def test_other_subagent_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'",
                                          agent_id="subagent:data-analyst"))
        assert decision.allow


class TestCodeExecutorBashAllowList:
    def test_script_invocation_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio --input /tmp/a.txt --output /tmp/o.json",
        ))
        assert decision.allow

    def test_common_script_invocation_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts._common.plot_trajectory --input /tmp/a.txt --output /tmp/p.png",
        ))
        assert decision.allow

    def test_mkdir_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mkdir -p /mnt/user-data/workspace/outputs"))
        assert decision.allow

    def test_cp_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cp /a /b"))
        assert decision.allow

    def test_ls_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="ls /tmp"))
        assert decision.allow


class TestCodeExecutorBashDenyList:
    def test_python_c_help_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight import parse; help(parse.parse_trajectory)"',
        ))
        assert not decision.allow
        assert decision.reasons
        assert "script" in decision.reasons[0].message.lower()

    def test_python_c_import_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight import charts; print(charts.box_plot)"',
        ))
        assert not decision.allow

    def test_python_c_smoke_test_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight.metrics.epm import *; print(\'OK\')"',
        ))
        assert not decision.allow

    def test_pip_install_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="pip install pandas"))
        assert not decision.allow

    def test_arbitrary_python_script_denied(self, provider):
        # python invoking a custom script outside ethoinsight.scripts is denied
        decision = provider.evaluate(_req(
            "bash",
            command="python /tmp/my_analysis.py --input /tmp/a.txt",
        ))
        assert not decision.allow

    def test_rm_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="rm /tmp/file"))
        assert not decision.allow

    def test_curl_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com"))
        assert not decision.allow


class TestDenyReason:
    def test_reason_message_contains_correct_path_hint(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "python -m ethoinsight.scripts" in msg
        assert "by-paradigm" in msg

    def test_reason_code_is_stable(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'"))
        assert decision.reasons[0].code == "script_invocation_only.not_a_script_call"


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate(provider):
    req = _req("bash", command="python -c 'help(x)'")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
    assert sync_decision.reasons[0].code == async_decision.reasons[0].code
```

- [ ] **Step 2: 跑测试验证失败**

Run: `cd packages/agent/backend && PYTHONPATH=. python -m pytest tests/test_script_invocation_only_provider.py -v`
Expected: 失败，`ModuleNotFoundError: deerflow.guardrails.script_invocation_only_provider`

- [ ] **Step 3: 实现 Guardrail provider**

文件 `packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`：

```python
"""ScriptInvocationOnlyProvider — gate code-executor's bash to script invocations + file ops only.

This Guardrail enforces the 'script-per-metric' architecture: code-executor's
bash tool must only be used to either invoke an `ethoinsight.scripts.*` script
via ``python -m`` or perform safe file operations (mkdir / cp / mv / ls / cat /
grep / head / tail). Any other bash command (including ``python -c``,
``pip install``, arbitrary scripts) is denied with a reason that tells the
agent the correct path forward.

White-list rather than black-list: the allowed shape is small and stable; new
scripts are auto-allowed by the same pattern without touching this provider.

Only applies to subagents whose agent_id starts with 'subagent:code-executor'.
Lead agent and other subagents pass through unchanged.

See: docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §4
"""

from __future__ import annotations

import re

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


# Match `python -m ethoinsight.scripts.<paradigm>.<script>` at the start of the command.
# Supports leading whitespace and any args after the module name.
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.scripts\.\w+\.\w+(\s|$)"
)

# Match safe file-operation commands at start of command.
_ALLOWED_FILE_OPS = re.compile(
    r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
)


_DENY_MESSAGE = (
    "该 bash 命令不是脚本调用。code-executor 仅可：\n"
    "  1. 调脚本：python -m ethoinsight.scripts.<paradigm>.<name> --input ... --output ...\n"
    "  2. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
    "请改用脚本调用形式。可用脚本清单见 by-paradigm/<范式>.md。"
)


class ScriptInvocationOnlyProvider:
    """Whitelist bash commands for code-executor to script invocations + file ops."""

    name = "script_invocation_only"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Only gate bash tool calls.
        if request.tool_name != "bash":
            return GuardrailDecision(allow=True)

        # Only gate code-executor subagent.
        if "code-executor" not in (request.agent_id or ""):
            return GuardrailDecision(allow=True)

        cmd = request.tool_input.get("command", "")

        if _ALLOWED_PYTHON_PATTERN.match(cmd):
            return GuardrailDecision(allow=True)
        if _ALLOWED_FILE_OPS.match(cmd):
            return GuardrailDecision(allow=True)

        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="script_invocation_only.not_a_script_call",
                    message=_DENY_MESSAGE,
                )
            ],
            policy_id="script_invocation_only",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Pure sync logic; expose async for protocol compliance.
        return self.evaluate(request)
```

- [ ] **Step 4: 跑测试验证通过**

Run: `cd packages/agent/backend && PYTHONPATH=. python -m pytest tests/test_script_invocation_only_provider.py -v`
Expected: 全部 PASS（~17 个测试）

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py \
        packages/agent/backend/tests/test_script_invocation_only_provider.py
git commit -m "feat(guardrails): ScriptInvocationOnlyProvider 白名单 Guardrail（Plan A T8）

code-executor 的 bash 命令必须是：
- python -m ethoinsight.scripts.<paradigm>.<name> （脚本调用）
- mkdir / cp / mv / ls / cat / grep / head / tail （文件操作）
其他一律 deny + 反馈正确路径。

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9：把 Guardrail 挂到 SubagentExecutor

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:323-335`

- [ ] **Step 1: 读现有挂载段确认位置**

Run: `grep -n "HandoffIsolationProvider\|GuardrailMiddleware" packages/agent/backend/packages/harness/deerflow/subagents/executor.py`
Expected: 输出大约 5 行，包含 line 323-335 的 import + append。

- [ ] **Step 2: 修改 executor.py 追加挂载**

`packages/agent/backend/packages/harness/deerflow/subagents/executor.py` 第 323-335 行附近，在 `HandoffIsolationProvider` 挂载**之后**追加 `ScriptInvocationOnlyProvider` 挂载。

找到现有代码块（参考 executor.py 当前内容）：

```python
        # Attach HandoffIsolationProvider so subagent's read_file on
        # handoff_*.json files is gated by lead's {{handoff://}} authorization.
        from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
        from deerflow.guardrails.middleware import GuardrailMiddleware

        handoff_isolation = HandoffIsolationProvider(
            authorized_paths=self.authorized_handoff_paths,
            self_outbox_subagent_name=self.config.name,
        )
        middlewares.append(GuardrailMiddleware(
            provider=handoff_isolation,
            passport=f"subagent:{self.config.name}",
        ))
```

**在它的下方追加**：

```python
        # Attach ScriptInvocationOnlyProvider so code-executor's bash tool is
        # whitelisted to ethoinsight.scripts.* invocations + safe file ops.
        # Non-code-executor subagents pass through (provider self-gates by agent_id).
        from deerflow.guardrails.script_invocation_only_provider import (
            ScriptInvocationOnlyProvider,
        )

        middlewares.append(GuardrailMiddleware(
            provider=ScriptInvocationOnlyProvider(),
            passport=f"subagent:{self.config.name}",
        ))
```

- [ ] **Step 3: 跑 agent backend 全量测试**

Run: `cd packages/agent/backend && make test`
Expected: 全绿（原有 test 不受影响；ScriptInvocationOnlyProvider 不会影响非 code-executor 的 subagent）

- [ ] **Step 4: 手动验证两个 Guard 并存**

Run:
```bash
cd packages/agent/backend
PYTHONPATH=. python -c "
from deerflow.subagents.config import SubagentConfig
from deerflow.subagents.executor import SubagentExecutor

# Pick code-executor builtin config
from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
executor = SubagentExecutor(
    config=CODE_EXECUTOR_CONFIG,
    tools=[],
    parent_model='deepseek-v4-pro',
    trace_id='test-mount',
    thread_id='test-thread',
    authorized_handoff_paths=set(),
)
agent = executor._create_agent()  # noqa: SLF001
# Inspect middlewares
from deerflow.guardrails.middleware import GuardrailMiddleware
gm = [m for m in agent.middleware if isinstance(m, GuardrailMiddleware)]
print(f'GuardrailMiddleware count: {len(gm)}')
for m in gm:
    print(f'  provider: {m.provider.name}')
"
```

Expected: 输出 `GuardrailMiddleware count: 2`，两个 provider 分别是 `handoff_isolation` 和 `script_invocation_only`。

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/executor.py
git commit -m "feat(executor): 挂载 ScriptInvocationOnlyProvider 到 subagent（Plan A T9）

在 HandoffIsolationProvider 之后追加挂载。两个 Guard 正交，并存：
- HandoffIsolationProvider 管 handoff 文件读取隔离
- ScriptInvocationOnlyProvider 管 code-executor 的 bash 命令白名单

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10：重写 by-paradigm/epm.md（决策手册）

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`（全文重写）

- [ ] **Step 1: 重写 epm.md**

替换 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md` 全部内容为：

```markdown
# EPM (Elevated Plus Maze) 范式

> 学术范式 key: `epm`
> EV19 模板映射: `Elevated Plus Maze` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md`

## 可用脚本清单

所有脚本以 `python -m <module_path> --input ... --output ...` 调用。

### 核心指标脚本（compute_*.py）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.epm.compute_open_arm_time_ratio` | 单轨迹文件 | metric JSON | `{"metric": "open_arm_time_ratio", "value": float \| null}` | 开臂时间占比 |
| `ethoinsight.scripts.epm.compute_open_arm_entry_count` | 单轨迹文件 | metric JSON | `{"metric": "open_arm_entry_count", "value": int \| null}` | 开臂进入次数 |
| `ethoinsight.scripts.epm.compute_open_arm_entry_ratio` | 单轨迹文件 | metric JSON | `{"metric": "open_arm_entry_ratio", "value": float \| null}` | 开臂进入次数 / 总进臂次数 |
| `ethoinsight.scripts.epm.compute_open_arm_time` | 单轨迹文件 | metric JSON | `{"metric": "open_arm_time", "value": float \| null}` | 开臂总停留时间（秒） |
| `ethoinsight.scripts.epm.compute_total_entry_count` | 单轨迹文件 | metric JSON | `{"metric": "total_entry_count", "value": int \| null}` | 总进臂次数 |

### 通用指标脚本（任范式可用）

| 脚本 module | --input | --output | 输出 JSON | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.compute_distance_moved` | 单轨迹文件 | metric JSON | `{"metric": "distance_moved", "value": float \| null}` | 总移动距离 |
| `ethoinsight.scripts._common.compute_velocity_stats` | 单轨迹文件 | metric JSON | `{"metric": "velocity_stats", "value": {mean, std, max, min, median} \| null}` | 速度描述统计 |

### 可视化脚本（plot_*.py）

| 脚本 module | --input / --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts._common.plot_trajectory` | `--input <单文件>` 或 `--inputs <inputs.json>` | — | PNG | 轨迹图（**用户提到"轨迹"必跑**） |
| `ethoinsight.scripts.epm.plot_box_open_arm` | `--inputs <inputs.json>` | `--groups <groups.json>` | PNG | 开臂时间组间对比箱线图 |

### 统计脚本（run_*_stats.py）

| 脚本 module | --inputs | --groups | --output | 含义 |
|---|---|---|---|---|
| `ethoinsight.scripts.epm.run_groupwise_stats` | `<inputs.json>` | `<groups.json>` | stats JSON | 5 指标全 Shapiro-Wilk 决策树检验 |

## 输入文件格式约定

### `--input`（单文件）
直接传 EthoVision 导出 `.txt` 文件的路径。

### `--inputs`（多文件聚合）
传一个 JSON 文件路径，内容是文件路径数组。subagent 先用 `write_file` 生成此 JSON：

```json
["/mnt/user-data/uploads/subject1.txt", "/mnt/user-data/uploads/subject2.txt"]
```

### `--groups`
传一个 JSON 文件路径，内容是分组映射：

```json
{
  "control": ["Subject 1", "Subject 2", "Subject 3"],
  "treatment": ["Subject 4", "Subject 5", "Subject 6"]
}
```

subject 名称由 EthoVision 文件 header 的 `对象名称` 字段决定。

## 实验设计决策树

根据 lead 提供的 `实验设计` 字段裁剪脚本列表：

### n=1（单样本描述性分析）
- ✅ 跑全部 `compute_*.py`（5 个 EPM 核心 + 2 个通用）
- ✅ 跑 `_common.plot_trajectory` 单文件版（用户场景大概率要看轨迹）
- ❌ 跳过 `plot_box_open_arm`（无组间对比意义）
- ❌ 跳过 `run_groupwise_stats`（无统计意义）

### n_per_group ∈ [3, 4]（小样本）
- ✅ 跑全部 `compute_*.py` for each subject
- ✅ 跑 `plot_box_open_arm`（描述性，注意小样本）
- ✅ 跑 `_common.plot_trajectory --inputs` 多文件版
- ⚠️ 跑 `run_groupwise_stats`，但在 handoff 标注 `data_quality_warnings: 小样本统计功效不足`

### n_per_group ≥ 5（标准）
- ✅ 跑全部脚本

### 用户特殊需求
- "只要轨迹图" → 仅跑 `_common.plot_trajectory`
- "跳过统计" → 跳过 `run_groupwise_stats`
- "我想看 X 指标" → 仅跑对应 `compute_X.py`

## handoff JSON 必须字段

`${workspace_path}/handoff_code_executor.json` schema:

```json
{
  "paradigm": "epm",
  "per_subject": {
    "Subject 1": {"open_arm_time_ratio": 0.35, "open_arm_entry_count": 5, ...},
    ...
  },
  "charts": ["/mnt/user-data/workspace/outputs/epm_box.png", ...],
  "statistics": { /* 直接复制 run_groupwise_stats 的输出 JSON，可选 */ },
  "data_quality_warnings": [ /* 见下方 */ ],
  "errors": [ /* 脚本执行报错记录 */ ]
}
```

### data_quality_warnings 触发条件
- subject 数 < 5/组 → `{"severity": "warning", "message": "小样本统计功效不足"}`
- `total_entry_count < 8`（某 subject）→ `{"severity": "warning", "message": "<subject> 总进臂次数过低，疑为运动抑制"}`
- 某指标在所有 subject 都返回 None → `{"severity": "critical", "message": "<metric> 列名识别失败，可能不是 EPM 数据"}`

## 错误处理

脚本返回 non-zero 退出码或 stderr 非空时：

| 错误模式 | 处理 |
|---|---|
| `ValueError: must be a JSON array` | inputs/groups JSON 格式错 → 检查 write_file 生成的 JSON 内容 |
| `FileNotFoundError: <path>` | 路径不存在 → 用 `ls` 核对 |
| `KeyError: 'in_zone_open_arm_1'` 等 | 列名识别失败 → 写入 `data_quality_warnings`，标 critical，向 lead 反问 |
| `UnicodeDecodeError` | 文件编码不是 UTF-16-LE → 文件可能不是 EthoVision 导出 |

## 编排流程（subagent 工作流）

1. **read** 本文件，看清单和决策树
2. **裁剪**：根据 lead 给的实验设计 + 用户需求，决定要跑哪些脚本
3. **准备输入文件**：用 `write_file` 生成 `inputs.json` 和 `groups.json`（如需要）
4. **bash 循环调脚本**：每个脚本一个 bash 调用，**不要把多个脚本拼在一行**
5. **收集**：每个脚本调用后，stdout 含 `[result] {...}` 行；用 `read_file` 读各 metric JSON
6. **聚合**：构造 handoff JSON（schema 见上）
7. **写 handoff**：`write_file` 到 `${workspace_path}/handoff_code_executor.json`
8. **输出 [gate_signals]** 块（详见 code-executor system_prompt 的 `<output>` 段）

## 范式简介（领域知识）

- 焦虑研究主战场：开臂时间 ↓ ＝ 焦虑 ↑
- 标准实验：5 分钟、单次曝光
- 关键混杂因素：运动抑制（看 `total_entry_count` 和 `distance_moved`）
- 详细领域知识：见 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md`
```

- [ ] **Step 2: 验证 md 没有错别字 / 链接没坏**

Run: `grep -E '^\|.*\|.*\|' packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md | head -20`
Expected: 输出 markdown 表格行，格式工整。

人工 review 一遍 md 的可读性。

- [ ] **Step 3: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md
git commit -m "skill(ethoinsight-code): epm.md 重写为脚本清单 + 决策手册（Plan A T10）

从「胶水脚本范例」改为：
- 可用脚本清单（compute / plot / stats 分类）
- 输入文件格式约定（inputs.json / groups.json）
- 实验设计决策树（n=1 / 小样本 / 标准 / 特殊需求）
- handoff JSON schema
- 错误处理对照表
- 编排流程

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11：改造 ethoinsight-code/SKILL.md 顶层 + code_executor.py prompt

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（顶部 workflow 段）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`（system_prompt workflow 段）

- [ ] **Step 1: 读现有 SKILL.md**

Run: `cat packages/agent/skills/custom/ethoinsight-code/SKILL.md`

记录现有结构，准备替换 workflow 段。

- [ ] **Step 2: 修改 SKILL.md 顶部**

把 SKILL.md 的工作流段（通常在文件顶部 `## Workflow` 或 `# 工作流` 部分）替换为以下内容：

```markdown
## 工作流（脚本即指标架构）

code-executor 的工作流程：

1. **read** `references/by-paradigm/<paradigm>.md` —— 看可用脚本清单 + 实验设计决策树
2. **裁剪**：根据 lead 给的实验信息（范式、n、分组、用户特殊需求），决定要跑哪些脚本
3. **准备输入**（如需多文件聚合）：`write_file` 生成 `inputs.json` 和 `groups.json`
4. **bash 循环调脚本**：每个脚本一次 bash 调用，形如：
   ```
   python -m ethoinsight.scripts.<paradigm>.<script_name> --input ... --output ...
   ```
5. **收集**：脚本输出 JSON / PNG，stdout 含 `[result] {...}` 行
6. **聚合**：构造 handoff JSON
7. **写 handoff**：`write_file` 到 `${workspace_path}/handoff_code_executor.json`
8. **输出 [gate_signals]** 块给 lead

### 重要约束

- 不要写胶水脚本拼接代码 —— 所有指标计算已经在脚本里，subagent 只是编排者
- bash 命令必须是脚本调用（`python -m ethoinsight.scripts.*`）或文件操作（mkdir / cp / mv / ls / cat / grep / head / tail）。其他形式的 bash（包括 `python -c`、`pip install`、运行自定义脚本）会被运行时拦截
- 遇到脚本报错：读 stderr → 查对应范式 md 的「错误处理」段 → 决定重试 / 跳过 / 反问 lead

## Reference Materials

- `references/by-paradigm/<paradigm>.md` — 每个范式的脚本清单 + 决策手册
- `templates/output-contract.md` — handoff JSON schema 详细约定
- `references/error-recovery.md` — 通用错误恢复指引
- `references/quality-checks.md` — handoff 写入前的自检清单
```

- [ ] **Step 3: 修改 code_executor.py 的 system_prompt**

`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` 现有 system_prompt 的 `<workflow>` 段替换为：

找到现有的：

```python
<workflow>
1. read `ethoinsight-code/references/by-paradigm/<paradigm>.md` — 看本范式可用的指标函数 + 胶水脚本范例 + handoff schema
2. read `ethoinsight-charts` skill — 按数据特性选图
3. write_file 写胶水脚本（在 `${workspace_path}/analysis.py`） — import ethoinsight.metrics.<范式> + 算指标 + 跑统计 + 出图 + 写 handoff_code_executor.json
4. bash `python ${workspace_path}/analysis.py`
5. 如果失败，traceback 自动回来，改代码重跑（最多 2 次）
</workflow>
```

**替换为**：

```python
<workflow>
1. read `ethoinsight-code/references/by-paradigm/<paradigm>.md` — 看本范式可用脚本清单 + 实验设计决策树
2. 根据 lead 给的实验信息（范式、n、分组、用户特殊需求），决定要跑哪些脚本
3. （如需多文件聚合）write_file 生成 inputs.json 和 groups.json
4. for script in 选中列表:
     bash `python -m ethoinsight.scripts.<paradigm>.<script_name> --input ... --output ...`
5. 收集各脚本输出（JSON 文件 + stdout 的 [result] 行），构造 handoff JSON
6. write_file `${workspace_path}/handoff_code_executor.json`
</workflow>

<bash_constraints>
你的 bash 命令必须是以下两种之一：
- 脚本调用：python -m ethoinsight.scripts.<paradigm>.<name> --input ... --output ...
- 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail

其他形式（包括 python -c、pip install、运行自定义脚本）会被运行时拦截。
所有指标计算逻辑都已封装在 ethoinsight.scripts 脚本里，你只需编排调用。
</bash_constraints>
```

同时把 `<failure>` 段替换为：

```python
<failure>
- 脚本 stderr 非空: 读 traceback → 查范式 md「错误处理」段 → 决定重试 / 跳过 / 反问 lead
- 脚本反复失败: loop_detection middleware 会自动中断，向 lead 报错
- bash 命令被 Guardrail 拒绝: 反馈消息已经告诉你正确路径，直接改用脚本调用形式
</failure>
```

- [ ] **Step 4: 跑 agent backend 测试**

Run: `cd packages/agent/backend && make test`
Expected: 全绿。code_executor.py 的改动只是 prompt 字符串，无逻辑变化。

- [ ] **Step 5: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-code/SKILL.md \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
git commit -m "prompt(code-executor): 切换到脚本即指标工作流（Plan A T11）

SKILL.md 顶部 workflow 段 + code_executor.py system_prompt <workflow> 段
都改为 3 步硬路径：read paradigm md → 选脚本编排 → bash 调脚本。

新增 <bash_constraints> 段，正向描述允许的 bash 模式（白名单）。
Guardrail 在运行时强制执行同样的约束。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12：删除残留的胶水脚本相关内容

**Files:**
- Search and clean: `packages/agent/skills/custom/ethoinsight-code/` 下任何 `analysis.py` 范例 / 胶水脚本 / `compute_paradigm_metrics` 引用

- [ ] **Step 1: 搜索胶水脚本残留**

Run:
```bash
grep -rn "analysis.py\|胶水脚本\|compute_paradigm_metrics" \
  packages/agent/skills/custom/ethoinsight-code/ \
  packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
```

记录所有命中。

- [ ] **Step 2: 逐个评估并清理**

对每个命中：
- 若是 by-paradigm/epm.md：已经在 Task 10 重写，不应有残留 —— 重新检查
- 若是 SKILL.md / SKILL 其他段：删掉胶水脚本相关描述
- 若是 templates/output-contract.md：保留（这是 handoff schema 文档，不是胶水脚本），但 review 一遍措辞确保不再暗示要写 `analysis.py`
- 若是 references/error-recovery.md / quality-checks.md：保留，但 review 措辞

注意：**`ethoinsight.metrics.dispatcher.compute_paradigm_metrics`** 是库代码合法 API，被脚本（如 `plot_box_open_arm.py`、`run_groupwise_stats.py`）使用，**保留**。只清理 **skill md / prompt 中暗示 agent 直接调用它** 的描述。

- [ ] **Step 3: 跑全量测试 + lint**

Run:
```bash
cd packages/ethoinsight && python -m pytest tests/ -q
cd packages/agent/backend && make test && make lint
```

Expected: 全绿。

- [ ] **Step 4: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-code/
git commit -m "skill(ethoinsight-code): 清理胶水脚本残留（Plan A T12）

删除 analysis.py / 胶水脚本 / agent 直接调 compute_paradigm_metrics
的描述。库代码层面的 compute_paradigm_metrics 仍然存在并被脚本调用。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13：本 plan 全量验收（合成数据 + agent 实测）

**Files:**
- 全部已修改的文件

- [ ] **Step 1: ethoinsight 全量测试**

Run: `cd packages/ethoinsight && python -m pytest tests/ -v --tb=short`
Expected: 全绿，特别注意 `tests/scripts/` 下所有新测试 PASS

- [ ] **Step 2: agent backend 全量测试 + lint**

Run:
```bash
cd packages/agent/backend
make test
make lint
```

Expected: 全绿

- [ ] **Step 3: 手动 agent dogfood（合成数据，模拟前端上传）**

启动开发栈：

```bash
cd packages/agent
make dev
```

在浏览器 `localhost:2026`：
- 上传一个合成 EPM 文件（用 Task 2 fixture 同样的方式生成一个 `.txt`）
- 说："我刚做完高架十字迷宫实验，这是 Subject 1 单样本 Drug 组数据，做单样本描述性分析"
- 观察 code-executor subagent 的工具调用序列

**期望行为**：

1. code-executor read by-paradigm/epm.md ✓
2. code-executor 看完后直接 `bash python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio ...`（**不再有 `python -c help` 探测**）
3. 跑 5 个 compute 脚本 + 1 个 `_common.plot_trajectory`
4. write_file handoff_code_executor.json
5. 在 `max_turns=12` 内完成（实际预期 6-10 turn）

**如果观察到 agent 仍然试 `python -c` 探测**：Guardrail 应该拦截并返回 error ToolMessage，agent 应在 1 个 turn 内改路径。

**如果 agent 没有按预期走** —— 不强行修，但记录现象到 Plan B 的 follow-up（可能需要进一步调 prompt）。

- [ ] **Step 4: 验收清单**

确认下列都满足：

- [ ] ethoinsight `tests/scripts/` 下所有测试 PASS（≥ 13 个新测试）
- [ ] agent backend `tests/test_script_invocation_only_provider.py` 所有测试 PASS（~17 个）
- [ ] `make lint` 全绿
- [ ] EPM 7 个脚本（5 compute + 1 plot + 1 stats）+ 3 个通用脚本（2 common compute + 1 plot trajectory）全部存在并可独立运行
- [ ] `by-paradigm/epm.md` 是「脚本清单 + 决策手册」格式
- [ ] `code_executor.py` system_prompt 的 `<workflow>` 段是 3 步硬路径
- [ ] `executor.py` 挂载了 `ScriptInvocationOnlyProvider`
- [ ] 手动 dogfood 中 code-executor 不再用 `python -c` 探测（即使探测了也被 Guard 拦截后立刻改路径）

- [ ] **Step 5: 写本 plan 完成 handoff 文档**

文件 `docs/handoffs/2026-05/2026-05-12-script-per-metric-epm-completed-handoff.md`：

```markdown
# 2026-05-12 Plan A 完成 + Plan B/C 待办 交接

## 已完成（Plan A：脚本即指标 + EPM 验证）

- ethoinsight/scripts/ 包骨架 + CLI helper（_cli.py）
- _common/ 通用脚本（compute_distance_moved / compute_velocity_stats / plot_trajectory）
- epm/ 全量脚本（5 compute + 1 plot + 1 stats）
- ScriptInvocationOnlyProvider 白名单 Guardrail + 挂载
- by-paradigm/epm.md 重写为决策手册
- ethoinsight-code SKILL.md + code_executor.py prompt 切换到脚本编排
- 13 个脚本测试 + 17 个 Guardrail 测试 + 2 个 e2e 编排测试 全绿
- 手动 dogfood 验证 code-executor 走脚本调用路径

## 待办（Plan B：剩余 6 个范式按 EPM 模板补齐）

按 EPM 模板复制：
1. oft/ —— compute_center_time_ratio / compute_thigmotaxis_index / compute_center_distance_ratio / compute_center_entry_count + plot_box_center + run_groupwise_stats
2. zero_maze/ —— compute_open_zone_* (4 个) + plot_box + run_groupwise_stats
3. ldb/ —— compute_light_time_ratio / compute_transition_count / compute_light_latency + plot_box + run_groupwise_stats
4. fst/ —— compute_immobility_* (3 个) + plot_box + run_groupwise_stats
5. tst/ —— compute_immobility_* (3 个) + plot_box + run_groupwise_stats
6. shoaling/ —— compute_inter_individual_distance / compute_nearest_neighbor_distance / compute_group_polarity + plot_box + run_groupwise_stats

每个范式按 EPM 模板：
- 写脚本（照 T3-T6）
- 写 by-paradigm/<paradigm>.md（照 T10）
- 加测试（照 T3-T7）

## 待办（Plan C：前端 reasoning 重复修复）

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §5

## 注意事项

- 真数据列名 regex 调校还没做（接续 2026-05-11-handoff.md），等同事提供真 EthoVision 数据后再做
- Plan A 验证了"agent 走脚本路径"成立，Plan B 是机械复制工作
- 若 dogfood 仍发现 agent 偶尔走偏，先确认 Guardrail 有效拦截，再考虑 prompt 微调

## 关键文件速查

- Spec: docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md
- Plan A: docs/superpowers/plans/2026-05-12-plan-a-script-per-metric-epm.md
- Plan B（计划中）: docs/superpowers/plans/<待生成>
- Plan C（计划中）: docs/superpowers/plans/<待生成>
```

- [ ] **Step 6: Commit handoff**

```bash
git add docs/handoffs/2026-05/2026-05-12-script-per-metric-epm-completed-handoff.md
git commit -m "docs(handoff): Plan A 完成（脚本即指标 + EPM 验证）

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md
和 docs/superpowers/plans/2026-05-12-plan-a-script-per-metric-epm.md。

下一步：Plan B（剩余 6 范式补齐）+ Plan C（前端 reasoning 修复）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

完成全部 13 个任务后，对照 spec §6 自检：

### Spec 覆盖检查

| Spec 章节 | 覆盖 task |
|---|---|
| §6.1.1 新建 scripts/ 包 | T1 |
| §6.1.2 EPM metrics 函数包装为脚本 | T3 + T4 |
| §6.1.3 charts.py 图函数包装 | T5 (plot_trajectory) + T6 (plot_box_open_arm) |
| §6.1.4 statistics.run_groupwise 包装 | T6 (run_groupwise_stats) |
| §6.1.5 每个脚本独立单测 | T3-T7 |
| §6.1.6 SKILL.md + by-paradigm/epm.md 改造 | T10 + T11 |
| §6.1.7 code_executor.py prompt 重写 | T11 |
| §6.1.8 ScriptInvocationOnlyProvider | T8 + T9 |
| §6.2 前端 reasoning 修复 | **不在本 plan，留给 Plan C** |
| §6.3 不动的部分 | 全 plan 不触碰 |
| §6.4 验收标准 | T13 |

**Plan A 不覆盖**：oft / zero_maze / ldb / fst / tst / shoaling 范式脚本化（Plan B 范围）；前端修复（Plan C 范围）。两者已在 Task 13 Step 5 的 handoff 文档中列为待办。

### Placeholder 扫描

无 TBD / TODO / 占位符。所有 code block 都是完整可执行内容。

### Type 一致性

- `emit_result(payload: dict)` 在 T1 定义，T3-T6 全部脚本调用形式一致 ✓
- `make_compute_parser` / `make_plot_parser` / `make_stats_parser` 在 T1 定义，下游使用形式一致 ✓
- `read_inputs_json` / `read_groups_json` 在 T1 定义，T6 stats 脚本使用形式一致 ✓
- Guardrail Decision 类型符合 `provider.py` 中既有 `GuardrailDecision / GuardrailReason` 接口 ✓
- `ScriptInvocationOnlyProvider.name = "script_invocation_only"`，T8 测试和 T9 dogfood 引用一致 ✓
