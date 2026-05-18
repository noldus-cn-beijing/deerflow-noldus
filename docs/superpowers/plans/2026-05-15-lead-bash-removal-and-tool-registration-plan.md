# Lead Bash 移除 + Python 工具注册 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 彻底消除 P0 bug 根因：lead agent 不再有 bash tool，所有 `parse.*`/`catalog.*` 调用包成 deerflow 注册的 Python 工具 `prep_metric_plan`。

**Architecture:** Task 1 修复 LoopDetectionMiddleware 按 tool name 计数兜底（独立可 commit）；Task 2 建 `prep_metric_plan` 工具直接调 `ethoinsight.parse._core` + `ethoinsight.catalog.resolve`，不走 sandbox bash；Task 3 从 lead 工具列表移除 bash/write_file/str_replace；Task 4 删除 G4 LeadAgentExecutionBoundaryProvider + 改 prompt + 改 skill + dogfood 验证。

**Tech Stack:** Python 3.12+, langchain ToolRuntime, ethoinsight library, ThreadDataState (TypedDict)

---

## 全局准备

### Step P.1: 建专用 worktree

```bash
cd /home/wangqiuyang/noldus-insight
git worktree add .claude/worktrees/p0-lead-bash-removal -b sync/p0-lead-bash-removal dev
cd .claude/worktrees/p0-lead-bash-removal
```

之后所有操作在此 worktree 内进行。不要在 dev 主工作树直接跑。

### Step P.2: 确认前置依赖

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal
git log --oneline -5
# 应看到 3ee6bd07 "fix: G1+G4 修复 — 输出宪法 + 强制阶段播报" 在顶部附近
```

### Step P.3: 跑基线测试

```bash
cd packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -5
# 预期: 2315 passed / 3 failed (全预存)
```

记录实际 pass/fail 数量作为 baseline。

### Step P.4: 创建 plan 文件目录

```bash
mkdir -p docs/superpowers/plans
```

---

### Task 1: 修 LoopDetectionMiddleware 兜底（独立可 commit、优先）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py:29-35,130-136,219-221`
- Modify: `packages/agent/backend/tests/test_loop_detection_middleware.py`（新增 5 个测试）

#### 现状分析

`loop_detection_middleware.py` 已有双层检测：

1. **Layer 1 (hash-based)**: 行 107-125，对 tool_calls 做 hash（name + args），同 hash 出现 ≥3 次 warn、≥5 次 hard stop。工作正常但 lead 每次 retry 微调 command 字符串 → hash 不同 → 不触发。

2. **Layer 2 (tool_freq)**: 行 273-305，按 tool name 计数。当前阈值 `_DEFAULT_TOOL_FREQ_WARN = 30`、`_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50`。100 次仍不够触发 —— lead 跑到 recursion 100 次耗尽时 Layer 2 还没到阈值。

**根因**: Layer 2 阈值太高（30/50），lead 跑了 100 次 recursion 耗尽，Layer 2 理论上应该触发但实际没触发 —— 需确认是 (a) 阈值太高导致 100 次不够，还是 (b) counter 被某机制重置。从代码看 `_tool_freq` 计数器按 thread_id 累计，无重置逻辑，因此 100 次 bash 应该 `tc_count >= 30` 触发 warn。如果实际没触发，可能是 `after_model` 在 recursion 耗尽场景下没被调用。

不管具体原因，修复方案：**降阈值到 3/5** 确保即使 hash-based 漏过，tool_freq 也能兜底。

- [ ] **Step 1: 写 5 个 failing test**

在 `packages/agent/backend/tests/test_loop_detection_middleware.py` 末尾追加：

```python
# === Task 1: tool name frequency with lower thresholds ===


def _make_minimal_runtime(thread_id="test-thread"):
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id}
    return runtime


def _make_state_with_tool_calls(tool_calls, content=""):
    msg = AIMessage(content=content, tool_calls=tool_calls)
    return {"messages": [msg]}


class TestToolNameFreqWithBash:
    """P0 fix: tool name frequency catches repeated bash calls with varying args."""

    def test_bash_tool_freq_warns_at_3_with_different_commands(self):
        """同 bash 3 次不同 command → 触发 warn (hash 不同但 tool name 同)。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_minimal_runtime()

        results = []
        for i in range(4):
            tc = {
                "name": "bash",
                "args": {"command": f"cd /tmp && python -m ethoinsight.parse.dump_headers --input file_{i}.txt"},
            }
            warning, hard_stop = mw._track_and_check(_make_state_with_tool_calls([tc]), runtime)
            results.append((warning, hard_stop))

        # 第 1-2 次: 不触发
        assert results[0] == (None, False)
        assert results[1] == (None, False)
        # 第 3 次: 触发 warn
        assert results[2][0] is not None
        assert "bash" in results[2][0]
        assert results[2][1] is False
        # 第 4 次: 不再重复 warn (已 warned)
        assert results[3][0] is None

    def test_bash_tool_freq_hard_stop_at_5(self):
        """同 bash 5 次不同 command → hard limit + strip tool_calls。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_minimal_runtime()

        for i in range(5):
            tc = {
                "name": "bash",
                "args": {"command": f"python -m ethoinsight.parse.dump_headers --input file_{i}.txt"},
            }
            _ = mw._track_and_check(_make_state_with_tool_calls([tc]), runtime)

        # 第 5 次: hard stop
        state6 = _make_state_with_tool_calls([{
            "name": "bash",
            "args": {"command": "echo still trying"},
        }])
        result = mw._apply(state6, runtime)
        assert result is not None
        assert "messages" in result
        updated_msg = result["messages"][0]
        assert updated_msg.tool_calls == []
        # content 必须含 hard stop 消息
        content_str = updated_msg.content if isinstance(updated_msg.content, str) else str(updated_msg.content)
        assert "FORCED STOP" in content_str or "exceeded" in content_str.lower()

    def test_different_tool_names_dont_trigger(self):
        """不同 tool name 各 4 次 → 不触发 (不是同 tool)。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_minimal_runtime()

        for _ in range(4):
            _ = mw._track_and_check(
                _make_state_with_tool_calls([{"name": "bash", "args": {"command": "ls"}}]),
                runtime,
            )
        for _ in range(4):
            _ = mw._track_and_check(
                _make_state_with_tool_calls([{"name": "read_file", "args": {"path": "/tmp/x"}}]),
                runtime,
            )
        for _ in range(4):
            _ = mw._track_and_check(
                _make_state_with_tool_calls([{"name": "task", "args": {"description": "x", "prompt": "x"}}]),
                runtime,
            )
        # 三个 tool 各 4 次 = 12 次 total，但 bash 没到 5 的 hard_limit
        # 最后一次 check bash 到 4 不触发 hard_stop
        result = mw._track_and_check(
            _make_state_with_tool_calls([{"name": "bash", "args": {"command": "ls"}}]),
            runtime,
        )
        assert result == (None, False)

    def test_counter_does_not_reset_when_other_tool_interleaves(self):
        """同 bash 2 次后切别的 tool 1 次后再 bash 1 次 → bash counter 继续累加不重置。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_minimal_runtime()

        # bash x2
        for i in range(2):
            _ = mw._track_and_check(
                _make_state_with_tool_calls([{"name": "bash", "args": {"command": f"ls {i}"}}]),
                runtime,
            )
        # 切 read_file
        _ = mw._track_and_check(
            _make_state_with_tool_calls([{"name": "read_file", "args": {"path": "/tmp/x"}}]),
            runtime,
        )
        # 回来 bash → 第 3 次，触发 warn
        warning, hard_stop = mw._track_and_check(
            _make_state_with_tool_calls([{"name": "bash", "args": {"command": "ls 99"}}]),
            runtime,
        )
        assert warning is not None
        assert "bash" in warning
        assert hard_stop is False

    def test_warn_message_suggests_code_executor(self):
        """warn 注入消息含"task(code-executor)"建议。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_minimal_runtime()

        for i in range(3):
            _ = mw._track_and_check(
                _make_state_with_tool_calls([{"name": "bash", "args": {"command": f"ls {i}"}}]),
                runtime,
            )

        state = _make_state_with_tool_calls([{"name": "bash", "args": {"command": "ls 99"}}])
        result = mw._apply(state, runtime)
        assert result is not None
        updated_msg = result["messages"][0]
        content = updated_msg.content if isinstance(updated_msg.content, str) else str(updated_msg.content)
        assert "code-executor" in content.lower()
```

- [ ] **Step 2: 运行新测试验证它们 fail**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_loop_detection_middleware.py::TestToolNameFreqWithBash -v
```

预期：5 个新测试全部 FAIL（因为 `_DEFAULT_TOOL_FREQ_WARN=30`，3 次远不够触发 warn，且现有消息不含 "code-executor"）。

- [ ] **Step 3: 修改默认阈值**

在 `loop_detection_middleware.py` 第 33-35 行：

```python
# 旧:
_DEFAULT_TOOL_FREQ_WARN = 30  # warn after 30 calls to the same tool type
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50  # force-stop after 50 calls to the same tool type

# 改为:
_DEFAULT_TOOL_FREQ_WARN = 3  # warn after 3 calls to the same tool type
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 5  # force-stop after 5 calls to the same tool type
```

- [ ] **Step 4: 修改 `_TOOL_FREQ_WARNING_MSG`**

在 `loop_detection_middleware.py` 第 130-132 行：

```python
# 旧:
_TOOL_FREQ_WARNING_MSG = (
    "[LOOP DETECTED] You have called {tool_name} {count} times without producing a final answer. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."
)

# 改为:
_TOOL_FREQ_WARNING_MSG = (
    "[LOOP DETECTED] You have called {tool_name} {count} times without success."
    " If you are trying to run analysis commands (parse.*, catalog.*), use task(code-executor) instead of bash."
    " If you need to generate metric_plan.json, use the prep_metric_plan tool."
    " Stop using {tool_name} and produce a decision now."
)
```

- [ ] **Step 5: 修改 `_TOOL_FREQ_HARD_STOP_MSG`**

在 `loop_detection_middleware.py` 第 136 行：

```python
# 旧:
_TOOL_FREQ_HARD_STOP_MSG = "[FORCED STOP] Tool {tool_name} called {count} times — exceeded the per-tool safety limit. Producing final answer with results collected so far."

# 改为:
_TOOL_FREQ_HARD_STOP_MSG = "[FORCED STOP] Tool {tool_name} called {count} times — exceeded the per-tool safety limit. All tool_calls stripped. Produce a final text answer now summarizing what to do next (e.g., dispatch task(code-executor) or ask_clarification)."
```

- [ ] **Step 6: 运行全部 loop_detection 测试**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_loop_detection_middleware.py -v
```

预期：ALL PASS（旧测试 + 5 个新测试）。

- [ ] **Step 7: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py
git add packages/agent/backend/tests/test_loop_detection_middleware.py
git commit -m "$(cat <<'EOF'
fix: LoopDetectionMiddleware 按 tool name 计数(P0 兜底)

tool_freq 阈值从 30/50 降至 3/5，hash-based 漏过时兜底。
warn 注入消息含 task(code-executor) 建议。
EOF
)"
```

---

### Task 2: 新建 `prep_metric_plan` Python 工具

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py:1-13`
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/tools.py:14-18`
- Create: `packages/agent/backend/tests/test_prep_metric_plan_tool.py`

- [ ] **Step 1: 写 5 个 failing test**

创建 `packages/agent/backend/tests/test_prep_metric_plan_tool.py`：

```python
"""Tests for prep_metric_plan_tool."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deerflow.tools.builtins.prep_metric_plan_tool import (
    prep_metric_plan_tool,
    _ERROR_HINTS,
)


def _make_runtime(workspace_path, uploads_path=None, outputs_path=None, thread_id="test-thread"):
    """Build a minimal ToolRuntime[ContextT, ThreadState] mock."""
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id}
    runtime.state = {
        "thread_data": {
            "workspace_path": workspace_path,
            "uploads_path": uploads_path or workspace_path.replace("workspace", "uploads"),
            "outputs_path": outputs_path or workspace_path.replace("workspace", "outputs"),
        }
    }
    return runtime


def _write_ethovision_file(path: str, columns: list[str]):
    """Write a minimal UTF-16 LE EthoVision trajectory file.

    Writes a file with BOM, header line count line, column names, and one data row.
    """
    content = f'﻿"10"\r\n'
    content += f'"{";".join(columns)}"\r\n'
    content += f'"{";".join(columns)}"\r\n'
    for i in range(9):
        content += '-1.0\r\n'
    with open(path, "w", encoding="utf-16-le") as f:
        f.write(content)


EPM_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "in zone Open arms 1 / Center-point",
    "in zone Open arms 2 / Center-point",
    "in zone Closed arm 1 / Center-point",
    "in zone Closed arm 2 / Center-point",
    "in zone Center-point / Center-point",
]


class TestPrepMetricPlanToolOk:
    def test_normal_path_with_epm_data(self):
        """正常路径: mock EthoVision EPM 数据 → status=ok, metric_count > 0。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            uploads = Path(tmpdir) / "uploads"
            uploads.mkdir()

            data_file = uploads / "test_epm.txt"
            _write_ethovision_file(str(data_file), EPM_COLUMNS)

            runtime = _make_runtime(str(workspace), str(uploads))
            result = prep_metric_plan_tool(
                runtime=runtime,
                uploaded_file=f"/mnt/user-data/uploads/test_epm.txt",
                paradigm="epm",
            )

            assert result["status"] == "ok"
            assert result["plan_summary"]["paradigm"] == "epm"
            assert result["plan_summary"]["metric_count"] > 0
            assert "open_arm_time_ratio" in result["plan_summary"]["metric_ids"]
            # plan_path 真实存在
            plan_path = result["plan_path"]
            assert Path(plan_path).exists()
            plan_data = json.loads(Path(plan_path).read_text())
            assert "metrics" in plan_data


class TestPrepMetricPlanToolErrors:
    def test_file_not_found(self):
        """传不存在的路径 → error_code=file_not_found, hint 含 ask_clarification。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            uploads = Path(tmpdir) / "uploads"
            uploads.mkdir()

            runtime = _make_runtime(str(workspace), str(uploads))
            result = prep_metric_plan_tool(
                runtime=runtime,
                uploaded_file="/mnt/user-data/uploads/nonexistent.txt",
                paradigm="epm",
            )

            assert result["status"] == "error"
            assert result["error_code"] == "file_not_found"
            assert "ask_clarification" in result["hint"].lower()

    def test_unknown_paradigm(self):
        """传 paradigm='invalid' → error_code=unknown_paradigm。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            uploads = Path(tmpdir) / "uploads"
            uploads.mkdir()

            data_file = uploads / "test.txt"
            _write_ethovision_file(str(data_file), EPM_COLUMNS)

            runtime = _make_runtime(str(workspace), str(uploads))
            result = prep_metric_plan_tool(
                runtime=runtime,
                uploaded_file=f"/mnt/user-data/uploads/test.txt",
                paradigm="invalid_paradigm",
            )

            assert result["status"] == "error"
            assert result["error_code"] == "unknown_paradigm"

    def test_columns_missing(self):
        """mock 数据缺 in_zone_open_arms_* 列, paradigm=epm → error_code=columns_missing, hint 含'录制设置'。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            uploads = Path(tmpdir) / "uploads"
            uploads.mkdir()

            # 只有基础列，没有 in zone Open arms
            minimal_columns = [
                "Trial time",
                "Recording time",
                "X center",
                "Y center",
            ]
            data_file = uploads / "minimal.txt"
            _write_ethovision_file(str(data_file), minimal_columns)

            runtime = _make_runtime(str(workspace), str(uploads))
            result = prep_metric_plan_tool(
                runtime=runtime,
                uploaded_file=f"/mnt/user-data/uploads/minimal.txt",
                paradigm="epm",
            )

            assert result["status"] == "error"
            assert result["error_code"] == "columns_missing"
            assert "录制设置" in result["hint"] or "列" in result["hint"]

    def test_plan_file_written_on_success(self):
        """status=ok 后 plan_path 真实存在 + JSON 可读。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            uploads = Path(tmpdir) / "uploads"
            uploads.mkdir()

            data_file = uploads / "test2.txt"
            _write_ethovision_file(str(data_file), EPM_COLUMNS)

            runtime = _make_runtime(str(workspace), str(uploads))
            result = prep_metric_plan_tool(
                runtime=runtime,
                uploaded_file=f"/mnt/user-data/uploads/test2.txt",
                paradigm="epm",
            )

            assert result["status"] == "ok"
            plan_path = Path(result["plan_path"])
            assert plan_path.exists()
            plan_data = json.loads(plan_path.read_text())
            assert isinstance(plan_data, dict)
            assert "metrics" in plan_data
            assert len(plan_data["metrics"]) == result["plan_summary"]["metric_count"]
```

- [ ] **Step 2: 运行新测试验证它们 fail**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py -v
```

预期：5 个测试 FAIL（`prep_metric_plan_tool` 不存在）。

- [ ] **Step 3: 写 `prep_metric_plan_tool.py`**

创建 `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`：

```python
"""prep_metric_plan — 一步生成 metric_plan.json，无需 bash。

lead agent 专用：直接调 ethoinsight Python 函数解析 EthoVision 文件 +
catalog resolve，结果写入 workspace/metric_plan.json。

这是 P0 fix 的核心：lead 不再有 bash tool，所有 ethoinsight CLI 调用
强制走此 Python 工具。
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.sandbox.tools import replace_virtual_path

logger = logging.getLogger(__name__)

# 错误码→hint 模板（给 lead 看的下一步建议）
_ERROR_HINTS: dict[str, str] = {
    "file_not_found": (
        "数据文件不存在，可能用户上传失败。用 ask_clarification 让用户重新上传。"
    ),
    "format_unrecognized": (
        "文件不是 EthoVision XT 导出格式。用 ask_clarification 让用户确认导出方式。"
    ),
    "parse_failed": (
        "数据文件损坏，无法解析。用 ask_clarification 让用户重新导出。"
    ),
    "unknown_paradigm": (
        "范式不在 catalog 内。用 ask_clarification 让用户确认范式，"
        "或检查 set_experiment_paradigm 调用是否正确。"
    ),
    "columns_missing": (
        "数据缺关键列（可能录制设置漏了 Open/Closed arms 进入次数或相关区域）。"
        "用 ask_clarification 让用户确认实验录制设置。"
    ),
    "schema_violation": (
        "catalog YAML 损坏——这是项目内部 bug。present_files 把错误信息呈现给用户，让他报 bug。"
    ),
    "empty_plan": (
        "按当前参数一项指标都跑不了。用 ask_clarification 确认用户需求。"
    ),
    "unknown_metric": (
        "用户要求的指标不在 catalog 中。用 ask_clarification 让用户从可用指标中选择。"
    ),
}


@tool("prep_metric_plan", parse_docstring=True)
def prep_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_file: str,
    paradigm: str,
) -> dict:
    """一步生成 metric_plan.json，无需 bash。

    Args:
      uploaded_file: 虚拟路径如 /mnt/user-data/uploads/xxx.txt
      paradigm: 范式如 'epm' / 'oft' / 'fst' / 'ldb' / 'tst' / 'zero_maze'
                / 'shoaling'

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/metric_plan.json",
         "plan_summary": {"paradigm": "epm", "metric_count": 5,
                          "metric_ids": ["open_arm_time_ratio", ...]}}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found"|"format_unrecognized"|"parse_failed"|
                       "unknown_paradigm"|"columns_missing"|"schema_violation",
         "message": str,
         "hint": str}
    """
    thread_data = runtime.state.get("thread_data") if runtime.state else None

    # Resolve real paths from virtual paths
    real_workspace_path = _resolve_workspace_real_path(thread_data)
    real_file_path = replace_virtual_path(uploaded_file, thread_data)

    # Step 1: check file exists
    if not Path(real_file_path).exists():
        return _error_result(
            "file_not_found",
            f"File not found: {uploaded_file} (resolved to {real_file_path})",
        )

    # Step 2: detect EthoVision format
    try:
        from ethoinsight.parse._core import detect_ethovision
    except ImportError as e:
        return _error_result(
            "parse_failed",
            f"Cannot import ethoinsight library: {e}",
        )

    if not detect_ethovision(real_file_path):
        return _error_result(
            "format_unrecognized",
            f"File {uploaded_file} is not an EthoVision XT export.",
        )

    # Step 3: parse header to get column names
    try:
        from ethoinsight.parse._core import parse_header
    except ImportError:
        pass  # already imported above effectively

    try:
        header = parse_header(real_file_path)
    except Exception as e:
        logger.warning("parse_header failed for %s: %s", uploaded_file, e)
        return _error_result(
            "parse_failed",
            f"Failed to parse header: {e}",
        )

    columns = header.get("columns", [])
    if not columns:
        return _error_result(
            "parse_failed",
            "Parsed header contains no column names.",
        )

    # Step 4: resolve catalog → Plan
    try:
        from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve
    except ImportError as e:
        return _error_result(
            "parse_failed",
            f"Cannot import ethoinsight catalog: {e}",
        )

    try:
        plan = resolve(
            paradigm=paradigm,
            columns=columns,
            raw_files=[real_file_path],
            workspace_dir=real_workspace_path,
            virtual_workspace_dir="/mnt/user-data/workspace",
        )
    except ResolveError as e:
        return _error_result(
            e.code,
            str(e),
            extra_details=e.details,
        )
    except Exception as e:
        logger.exception("Unexpected error during resolve for paradigm=%s", paradigm)
        return _error_result(
            "parse_failed",
            f"Unexpected error during catalog resolve: {e}",
        )

    # Step 5: serialize plan to workspace/metric_plan.json
    plan_dict = plan_to_dict(plan)
    plan_path = Path(real_workspace_path) / "metric_plan.json"
    try:
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return _error_result(
            "parse_failed",
            f"Failed to write metric_plan.json: {e}",
        )

    # Step 6: build summary (只 paradigm/metric_count/metric_ids，不含完整 plan)
    metric_ids = [m.get("id", "") for m in plan_dict.get("metrics", [])]

    logger.info(
        "prep_metric_plan success: paradigm=%s, metric_count=%d, plan=%s",
        paradigm,
        len(metric_ids),
        plan_path,
    )

    return {
        "status": "ok",
        "plan_path": "/mnt/user-data/workspace/metric_plan.json",
        "plan_summary": {
            "paradigm": paradigm,
            "metric_count": len(metric_ids),
            "metric_ids": metric_ids,
        },
    }


def _resolve_workspace_real_path(thread_data: dict | None) -> str:
    """Extract workspace real path from thread_data."""
    if thread_data:
        ws = thread_data.get("workspace_path")
        if ws:
            return ws
    # Fallback for testing
    import tempfile
    return str(Path(tempfile.gettempdir()) / "deerflow-workspace")


def _error_result(code: str, message: str, extra_details: dict | None = None) -> dict:
    """Build a standardised error response dict."""
    hint = _ERROR_HINTS.get(code, "未知错误，请联系开发者。")
    result: dict = {
        "status": "error",
        "error_code": code,
        "message": message,
        "hint": hint,
    }
    if extra_details:
        result["details"] = extra_details
    return result
```

- [ ] **Step 4: 在 `builtins/__init__.py` 导出新工具**

修改 `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py`：

```python
from .clarification_tool import ask_clarification_tool
from .prep_metric_plan_tool import prep_metric_plan_tool
from .present_file_tool import present_file_tool
from .setup_agent_tool import setup_agent
from .task_tool import task_tool
from .view_image_tool import view_image_tool

__all__ = [
    "setup_agent",
    "present_file_tool",
    "ask_clarification_tool",
    "view_image_tool",
    "task_tool",
    "prep_metric_plan_tool",
]
```

- [ ] **Step 5: 在 `tools.py` 的 `BUILTIN_TOOLS` 列表加入 `prep_metric_plan_tool`**

修改 `packages/agent/backend/packages/harness/deerflow/tools/tools.py` 第 14-18 行：

```python
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    set_experiment_paradigm_tool,
    prep_metric_plan_tool,
]
```

同时在文件顶部 import（第 9 行后追加）：

```python
from deerflow.tools.builtins import ask_clarification_tool, prep_metric_plan_tool, present_file_tool, task_tool, view_image_tool
```

将原来的第 9 行：
```python
from deerflow.tools.builtins import ask_clarification_tool, present_file_tool, task_tool, view_image_tool
```

替换为：
```python
from deerflow.tools.builtins import ask_clarification_tool, prep_metric_plan_tool, present_file_tool, task_tool, view_image_tool
```

- [ ] **Step 6: 运行新工具的全部测试验证 pass**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py -v
```

预期：5 tests PASS。

- [ ] **Step 7: 运行全量测试确保无回归**

```bash
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -5
```

预期：pass 数比 baseline +5（新测试），fail 数不变（3 pre-existing）。

- [ ] **Step 8: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py
git add packages/agent/backend/packages/harness/deerflow/tools/tools.py
git add packages/agent/backend/tests/test_prep_metric_plan_tool.py
git commit -m "$(cat <<'EOF'
feat(tools): 加 prep_metric_plan Python 工具, lead 一步生成 metric_plan.json 不走 bash
EOF
)"
```

---

### Task 3: 从 lead 工具列表移除 bash + write_file + str_replace

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:436-438`
- Create: `packages/agent/backend/tests/test_lead_tool_filtering.py`

- [ ] **Step 1: 写 8 个 failing test**

创建 `packages/agent/backend/tests/test_lead_tool_filtering.py`：

```python
"""Tests for lead agent tool filtering (Task 3: remove bash/write_file/str_replace)."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableConfig


def _make_config(subagent_enabled=True, **kwargs):
    """Build a minimal RunnableConfig for make_lead_agent."""
    return {
        "configurable": {
            "subagent_enabled": subagent_enabled,
            **kwargs,
        }
    }


class TestLeadToolExclusions:
    """lead 工具列表中不应有 bash / write_file / str_replace，但 ls / read_file / prep_metric_plan 应在。"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Patch heavy dependencies so make_lead_agent can be imported."""
        self._patchers = [
            patch("deerflow.agents.lead_agent.agent.set_current_user", autospec=True),
            patch("deerflow.agents.lead_agent.agent.get_app_config", autospec=True),
            patch("deerflow.agents.lead_agent.agent.get_summarization_config", autospec=True),
            patch("deerflow.agents.lead_agent.agent.load_agent_config", autospec=True),
            patch("deerflow.agents.lead_agent.prompt.apply_prompt_template", autospec=True),
            patch("deerflow.models.create_chat_model", autospec=True),
            patch("deerflow.tools.resolve_variable", autospec=True),
            patch("deerflow.tools.is_host_bash_allowed", return_value=True),
            patch("deerflow.config.app_config.AppConfig", autospec=True),
            patch("deerflow.config.extensions_config.ExtensionsConfig.from_file", autospec=True),
            patch("deerflow.mcp.cache.get_cached_mcp_tools", return_value=[]),
        ]
        self._mocks = {}
        for p in self._patchers:
            self._mocks[p.attribute] = p.start()

        # Build a minimal mock AppConfig
        mock_config = self._mocks["get_app_config"].return_value
        mock_config.models = [MagicMock(name="default_model")]
        mock_config.models[0].name = "test-model"
        mock_config.models[0].supports_vision = False
        mock_config.models[0].supports_thinking = False
        mock_config.tool_groups = []
        mock_config.tools = []
        mock_config.token_usage.enabled = False
        mock_config.tool_search.enabled = False
        mock_config.guardrails = None
        mock_config.skills.container_path = "/mnt/skills"

        def _get_model_config(name):
            return mock_config.models[0] if name else None
        mock_config.get_model_config = _get_model_config

        # Patch guardrails config to be disabled so no G4 middleware
        with patch("deerflow.config.guardrails_config.get_guardrails_config") as mock_gc:
            mock_gc.return_value.enabled = False
            yield
            mock_gc.stop()

        for p in self._patchers:
            p.stop()

    def _get_lead_tools(self):
        from deerflow.agents.lead_agent.agent import make_lead_agent
        agent = make_lead_agent(RunnableConfig(_make_config()))
        tools = agent.tools  # BaseTool list bound to agent
        return {t.name for t in tools}

    def test_bash_not_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "bash" not in tool_names, f"bash should not be in lead tools, got: {sorted(tool_names)}"

    def test_write_file_not_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "write_file" not in tool_names

    def test_str_replace_not_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "str_replace" not in tool_names

    def test_ls_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "ls" in tool_names, f"ls should be in lead tools (Q4 decision), got: {sorted(tool_names)}"

    def test_read_file_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "read_file" in tool_names, "lead needs read_file for handoff inspection"

    def test_prep_metric_plan_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        assert "prep_metric_plan" in tool_names, "prep_metric_plan must be registered for lead"

    def test_task_and_ask_clarification_and_present_files_in_lead_tools(self):
        tool_names = self._get_lead_tools()
        for name in ("task", "ask_clarification", "present_files"):
            assert name in tool_names, f"{name} must be in lead tools"

    def test_code_executor_tools_still_has_bash(self):
        """subagent(code-executor) 工具列表依然含 bash。"""
        from deerflow.subagents.registry import get_subagent_config
        config = get_subagent_config("code-executor")
        assert config is not None, "code-executor subagent config must exist"
        # tools=None means "all tools available" → bash is available
        if config.tools is not None:
            assert "bash" in config.tools, f"if tools restricted, bash must be in list; got: {config.tools}"
        if config.disallowed_tools is not None:
            assert "bash" not in config.disallowed_tools, f"bash must not be disallowed for code-executor; got: {config.disallowed_tools}"
```

- [ ] **Step 2: 运行新测试验证它们 fail**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_lead_tool_filtering.py -v
```

预期：test_bash_not_in_lead_tools、test_write_file_not_in_lead_tools、test_str_replace_not_in_lead_tools 全部 FAIL（当前 lead 有 bash）。

- [ ] **Step 3: 修改 `agent.py` 的 `make_lead_agent`**

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 第 436-438 行区域：

```python
# 现状（第 436-438 行）:
return create_agent(
    model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort),
    tools=get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled),
    ...

# 改为:
_LEAD_EXCLUDED_TOOLS = frozenset({"bash", "write_file", "str_replace"})
all_lead_tools = get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled)
filtered_lead_tools = [t for t in all_lead_tools if t.name not in _LEAD_EXCLUDED_TOOLS]
logger.info(
    "Lead tools after filtering: %d→%d (excluded: %s)",
    len(all_lead_tools),
    len(filtered_lead_tools),
    sorted(_LEAD_EXCLUDED_TOOLS),
)

return create_agent(
    model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort),
    tools=filtered_lead_tools,
    ...
```

具体实现位置：在 `agent.py` 第 434 行 `lead_tool_groups = ...` 之后、第 436 行 `return create_agent(` 之前插入过滤逻辑。最终第 434-443 行区域变为：

```python
        declared_groups = [g.name for g in app_config.tool_groups] if app_config.tool_groups else None
        lead_tool_groups = declared_groups if declared_groups else None

    _LEAD_EXCLUDED_TOOLS = frozenset({"bash", "write_file", "str_replace"})
    all_lead_tools = get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled)
    filtered_lead_tools = [t for t in all_lead_tools if t.name not in _LEAD_EXCLUDED_TOOLS]
    logger.info(
        "Lead tools after filtering: %d→%d (excluded: %s)",
        len(all_lead_tools),
        len(filtered_lead_tools),
        sorted(_LEAD_EXCLUDED_TOOLS),
    )

    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort),
        tools=filtered_lead_tools,
        middleware=_build_middlewares(config, model_name=model_name, agent_name=agent_name),
```

**注意**：不修改 bootstrap agent 路径（第 414-422 行）—— bootstrap 是特殊流程，保留其工具列表不变。

- [ ] **Step 4: 运行 Task 3 测试验证 pass**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_lead_tool_filtering.py -v
```

预期：8 tests PASS。

- [ ] **Step 5: 运行全量测试确保无回归**

```bash
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -5
```

预期：pass 数比 baseline +13（Task 1 5个 + Task 2 5个 + Task 3 8个，总共新增 18 个测试），fail 数不变。

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git add packages/agent/backend/tests/test_lead_tool_filtering.py
git commit -m "$(cat <<'EOF'
feat(lead): 从 lead 工具列表移除 bash / write_file / str_replace, 所有 ethoinsight CLI 调用强制走 prep_metric_plan
EOF
)"
```

---

### Task 4: 删除 LeadAgentExecutionBoundaryProvider + 改 prompt + 改 skill + dogfood

**Files:**
- Delete: `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`
- Delete: `packages/agent/backend/tests/test_lead_execution_boundary_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:316-326`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:453-454,1107-1127`
- Modify: `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md:26-93`
- Create: `docs/handoffs/2026-05/2026-05-15-p0-fix-dogfood-validation.md`

#### 子任务 4a: 删除 G4 boundary 文件

- [ ] **Step 4a.1: 删 `lead_execution_boundary_provider.py`**

```bash
rm packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py
```

- [ ] **Step 4a.2: 删对应的测试文件**

```bash
rm packages/agent/backend/tests/test_lead_execution_boundary_provider.py
```

- [ ] **Step 4a.3: 从 `agent.py` 移除 G4 boundary 的 import + 注册**

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 第 316-326 行：

```python
# 删除以下整段（第 316-326 行）:
        # LeadAgentExecutionBoundary — block lead from writing scripts or running
        # non-whitelisted bash. Self-gates by agent_id; subagents pass through.
        # See: spec §5.5.1, fix thread b0d3a611 E2E failure root cause A.
        from deerflow.guardrails.lead_execution_boundary_provider import (
            LeadAgentExecutionBoundaryProvider,
        )

        middlewares.append(GuardrailMiddleware(
            provider=LeadAgentExecutionBoundaryProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))


# 删除后，该位置仅剩 Ev19TemplateGuardrail 的注册（第 306-314 行）。
```

注意：保留 `Ev19TemplateGuardrailProvider` 的注册（第 306-314 行），只删 `LeadAgentExecutionBoundaryProvider` 部分。

- [ ] **Step 4a.4: 运行测试确认删除不破坏任何东西**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/ -k "not test_lead_execution_boundary" -v 2>&1 | tail -10
# 确保没有 ImportError
make test 2>&1 | tail -5
```

- [ ] **Step 4a.5: Commit 子任务 4a**

```bash
git rm packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py
git rm packages/agent/backend/tests/test_lead_execution_boundary_provider.py
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "$(cat <<'EOF'
feat: 删除 G4 LeadAgentExecutionBoundaryProvider (bash 已从 lead tool 列表移除)
EOF
)"
```

#### 子任务 4b: 改 lead prompt

- [ ] **Step 4b.1: 更新 transparency 表（第 453-454 行）**

修改 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 第 453-454 行：

```python
# 旧（第 453-454 行）:
| 跑 `python -m ethoinsight.parse.dump_headers` | "📂 正在解析 EthoVision 文件结构..." |
| 跑 `python -m ethoinsight.catalog.resolve` | "📋 正在生成指标计划..." |

# 改为:
| 调 `prep_metric_plan` | "📋 正在生成指标计划..." |
```

删除第 453 行整行，将第 454 行的内容替换为新的一行。

- [ ] **Step 4b.2: 更新 Step 0.5 段（第 1119-1127 行）**

```python
# 旧（第 1119-1127 行）:
### Step 0.5: 生成 metric_plan.json（**派遣 code-executor 前必做**，详见 ethoinsight-metric-catalog skill）

1. bash dump_headers 提取数据列名到 /mnt/user-data/workspace/columns.json
2. write_file /mnt/user-data/workspace/raw_files.json（JSON 数组含 raw 文件路径）
3. bash catalog.resolve 生成 /mnt/user-data/workspace/metric_plan.json
4. 派遣 prompt 仅需告诉 code-executor plan.json 路径，**不要展开指标清单**

resolve 失败时（stderr JSON 含 code 字段）按 skill 的话术映射反问用户。

# 改为:
### Step 0.5: 生成 metric_plan.json（**派遣 code-executor 前必做**，详见 ethoinsight-metric-catalog skill）

调 `prep_metric_plan(uploaded_file=<path>, paradigm=<id>)` 一步完成。
工具内部直接解析列名 + 用 catalog 生成 plan，无需 bash。

返回 `status="ok"` 时，`plan_summary` 含 `metric_count` + `metric_ids`，继续派 code-executor。
返回 `status="error"` 时，按 `hint` 字段建议处理（通常是 ask_clarification）。

派遣 prompt 仅需告诉 code-executor plan.json 路径，**不要展开指标清单**。
```

- [ ] **Step 4b.3: 更新 skill 说明段（第 1107 行）**

```python
# 旧（第 1107 行）:
- **ethoinsight-metric-catalog**: 范式指标 catalog 读取手册。**在派遣 code-executor 之前**，按 SKILL.md 中 lead role 段的指引：(1) bash dump_headers 提取列名 (2) bash catalog.resolve 生成 metric_plan.json。失败时按 stderr JSON 的 code 字段 ask_clarification。

# 改为:
- **ethoinsight-metric-catalog**: 范式指标 catalog 读取手册。**在派遣 code-executor 之前**，调 `prep_metric_plan` 工具生成 metric_plan.json。失败时按返回的 error_code + hint 字段 ask_clarification。
```

- [ ] **Step 4b.4: Commit 子任务 4b**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "$(cat <<'EOF'
feat(lead): prompt Step 0.5 改 prep_metric_plan 工具，删 bash CLI 指令
EOF
)"
```

#### 子任务 4c: 改 ethoinsight-metric-catalog skill

- [ ] **Step 4c.1: 重写 SKILL.md 的 lead role 段**

读取 `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`，将第 26 行到第 93 行（从 `### lead` 到 `### data-analyst` 之前）替换为：

```markdown
### lead

在派遣 code-executor **之前**，调 `prep_metric_plan` 工具一步完成列名解析 + catalog resolve。

```
prep_metric_plan(uploaded_file="/mnt/user-data/uploads/<raw_file>.txt", paradigm="<epm|oft|fst|...>")
```

**参数**：
- `uploaded_file`：用户上传的数据文件虚拟路径（如 `/mnt/user-data/uploads/轨迹.txt`）
- `paradigm`：范式 ID（`epm` / `oft` / `fst` / `ldb` / `tst` / `zero_maze` / `shoaling`）

**成功返回** (`status="ok"`)：
```json
{
  "status": "ok",
  "plan_path": "/mnt/user-data/workspace/metric_plan.json",
  "plan_summary": {
    "paradigm": "epm",
    "metric_count": 5,
    "metric_ids": ["open_arm_time_ratio", "open_arm_entries_ratio", ...]
  }
}
```

**失败返回** (`status="error"`)：

| error_code | 含义 | 怎么反问（hint） |
|------------|------|------------------|
| `file_not_found` | 数据文件不存在，可能用户上传失败 | ask_clarification 让用户重新上传 |
| `format_unrecognized` | 文件不是 EthoVision XT 导出格式 | ask_clarification 让用户确认导出方式 |
| `parse_failed` | 数据文件损坏 | ask_clarification 让用户重新导出 |
| `unknown_paradigm` | 范式不在 catalog 内 | ask_clarification 让用户确认范式或检查 set_experiment_paradigm 调用 |
| `columns_missing` | 数据缺关键列（可能录制设置漏了 Open/Closed arms 进入次数） | ask_clarification 让用户确认实验录制设置 |
| `schema_violation` | catalog YAML 损坏——项目内部 bug | present_files 把错误信息呈现给用户，让他报 bug |
| `empty_plan` | 按当前参数一项指标都跑不了 | ask_clarification 确认用户需求 |
| `unknown_metric` | 用户要求的指标不在 catalog 中 | ask_clarification 让用户从可用指标中选择 |

工具返回的 `hint` 字段已包含上述反问话术，可直接用于 ask_clarification。

**派遣 code-executor**

派遣 prompt 中只需要：

```
范式：{paradigm}
plan 路径：/mnt/user-data/workspace/metric_plan.json
分组：/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条执行
```

不要把指标清单展开在 prompt 里。
```

- [ ] **Step 4c.2: Commit 子任务 4c**

```bash
git add packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md
git commit -m "$(cat <<'EOF'
feat(skill): ethoinsight-metric-catalog lead role 段改用 prep_metric_plan 工具
EOF
)"
```

#### 子任务 4d: dogfood 验证

- [ ] **Step 4d.1: 启动应用**

```bash
cd packages/agent
make stop 2>/dev/null; sleep 1
make dev
# 等待所有服务就绪（LangGraph 2024, Gateway 8001, Frontend 3000, Nginx 2026）
# 用 until grep 确认:
# until curl -s http://localhost:2026 >/dev/null 2>&1; do sleep 2; done
```

- [ ] **Step 4d.2: 跑跟 P0 现场一致的 dogfood**

打开浏览器 `http://localhost:2026`：
1. 上传 EPM 数据文件（用 `demo-data/` 下的 EthoVision XT 轨迹文件）
2. 发送消息："请分析这份 EPM 数据"
3. 发送消息："做单样本分析"

或等效地用 `curl` / `DeerFlowClient` 编程式执行。

- [ ] **Step 4d.3: 验证清单**

逐项确认：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | lead 调了 prep_metric_plan 且返回 status=ok | 看 chat 输出或 langgraph.log grep "prep_metric_plan" |
| 2 | lead 成功派了 code-executor | 看 task 被触发，code-executor 返回了 `[gate_signals]` |
| 3 | 没有 lead 重复 bash 100 次 recursion 耗尽 | `grep -c "recursion_limit\|RuntimeError" packages/agent/logs/langgraph.log` |
| 4 | langgraph.log 没有 LoopDetectionMiddleware 因 bash 触发 (加分项) | `grep "LOOP DETECTED.*bash" packages/agent/logs/langgraph.log` 应为空 |
| 5 | 最终用户收到了分析结果（含图表 + 统计 + data-analyst 解读） | 浏览器查看 |

- [ ] **Step 4d.4: 如果验证不通过**

按优先级逐层回滚：
1. Task 4c（skill）→ `git revert <commit>` 检查是否是 skill 措辞误导 lead
2. Task 4b（prompt）→ `git revert <commit>` 检查是否是 prompt 措辞问题
3. Task 3（tool filtering）→ `git revert <commit>` 检查是否是应保留的 tool 被误删
4. Task 2（prep_metric_plan）→ `git revert <commit>` 检查是否是工具实现 bug

**不要瞎合**。单独 revert 定位问题层，fix 后再重新 apply 后续 commit。

- [ ] **Step 4d.5: 写 dogfood 验证报告**

创建 `docs/handoffs/2026-05/2026-05-15-p0-fix-dogfood-validation.md`：

```markdown
# P0 fix dogfood 验证报告

**日期**: 2026-05-15
**分支**: sync/p0-lead-bash-removal
**基线 commit**: (Task 4a-c 提交后)

## 验证环境

- worktree: .claude/worktrees/p0-lead-bash-removal
- 数据: demo-data/ (EPM EthoVision XT 轨迹文件)
- 范式: epm

## 验证流程

1. 上传 EPM 数据 → 发"请分析这份 EPM 数据" → "做单样本分析"

## 验证结果

| 检查项 | 结果 | 备注 |
|--------|------|------|
| lead 调 prep_metric_plan 成功 | ✅/❌ | |
| lead 派 code-executor 成功 | ✅/❌ | |
| 无 recursion 100 耗尽 | ✅/❌ | |
| langgraph.log 无 LoopDetectionMiddleware 触发 | ✅/❌ | |
| 用户收到完整分析结果 | ✅/❌ | |

## 问题记录

(如有验证不过的项，记录发现现象 + 定位过程 + fix commit)

## 结论

(修复验证通过 / 需要追加 fix)
```

- [ ] **Step 4d.6: Commit 子任务 4d（验证报告）**

```bash
git add docs/handoffs/2026-05/2026-05-15-p0-fix-dogfood-validation.md
git commit -m "docs: P0 fix dogfood 验证报告"
```

---

### 最终验证

- [ ] **Step F.1: 全量测试**

```bash
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -10
```

预期：baseline 2315 + 18 新增 = ~2333 passed，3 pre-existing failed（来自 test_client_live.py 或 test_gateway.py 的预存失败）。

- [ ] **Step F.2: 检查 commit 历史**

```bash
git log --oneline -5
# 应看到 4 个 commit（Task 1-4 各有 commit）
```

---

## 风险与约束

1. **不要 push** 任何东西直到用户授权
2. **不要碰这 4 个文件**（spec-handoff 交接文档明令）：
   - `docs/specs/llm-finetuning-strategy.md`
   - `packages/agent/frontend/src/app/page.tsx`
   - `packages/agent/scripts/serve.sh`
   - `packages/agent/skills/public/bootstrap/SKILL.md`
3. **Task 1 单独 commit** — 它是兜底机制，即使 Task 2-4 因故卡住，Task 1 单独合 dev 就能止血（recursion 100 不再发生）
4. **TDD 强制** — 每个 task 先写 failing test 再实现
5. **遇到 surgical merge 困难或决策点，STOP 上报** — 不要瞎判断（CLAUDE.md 第 18 条）
6. **`guardrails/__init__.py` 不需要改** — 它的 export 列表（AllowlistProvider, GuardrailMiddleware, GuardrailDecision, GuardrailProvider, GuardrailReason, GuardrailRequest）不包含 `LeadAgentExecutionBoundaryProvider`，所以删文件后不会有 import error
