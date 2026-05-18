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
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`（修改两处常量 `_DEFAULT_TOOL_FREQ_WARN` / `_DEFAULT_TOOL_FREQ_HARD_LIMIT`、两处文案 `_TOOL_FREQ_WARNING_MSG` / `_TOOL_FREQ_HARD_STOP_MSG`）
- Modify: `packages/agent/backend/tests/test_loop_detection_middleware.py`（新增 5 个测试）

#### 现状分析

`loop_detection_middleware.py` 已有双层检测：

1. **Layer 1 (hash-based)**: 对 tool_calls 做 hash（name + stable args key），同 hash 出现 ≥`_DEFAULT_WARN_THRESHOLD`(=3) 次 warn、≥`_DEFAULT_HARD_LIMIT`(=5) 次 hard stop。工作正常但 lead 每次 retry 微调 command 字符串 → hash 不同 → 不触发。
2. **Layer 2 (tool_freq)**: 按 tool name 计数（不看 args）。当前默认 `_DEFAULT_TOOL_FREQ_WARN = 30`、`_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50`。recursion 上限是 100，100 次 bash 理论上够触发 warn(30) 和 hard_stop(50)，但实际现场 lead 跑满 100 次都没出文字答复 —— **必须先实证根因再下结论**。

#### Step 0: 实证根因（必做，不能跳）

- [ ] **Step 0.1: 在 langgraph.log 搜 LoopDetection 实际行为**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal
grep -n "LOOP DETECTED\|FORCED STOP\|Tool frequency\|loop hard limit\|Repetitive tool calls" packages/agent/logs/langgraph.log 2>/dev/null | tail -40
```

记录观察到的现象：
- (a) 完全没有触发 → 计数器没在跑（thread_id 每次 retry 都新建？after_model 没被调用？）
- (b) 触发了 warn 但没触发 hard_stop → counter 累计但 50 不够
- (c) 触发了 hard_stop 但 lead 继续 → strip tool_calls 没生效或下一 turn 又生成

- [ ] **Step 0.2: 把观察结果写在 commit message 里**

不管 (a)/(b)/(c)，"降阈值到 3/5" 都能改善触发延迟，但根因不同对应的后续修复不同：
- (a) 应该追加 issue 调查 thread_id 生命周期
- (b) 降阈值就够
- (c) 应该追加 issue 调查 `_apply` 的 hard_stop 路径

把观察结果原文（grep 输出片段）和归类（a/b/c）写到 Step 7 commit message 的正文里，**不要丢失证据**。

#### 修复方案

不管 (a)/(b)/(c)，降阈值都能减少 lead 浪费 recursion 的次数，所以这一步可以无条件做。根因调查作为后续 follow-up issue。

- [ ] **Step 1: 写 5 个 failing test**

在 `packages/agent/backend/tests/test_loop_detection_middleware.py` 末尾追加（**复用文件顶部既有的 `_make_runtime` 和 `_make_state` 辅助**；下面只新增类 `TestToolNameFreqWithBash`，不重复定义辅助函数）：

```python
# === Task 1: tool name frequency with lower thresholds ===
# NOTE: 复用文件顶部既有 _make_runtime / _make_state 辅助。


class TestToolNameFreqWithBash:
    """P0 fix: tool name frequency catches repeated bash calls with varying args."""

    def test_bash_tool_freq_warns_at_3_with_different_commands(self):
        """同 bash 3 次不同 command → 触发 warn (hash 不同但 tool name 同)。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_runtime()

        results = []
        for i in range(4):
            tc = {
                "name": "bash",
                "args": {"command": f"cd /tmp && python -m ethoinsight.parse.dump_headers --input file_{i}.txt"},
            }
            warning, hard_stop = mw._track_and_check(_make_state([tc]), runtime)
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
        runtime = _make_runtime()

        for i in range(5):
            tc = {
                "name": "bash",
                "args": {"command": f"python -m ethoinsight.parse.dump_headers --input file_{i}.txt"},
            }
            _ = mw._track_and_check(_make_state([tc]), runtime)

        # 第 5 次: hard stop
        state6 = _make_state([{
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
        """每个 tool name 各调用 2 次（都未到 warn 阈值 3）→ 不触发。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_runtime()

        for _ in range(2):
            _ = mw._track_and_check(
                _make_state([{"name": "bash", "args": {"command": "ls"}}]),
                runtime,
            )
        for _ in range(2):
            _ = mw._track_and_check(
                _make_state([{"name": "read_file", "args": {"path": "/tmp/x"}}]),
                runtime,
            )
        for _ in range(2):
            _ = mw._track_and_check(
                _make_state([{"name": "task", "args": {"description": "x", "prompt": "x"}}]),
                runtime,
            )
        # 每个 tool 各 2 次，都 < warn 阈值 3，不应触发
        # 再调一次别的 tool（合计 3 个 tool 各 2 次 + 1 次 ls_tool = 7 次总，但单个 tool 都不超 3）
        result = mw._track_and_check(
            _make_state([{"name": "ls", "args": {"path": "/tmp"}}]),
            runtime,
        )
        assert result == (None, False)

    def test_counter_does_not_reset_when_other_tool_interleaves(self):
        """同 bash 2 次后切别的 tool 1 次后再 bash 1 次 → bash counter 继续累加不重置。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_runtime()

        # bash x2
        for i in range(2):
            _ = mw._track_and_check(
                _make_state([{"name": "bash", "args": {"command": f"ls {i}"}}]),
                runtime,
            )
        # 切 read_file
        _ = mw._track_and_check(
            _make_state([{"name": "read_file", "args": {"path": "/tmp/x"}}]),
            runtime,
        )
        # 回来 bash → 第 3 次，触发 warn
        warning, hard_stop = mw._track_and_check(
            _make_state([{"name": "bash", "args": {"command": "ls 99"}}]),
            runtime,
        )
        assert warning is not None
        assert "bash" in warning
        assert hard_stop is False

    def test_warn_message_suggests_code_executor(self):
        """warn 注入消息含"task(code-executor)"建议。"""
        mw = LoopDetectionMiddleware(tool_freq_warn=3, tool_freq_hard_limit=5)
        runtime = _make_runtime()

        for i in range(3):
            _ = mw._track_and_check(
                _make_state([{"name": "bash", "args": {"command": f"ls {i}"}}]),
                runtime,
            )

        state = _make_state([{"name": "bash", "args": {"command": "ls 99"}}])
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

在 `loop_detection_middleware.py` 中找到 `_DEFAULT_TOOL_FREQ_WARN` 和 `_DEFAULT_TOOL_FREQ_HARD_LIMIT` 两个常量（模块顶部 "Defaults — can be overridden via constructor" 注释下方），替换为：

```python
# 旧:
_DEFAULT_TOOL_FREQ_WARN = 30  # warn after 30 calls to the same tool type
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50  # force-stop after 50 calls to the same tool type

# 改为:
_DEFAULT_TOOL_FREQ_WARN = 3  # warn after 3 calls to the same tool type (P0 fix: lead 微调 bash command 让 hash 不同绕过 Layer 1)
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 5  # force-stop after 5 calls to the same tool type
```

**注意**：`_DEFAULT_WARN_THRESHOLD` 和 `_DEFAULT_HARD_LIMIT`（hash-based 的 3/5）保持不变。本次只改 tool_freq 那两个常量。

- [ ] **Step 4: 修改 `_TOOL_FREQ_WARNING_MSG`**

在 `loop_detection_middleware.py` 找到 `_TOOL_FREQ_WARNING_MSG = (` 这个赋值，替换为：

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

在 `loop_detection_middleware.py` 找到 `_TOOL_FREQ_HARD_STOP_MSG = ` 这个赋值，替换为：

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

Step 0 实证结果（langgraph.log 观察）：
<把 Step 0.1 的 grep 输出代表性几行贴在这里>
归类：a/b/c（Step 0.2）
EOF
)"
```

**commit 前必须把 `<把 Step 0.1 的 grep 输出代表性几行贴在这里>` 和 `归类：a/b/c` 改成实际内容**，不允许照抄占位符。

---

### Task 2: 新建 `prep_metric_plan` Python 工具

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py:1-13`
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/tools.py`（顶部 `from deerflow.tools.builtins import ...` import 行 + `BUILTIN_TOOLS = [...]` 列表，两处都加 `prep_metric_plan_tool`）
- Create: `packages/agent/backend/tests/test_prep_metric_plan_tool.py`

- [ ] **Step 1: 写 5 个 failing test**

创建 `packages/agent/backend/tests/test_prep_metric_plan_tool.py`：

```python
"""Tests for prep_metric_plan_tool."""

import json
import tempfile
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime

from deerflow.tools.builtins.prep_metric_plan_tool import (
    _ERROR_HINTS,
    prep_metric_plan_tool,
)


def _runtime_with_paths(workspace: Path, uploads: Path) -> ToolRuntime:
    """Build a real ToolRuntime with thread_data state (matches set_experiment_paradigm test style)."""
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


def _runtime_without_workspace() -> ToolRuntime:
    return ToolRuntime(
        state={"thread_data": None},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id="test-id",
        store=None,
    )


def _write_ethovision_file(path: str, columns: list[str]):
    """Write a UTF-16 LE EthoVision trajectory file with full metadata header.

    parse_header 期望: BOM + line-count + metadata kv 段 + column-header 行 + units 行 + data。
    简化的 mock 会让 parse_header 抛 ValueError 因为缺 metadata。下面 header_lines=36 含
    完整 raw_metadata（experiment/trial_name/subject/start_time/duration/arena 等），
    跟 ethoinsight/tests/conftest.py 的 fake EthoVision 文件结构一致。
    """
    header_lines = 36
    lines: list[str] = []
    # Line 1: header line count
    lines.append(f'"Number of header lines:";"{header_lines}"')
    # Lines 2..34: metadata key-value pairs（parse_header 只取 key + value 前两列）
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
    # Pad metadata to header_lines - 2 lines（留出 column-header + units 两行）
    while len(lines) < header_lines - 2:
        lines.append('""')
    # column-header line
    lines.append('"' + '";"'.join(columns) + '"')
    # units line（每列一个 unit 字符串,parse_header 会读但内容不影响 columns 提取）
    lines.append('"' + '";"'.join(["s"] * len(columns)) + '"')
    # 1 data row（parse_trajectory 不会被 prep_metric_plan_tool 调用,这里只是占位防文件意外被读）
    lines.append(";".join(["-1.0"] * len(columns)))
    content = "\r\n".join(lines) + "\r\n"
    # BOM + UTF-16 LE
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")  # UTF-16 LE BOM
        f.write(content.encode("utf-16-le"))


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
    def test_normal_path_with_epm_data(self, tmp_path):
        """正常路径: mock EthoVision EPM 数据 → status=ok, metric_count > 0。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test_epm.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test_epm.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        assert result["plan_summary"]["paradigm"] == "epm"
        assert result["plan_summary"]["metric_count"] > 0
        # plan_path 真实存在
        plan_path = workspace / "metric_plan.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert "metrics" in plan_data


class TestPrepMetricPlanToolErrors:
    def test_workspace_missing(self):
        """thread_data 为 None → error_code=workspace_missing, hint 含 'bug'。"""
        runtime = _runtime_without_workspace()
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/x.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })
        assert result["status"] == "error"
        assert result["error_code"] == "workspace_missing"
        assert "bug" in result["hint"].lower()

    def test_file_not_found(self, tmp_path):
        """传不存在的路径 → error_code=file_not_found, hint 含 ask_clarification。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/nonexistent.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "file_not_found"
        assert "ask_clarification" in result["hint"].lower()

    def test_unknown_paradigm(self, tmp_path):
        """传 paradigm='invalid' → error_code=unknown_paradigm。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test.txt",
            "paradigm": "invalid_paradigm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        assert result["error_code"] == "unknown_paradigm"

    def test_columns_missing(self, tmp_path):
        """mock 数据缺 in_zone_open_arms_* 列, paradigm=epm → error_code=columns_missing/empty_plan, hint 含'录制设置'或'指标'。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
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

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/minimal.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "error"
        # ResolveError 在这种情况会抛 columns_missing 或 empty_plan，两个都是合法的兜底
        assert result["error_code"] in {"columns_missing", "empty_plan"}

    def test_plan_file_written_on_success(self, tmp_path):
        """status=ok 后 plan_path 真实存在 + JSON 可读。"""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        uploads = tmp_path / "uploads"
        uploads.mkdir()

        data_file = uploads / "test2.txt"
        _write_ethovision_file(str(data_file), EPM_COLUMNS)

        runtime = _runtime_with_paths(workspace, uploads)
        result = prep_metric_plan_tool.invoke({
            "uploaded_file": "/mnt/user-data/uploads/test2.txt",
            "paradigm": "epm",
            "runtime": runtime,
        })

        assert result["status"] == "ok"
        plan_path = workspace / "metric_plan.json"
        assert plan_path.exists()
        plan_data = json.loads(plan_path.read_text())
        assert isinstance(plan_data, dict)
        assert "metrics" in plan_data
        assert len(plan_data["metrics"]) == result["plan_summary"]["metric_count"]
```

**测试调用风格说明**：用 `prep_metric_plan_tool.invoke({"runtime": runtime, ...args})` 通过 langchain 的 tool wrapper 调用 —— 这是项目里 `test_set_experiment_paradigm_ev19.py` 已验证的惯例。不要用 `prep_metric_plan_tool(runtime=..., ...)` 直接调用 `@tool` 装饰对象。

**`test_columns_missing` 的兜底**：ResolveError 在缺列场景可能抛 `columns_missing`（必需列缺）或 `empty_plan`（剪光所有 default），用 `in {"columns_missing", "empty_plan"}` 容忍两种合法返回。

- [ ] **Step 2: 运行新测试验证它们 fail**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py -v
```

预期：6 个测试 FAIL（`prep_metric_plan_tool` 不存在）。

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
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.sandbox.tools import replace_virtual_path
from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve
from ethoinsight.parse._core import detect_ethovision, parse_header

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
    "workspace_missing": (
        "thread_data.workspace_path 未设置——这是基础设施 bug（ThreadDataMiddleware 应该先建好 workspace）。"
        "present_files 把错误信息呈现给用户，让他报 bug。"
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
                       "unknown_paradigm"|"columns_missing"|"schema_violation"|
                       "empty_plan"|"unknown_metric"|"workspace_missing",
         "message": str,
         "hint": str}
    """
    # Step 1: resolve thread_data — workspace_path is mandatory, fail fast if missing
    thread_data = runtime.state.get("thread_data") if runtime.state else None
    if not thread_data or not thread_data.get("workspace_path"):
        return _error_result(
            "workspace_missing",
            "thread_data.workspace_path is not set",
        )
    real_workspace_path = thread_data["workspace_path"]
    real_file_path = replace_virtual_path(uploaded_file, thread_data)

    # Step 2: check file exists
    if not Path(real_file_path).exists():
        return _error_result(
            "file_not_found",
            f"File not found: {uploaded_file} (resolved to {real_file_path})",
        )

    # Step 3: detect EthoVision format
    if not detect_ethovision(real_file_path):
        return _error_result(
            "format_unrecognized",
            f"File {uploaded_file} is not an EthoVision XT export.",
        )

    # Step 4: parse header to get column names
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

    # Step 5: resolve catalog → Plan
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

    # Step 6: serialize plan to workspace/metric_plan.json
    plan_dict = plan_to_dict(plan)
    plan_path = Path(real_workspace_path) / "metric_plan.json"
    try:
        plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        return _error_result(
            "parse_failed",
            f"Failed to write metric_plan.json: {e}",
        )

    # Step 7: build summary (只 paradigm/metric_count/metric_ids，不含完整 plan)
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

修改 `packages/agent/backend/packages/harness/deerflow/tools/tools.py`。**现状（不要省略任何条目）**：

```python
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    set_experiment_paradigm_tool,
]
```

**改为**（追加 `prep_metric_plan_tool`，保留 `set_experiment_paradigm_tool` —— 它是 EV19 范式锁定 Gate 1 的核心，参见 CLAUDE.md 第 10 条；删它会破坏 Gate 1 流程）：

```python
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    set_experiment_paradigm_tool,
    prep_metric_plan_tool,
]
```

同时把文件顶部 `from deerflow.tools.builtins import ...` 这一行的 import 改为：

```python
# 旧:
from deerflow.tools.builtins import ask_clarification_tool, present_file_tool, task_tool, view_image_tool

# 改为（按字母序追加 prep_metric_plan_tool）:
from deerflow.tools.builtins import ask_clarification_tool, prep_metric_plan_tool, present_file_tool, task_tool, view_image_tool
```

- [ ] **Step 6: 运行新工具的全部测试验证 pass**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py -v
```

预期：6 tests PASS。

- [ ] **Step 7: 运行全量测试确保无回归**

```bash
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -5
```

预期：pass 数比 baseline +6（新测试），fail 数不变（3 pre-existing）。

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

**核心思路**：把过滤逻辑抽成模块级纯函数 `_filter_lead_tools(tools, excluded) -> list`，单测打到纯函数；`make_lead_agent` 只调用它。**不要在测试里 patch 整个 `make_lead_agent`** —— 它依赖太多东西，patch 链路脆弱。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（在 `make_lead_agent` 之前定义 `_LEAD_EXCLUDED_TOOLS` 常量 + `_filter_lead_tools` 纯函数；在 `make_lead_agent` 内调用它）
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`（**只读核对**：确认 code-executor 的 tools 列表含 bash，不修改）
- Create: `packages/agent/backend/tests/test_lead_tool_filtering.py`

- [ ] **Step 1: 写 failing test（针对纯函数 + 真实 get_available_tools 返回的过滤后集合）**

创建 `packages/agent/backend/tests/test_lead_tool_filtering.py`：

```python
"""Tests for lead agent tool filtering (Task 3: remove bash/write_file/str_replace)."""

from langchain.tools import BaseTool
from langchain_core.tools import tool as tool_decorator

from deerflow.agents.lead_agent.agent import _LEAD_EXCLUDED_TOOLS, _filter_lead_tools


def _make_named_tool(name: str) -> BaseTool:
    """Build a minimal BaseTool with a given .name attribute."""
    @tool_decorator(name, parse_docstring=False)
    def fn(x: str) -> str:
        """noop."""
        return x
    return fn


class TestFilterLeadToolsPureFunction:
    """纯函数测试：不 patch agent 工厂，直接打 _filter_lead_tools。"""

    def test_excludes_bash(self):
        tools = [_make_named_tool("bash"), _make_named_tool("read_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        names = {t.name for t in result}
        assert "bash" not in names
        assert "read_file" in names

    def test_excludes_write_file(self):
        tools = [_make_named_tool("write_file"), _make_named_tool("read_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "write_file" not in {t.name for t in result}

    def test_excludes_str_replace(self):
        tools = [_make_named_tool("str_replace"), _make_named_tool("ls")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "str_replace" not in {t.name for t in result}

    def test_keeps_ls(self):
        """Q4 决策：lead 保留 ls 验证 code-executor 产物。"""
        tools = [_make_named_tool("ls"), _make_named_tool("bash")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "ls" in {t.name for t in result}

    def test_keeps_read_file(self):
        """lead 需要 read_file 看 handoff JSON。"""
        tools = [_make_named_tool("read_file"), _make_named_tool("write_file")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert "read_file" in {t.name for t in result}

    def test_keeps_prep_metric_plan(self):
        """关键回归：prep_metric_plan 是 lead 替代 bash 调 parse/catalog 的唯一通道，绝不能被误加进 _LEAD_EXCLUDED_TOOLS。"""
        tools = [
            _make_named_tool("prep_metric_plan"),
            _make_named_tool("bash"),
            _make_named_tool("write_file"),
        ]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        names = {t.name for t in result}
        assert "prep_metric_plan" in names
        # 同时确认 bash / write_file 被过滤(防止把这条测试退化成空断言)
        assert "bash" not in names
        assert "write_file" not in names

    def test_excluded_set_is_frozen(self):
        """_LEAD_EXCLUDED_TOOLS 必须含三项，不多不少。"""
        assert _LEAD_EXCLUDED_TOOLS == frozenset({"bash", "write_file", "str_replace"})

    def test_empty_tools_returns_empty(self):
        assert _filter_lead_tools([], _LEAD_EXCLUDED_TOOLS) == []

    def test_no_excluded_tools_returns_all(self):
        tools = [_make_named_tool("read_file"), _make_named_tool("task")]
        result = _filter_lead_tools(tools, _LEAD_EXCLUDED_TOOLS)
        assert {t.name for t in result} == {"read_file", "task"}


class TestSubagentToolsUnchanged:
    """子代理（code-executor / data-analyst）工具列表不受影响 —— 子代理通过 SubagentConfig.tools 显式声明 bash，跟 _filter_lead_tools 完全独立。"""

    def test_code_executor_still_has_bash(self):
        """grep subagents/builtins/__init__.py 验证 code-executor 注册时 tools 含 bash。"""
        from deerflow.subagents.registry import get_subagent_config
        config = get_subagent_config("code-executor")
        assert config is not None, "code-executor subagent 必须注册"
        # tools=None 表示"全部工具"，即 bash 可用；
        # tools 是显式列表时，bash 必须在内；
        # disallowed_tools 不能拒绝 bash
        if config.tools is not None:
            assert "bash" in config.tools, (
                f"code-executor 必须能用 bash；当前 tools={config.tools}"
            )
        if config.disallowed_tools is not None:
            assert "bash" not in config.disallowed_tools
```

- [ ] **Step 2: 运行新测试验证它们 fail**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_lead_tool_filtering.py -v
```

预期：所有 `TestFilterLeadToolsPureFunction.*` 测试 FAIL（ImportError: `_filter_lead_tools` 不存在）。`TestSubagentToolsUnchanged::test_code_executor_still_has_bash` 应该已经 PASS（因为我们没改 subagent）。

- [ ] **Step 3: 在 `agent.py` 中定义 `_LEAD_EXCLUDED_TOOLS` + `_filter_lead_tools`**

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 找到 `def make_lead_agent(` 之前的位置（imports 段之后、函数定义之前），添加：

```python
# Lead agent 不该有 bash/write_file/str_replace —— 所有 ethoinsight CLI
# 调用走 prep_metric_plan 工具，所有写文件操作走 code-executor 子代理。
# 这是 P0 修复：lead 无 bash → 无 quoting retry → 无 recursion 100 耗尽。
# (subagent 通过 SubagentConfig.tools 显式声明 bash，不受此过滤影响)
_LEAD_EXCLUDED_TOOLS: frozenset[str] = frozenset({"bash", "write_file", "str_replace"})


def _filter_lead_tools(tools: list, excluded: frozenset[str]) -> list:
    """Drop tools whose .name is in excluded set. Pure function — single source of truth for the lead exclusion policy."""
    return [t for t in tools if t.name not in excluded]
```

- [ ] **Step 4: 在 `make_lead_agent` 内调用过滤函数**

找到 `make_lead_agent` 末尾的 `return create_agent(` 块。**当前**：

```python
return create_agent(
    model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort),
    tools=get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled),
    middleware=_build_middlewares(config, model_name=model_name, agent_name=agent_name),
    system_prompt=apply_prompt_template(
        subagent_enabled=subagent_enabled, max_concurrent_subagents=max_concurrent_subagents, agent_name=agent_name, available_skills=set(agent_config.skills) if agent_config and agent_config.skills is not None else None
    ),
    state_schema=ThreadState,
)
```

**改为**（在 `return create_agent(` 之前插入两行，把 tools 参数换成 filtered_lead_tools）：

```python
all_lead_tools = get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled)
filtered_lead_tools = _filter_lead_tools(all_lead_tools, _LEAD_EXCLUDED_TOOLS)
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
    system_prompt=apply_prompt_template(
        subagent_enabled=subagent_enabled, max_concurrent_subagents=max_concurrent_subagents, agent_name=agent_name, available_skills=set(agent_config.skills) if agent_config and agent_config.skills is not None else None
    ),
    state_schema=ThreadState,
)
```

**注意**：不修改 bootstrap agent 路径（同一个 `make_lead_agent` 函数前段有 bootstrap 分支，含独立的 `return create_agent(...)`）—— bootstrap 是特殊流程，保留其工具列表不变。修改时检查上下文，确保改的是文件末尾"normal lead"路径那一处 `return`，不是 bootstrap 那处。

- [ ] **Step 5: 运行 Task 3 测试验证 pass**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_lead_tool_filtering.py -v
```

预期：9 tests PASS。

- [ ] **Step 6: 运行全量测试确保无回归**

```bash
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -5
```

预期：pass 数比 baseline +20（Task 1 5个 + Task 2 6个 + Task 3 9个，总共新增 20 个测试），fail 数不变（3 pre-existing）。

- [ ] **Step 7: Commit**

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
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（删除 `LeadAgentExecutionBoundaryProvider` 的 import + 注册段；注册段在 `_build_middlewares` 函数内、`Ev19TemplateGuardrailProvider` 注册块的紧后方）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（两处：(a) "transparency 表"段含 `跑 \`python -m ethoinsight.parse.dump_headers\`` 和 `跑 \`python -m ethoinsight.catalog.resolve\`` 那两行；(b) `### Step 0.5: 生成 metric_plan.json` 段及紧邻的 `ethoinsight-metric-catalog` skill 说明段）
- Modify: `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`（从 `### lead` heading 起到下一个 `### ` heading（应为 `### data-analyst`）之前的整段 lead role 说明）
- Create: `docs/handoffs/2026-05/2026-05-15-p0-fix-dogfood-validation.md`

#### 子任务 4a: 删除 G4 boundary 文件

- [ ] **Step 4a.1: 用 `git rm` 删除两个文件（不要先 rm 再 git rm）**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/p0-lead-bash-removal
git rm packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py
git rm packages/agent/backend/tests/test_lead_execution_boundary_provider.py
```

`git rm` 已经同时从工作树和暂存区移除文件，**不需要先 `rm`**。

- [ ] **Step 4a.2: 从 `agent.py` 移除 G4 boundary 的 import + 注册**

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` 中搜 `LeadAgentExecutionBoundaryProvider` 定位（应该在 `_build_middlewares` 函数内、`if guardrails_cfg.enabled:` 分支中、紧跟在 `Ev19TemplateGuardrailProvider` 注册块后面）：

```python
# 删除以下整段:
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


# 删除后，该位置仅剩 Ev19TemplateGuardrail 的注册块。
```

注意：保留 `Ev19TemplateGuardrailProvider` 的注册块，只删 `LeadAgentExecutionBoundaryProvider` 部分。

- [ ] **Step 4a.3: 运行测试确认删除不破坏任何东西**

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/ -k "not test_lead_execution_boundary" -v 2>&1 | tail -10
# 确保没有 ImportError
make test 2>&1 | tail -5
```

- [ ] **Step 4a.4: Commit 子任务 4a**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git commit -m "$(cat <<'EOF'
feat: 删除 G4 LeadAgentExecutionBoundaryProvider (bash 已从 lead tool 列表移除)
EOF
)"
```

`git rm` 已在 Step 4a.1 把删除标记进了 index，本次 commit 一并提交。

#### 子任务 4b: 改 lead prompt

- [ ] **Step 4b.1: 更新 transparency 表中两条 bash 行**

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 中搜 `跑 \`python -m ethoinsight.parse.dump_headers\``，定位到 transparency 表里那两行：

```python
# 旧（两行）:
| 跑 `python -m ethoinsight.parse.dump_headers` | "📂 正在解析 EthoVision 文件结构..." |
| 跑 `python -m ethoinsight.catalog.resolve` | "📋 正在生成指标计划..." |

# 改为（合并成一行）:
| 调 `prep_metric_plan` | "📋 正在生成指标计划..." |
```

即删掉 dump_headers 那一行，把 catalog.resolve 那一行替换为 prep_metric_plan 行。

- [ ] **Step 4b.2: 更新 Step 0.5 段**

在同一个 `prompt.py` 中搜 `### Step 0.5: 生成 metric_plan.json` 定位段落，整段替换：

```python
# 旧:
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

- [ ] **Step 4b.3: 更新 skill 说明段**

在同一个 `prompt.py` 中搜 `- **ethoinsight-metric-catalog**:` 定位到含 `bash dump_headers` 那一行：

```python
# 旧:
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

读取 `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`，从 `### lead` heading 开始、到下一个同级 heading（应为 `### data-analyst`）之前的整段 lead role 说明，替换为：

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
1. 上传 EPM 数据文件 —— **真实数据在 `/home/wangqiuyang/DemoData/newdemodata/`**（不是仓库内的 `demo-data/`，CLAUDE.md 描述与实际位置不符）。挑一个 EthoVision XT EPM `.txt` 轨迹文件上传。如该目录下没有明确 EPM 范式的样本，先 `ls /home/wangqiuyang/DemoData/newdemodata/` 查清楚有什么，然后选最接近 EPM 的（如 Plus-Maze / Elevated-Plus）；选不出时停下问用户哪个文件可用，**不要瞎挑**。
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
- 数据: `/home/wangqiuyang/DemoData/newdemodata/<具体文件名>`（EthoVision XT EPM 轨迹文件）
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

预期：baseline 2315 + 20 新增 = ~2335 passed，3 pre-existing failed（来自 test_client_live.py 或 test_gateway.py 的预存失败）。

- [ ] **Step F.2: 检查 commit 历史**

```bash
git log --oneline -10
# 应看到 7 个 commit (按子任务粒度):
#   Task 1: fix: LoopDetectionMiddleware ...
#   Task 2: feat(tools): 加 prep_metric_plan ...
#   Task 3: feat(lead): 移除 bash / write_file ...
#   Task 4a: feat: 删除 G4 LeadAgentExecutionBoundaryProvider ...
#   Task 4b: feat(lead): prompt Step 0.5 改 prep_metric_plan ...
#   Task 4c: feat(skill): ethoinsight-metric-catalog ...
#   Task 4d: docs: P0 fix dogfood 验证报告
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
