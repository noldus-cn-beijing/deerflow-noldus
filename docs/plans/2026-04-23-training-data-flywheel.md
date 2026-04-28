# 训练数据飞轮实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让行为学专家使用 Claude-Sonnet 版 agent 做真实分析时，自动把每次交互沉淀为 Fireworks SFT/DPO 格式的训练样本，同时通过前端三按钮收集专家质量反馈，两个月内攒够 Phase 1 微调所需的 ~500 条真实样本 + ~300 对 DPO 数据。

**Architecture:** 四层组合。(1) 后端新增 `TrainingDataMiddleware` 挂在 lead agent middleware 链尾部，`after_agent` 钩子抽取 (lead, subagent, analyst, writer) 四类 (input, thinking, output) 样本落到 `.deer-flow/training-data/auto-collected/<thread_id>.jsonl`。(2) Gateway 新增 `/api/threads/{id}/feedback` 路由接收专家 ✅/⚠️/❌ 反馈，写入 `.deer-flow/training-data/feedback/<thread_id>.jsonl`。(3) 前端在 `subtask-card.tsx` 和 assistant message 下方嵌入三按钮；⚠️/❌ 弹出内联编辑框。(4) 后处理脚本 `scripts/extract_e2e_sessions.py` 把原始录制 + 反馈 join 成训练集，并输出日报统计。

**Tech Stack:** Python 3.12 / LangChain AgentMiddleware / FastAPI / pytest / Next.js + TypeScript / TailwindCSS。后端用 `langchain.agents.middleware.AgentMiddleware` 基类，参照 `MemoryMiddleware`；前端 hook 调用 `api-client.ts` 的统一客户端；数据格式是 Fireworks ChatML with `<think>` traces。

**Non-goals（明确不做）:**
- DPO 训练本身（Phase 1 结束后才做，飞轮只负责攒数据）
- 合成数据生成脚本（A/B/C/D 来源，另一个计划）
- Golden-case 标注 runner（已有独立计划）
- 用户权限/多租户（v0.1 内部试用，单用户就够）
- 数据隐私合规审查（按 [handoffs/2026-04-23](../handoffs/2026-04-23-m01-remaining-items-done.md) 项目已承诺"数据只在本地"，本计划继承该承诺）

**参考文档:**
- [finetuning-data-checklist.md §E + §G + §H](2026-04-15-fine-tuning-data-checklist.md) — 数据飞轮的原始设计
- [finetuning-strategy-update.md §2.1](2026-04-21-finetuning-strategy-update.md) — 蒸馏 Sonnet 作为教师的决策
- [behavioral-reasoning-design.md §6](2026-04-21-behavioral-reasoning-design.md) — Quality-reviewer 与飞轮数据的关系
- [DeerFlow backend CLAUDE.md](../../packages/agent/backend/CLAUDE.md) — middleware 链、ThreadState、路径系统

---

## 实施前须知（每个 Task 都要看）

**TDD 强制**：每个 Task 都按 write test → run fail → implement → run pass → commit 的节奏。`backend/CLAUDE.md` 明确"每个新功能/bug 修复都必须带单测，无例外"。

**Commit 节奏**：每个 Task 的最后一步都是 commit。Commit message 用中文，简洁描述意图（项目规范）。

**关键命令**：
```bash
# 跑单个测试文件
cd packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_<name>.py -v

# 跑全量后端测试
cd packages/agent/backend
source .venv/bin/activate && make test

# Lint
cd packages/agent/backend
source .venv/bin/activate && make lint
```

**路径常识**：
- Thread 数据目录：`packages/agent/backend/.deer-flow/threads/<thread_id>/`
- 本计划新增：`packages/agent/backend/.deer-flow/training-data/auto-collected/` 和 `feedback/`
- Middleware 在：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/`
- Gateway 路由在：`packages/agent/backend/app/gateway/routers/`
- 前端：`packages/agent/frontend/src/`

**受保护文件告警**（改了这些，下次同步 DeerFlow 上游会标冲突，计划里涉及两个）：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` — 需要注册新 middleware
- `packages/agent/backend/app/gateway/app.py` — 需要注册新 router

---

## Phase A: 后端录制中间件（Task 1-5）

### Task 1: TrainingDataMiddleware 骨架 + 目录创建

**目标**：创建空的 middleware 类，在 `before_agent` 阶段确定输出目录路径，返回 state；还不录任何数据。

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
- Create: `packages/agent/backend/tests/test_training_data_middleware.py`

**Step 1: Write the failing test**

Write to `packages/agent/backend/tests/test_training_data_middleware.py`:

```python
import pytest
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware


class TestTrainingDataMiddlewareInit:
    def test_before_agent_computes_output_dir_from_thread_id(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))

        result = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-abc"}),
        )

        assert result is None or result == {}
        expected_dir = tmp_path / "training-data" / "auto-collected"
        assert expected_dir.exists()

    def test_before_agent_skips_when_no_thread_id(self, tmp_path, monkeypatch):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        monkeypatch.setattr(
            "deerflow.agents.middlewares.training_data_middleware.get_config",
            lambda: {"configurable": {}},
        )

        result = middleware.before_agent(state={}, runtime=Runtime(context=None))

        assert result is None
        # Should not create the dir when thread_id cannot be resolved
        expected_dir = tmp_path / "training-data" / "auto-collected"
        assert not expected_dir.exists()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deerflow.agents.middlewares.training_data_middleware'`.

**Step 3: Write minimal implementation**

Write to `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`:

```python
"""Middleware that records every agent turn as a training data sample.

Written as part of the training-data flywheel (docs/plans/2026-04-23-training-data-flywheel.md).
Records Fireworks-compatible JSONL per thread to
`.deer-flow/training-data/auto-collected/<thread_id>.jsonl`.
"""
import logging
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class TrainingDataMiddlewareState(AgentState):
    training_data_path: NotRequired[str | None]


class TrainingDataMiddleware(AgentMiddleware[TrainingDataMiddlewareState]):
    """Record each agent conversation as a training sample JSONL."""

    state_schema = TrainingDataMiddlewareState

    def __init__(self, base_dir: str | None = None):
        super().__init__()
        self._base_dir = Path(base_dir) if base_dir else None

    def _resolve_thread_id(self, runtime: Runtime) -> str | None:
        ctx = runtime.context or {}
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        if thread_id:
            return thread_id
        try:
            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id")
        except Exception:
            return None

    def _resolve_base_dir(self) -> Path:
        if self._base_dir:
            return self._base_dir
        # Default to backend/.deer-flow/
        from deerflow.config.paths import get_paths

        return Path(get_paths().base_dir)

    @override
    def before_agent(self, state, runtime: Runtime) -> dict | None:
        thread_id = self._resolve_thread_id(runtime)
        if not thread_id:
            logger.debug("TrainingDataMiddleware: no thread_id, skipping")
            return None

        out_dir = self._resolve_base_dir() / "training-data" / "auto-collected"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"training_data_path": str(out_dir / f"{thread_id}.jsonl")}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py \
       packages/agent/backend/tests/test_training_data_middleware.py
git commit -m "feat(training): 新增 TrainingDataMiddleware 骨架"
```

---

### Task 2: 抽取 lead agent 的 (user_input, AI text) 样本

**目标**：`after_agent` 钩子里遍历 state.messages，抽 HumanMessage → 下一条 AIMessage 的 (input, output) 对，写入 JSONL。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
- Modify: `packages/agent/backend/tests/test_training_data_middleware.py`

**Step 1: Write the failing test**

Append to `test_training_data_middleware.py`:

```python
import json
from langchain_core.messages import AIMessage, HumanMessage


class TestTrainingDataMiddlewareRecording:
    def test_after_agent_writes_lead_sample(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        state_before = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-xyz"}),
        )
        state = {
            "training_data_path": state_before["training_data_path"],
            "messages": [
                HumanMessage(content="分析这份斑马鱼数据"),
                AIMessage(content="好的，我先解析轨迹文件。"),
            ],
        }

        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-xyz"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-xyz.jsonl"
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        lead_samples = [l for l in lines if l["role"] == "lead"]
        assert len(lead_samples) == 1
        assert lead_samples[0]["input"] == "分析这份斑马鱼数据"
        assert lead_samples[0]["output"] == "好的，我先解析轨迹文件。"
        assert lead_samples[0]["thread_id"] == "thread-xyz"

    def test_after_agent_skips_when_no_messages(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-empty"}),
        )
        middleware.after_agent(
            state={"training_data_path": None, "messages": []},
            runtime=Runtime(context={"thread_id": "thread-empty"}),
        )
        out = tmp_path / "training-data" / "auto-collected" / "thread-empty.jsonl"
        assert not out.exists() or out.read_text().strip() == ""
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py::TestTrainingDataMiddlewareRecording -v`
Expected: FAIL — `AttributeError: 'TrainingDataMiddleware' object has no attribute 'after_agent'` or tests fail on missing file.

**Step 3: Write minimal implementation**

In `training_data_middleware.py`, add imports and method:

```python
import json
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage
```

Add method to the class:

```python
def _append_jsonl(self, path: Path, sample: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

def _extract_lead_samples(self, messages: list, thread_id: str) -> list[dict]:
    """Pair each HumanMessage with the next AIMessage text reply."""
    samples: list[dict] = []
    pending_human: HumanMessage | None = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pending_human = msg
        elif isinstance(msg, AIMessage) and pending_human is not None:
            text = msg.content if isinstance(msg.content, str) else ""
            if text.strip():
                samples.append({
                    "role": "lead",
                    "thread_id": thread_id,
                    "input": pending_human.content if isinstance(pending_human.content, str) else str(pending_human.content),
                    "output": text,
                    "thinking": (msg.additional_kwargs or {}).get("reasoning_content") or "",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                pending_human = None
    return samples

@override
def after_agent(self, state, runtime: Runtime) -> dict | None:
    thread_id = self._resolve_thread_id(runtime)
    path_str = state.get("training_data_path") if isinstance(state, dict) else None
    if not thread_id or not path_str:
        return None
    messages = state.get("messages", []) if isinstance(state, dict) else []
    if not messages:
        return None
    samples = self._extract_lead_samples(messages, thread_id)
    if not samples:
        return None
    path = Path(path_str)
    for s in samples:
        self._append_jsonl(path, s)
    logger.info("TrainingDataMiddleware: wrote %d lead samples to %s", len(samples), path)
    return None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py \
       packages/agent/backend/tests/test_training_data_middleware.py
git commit -m "feat(training): 录制 lead agent 输入输出样本"
```

---

### Task 3: 抽取 subagent (task tool) 样本

**目标**：扫描 AIMessage 的 tool_calls 与对应的 ToolMessage，抽 (task_description, execution_result) 样本。Subagent 样本是 tool-calling 训练最关键的数据。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
- Modify: `packages/agent/backend/tests/test_training_data_middleware.py`

**Step 1: Write the failing test**

Append:

```python
from langchain_core.messages import ToolMessage


class TestSubagentSampleExtraction:
    def test_after_agent_writes_subagent_sample(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-sub"}),
        )
        ai_with_task = AIMessage(
            content="我需要 code-executor",
            tool_calls=[{
                "id": "call_1",
                "name": "task",
                "args": {
                    "description": "analyze shoaling",
                    "prompt": "Run ethoinsight-analysis on uploads",
                    "subagent_type": "code-executor",
                },
            }],
        )
        tool_result = ToolMessage(
            content="Analysis complete: 4 metrics computed",
            tool_call_id="call_1",
        )
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="分析"), ai_with_task, tool_result],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-sub"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-sub.jsonl"
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        subagent = [l for l in lines if l["role"] == "subagent"]
        assert len(subagent) == 1
        assert subagent[0]["subagent_type"] == "code-executor"
        assert "analyze shoaling" in subagent[0]["input"]
        assert "Analysis complete" in subagent[0]["output"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py::TestSubagentSampleExtraction -v`
Expected: FAIL — `assert len(subagent) == 1` but `len == 0`.

**Step 3: Write minimal implementation**

Add helper to the class:

```python
def _extract_subagent_samples(self, messages: list, thread_id: str) -> list[dict]:
    """For each AIMessage.tool_call of task tool, pair with its ToolMessage."""
    tool_results: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tool_results[msg.tool_call_id] = msg

    samples: list[dict] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in (msg.tool_calls or []):
            if call.get("name") != "task":
                continue
            call_id = call.get("id")
            if not call_id or call_id not in tool_results:
                continue
            args = call.get("args") or {}
            result = tool_results[call_id]
            result_text = result.content if isinstance(result.content, str) else str(result.content)
            samples.append({
                "role": "subagent",
                "thread_id": thread_id,
                "subagent_type": args.get("subagent_type", ""),
                "input": json.dumps({
                    "description": args.get("description", ""),
                    "prompt": args.get("prompt", ""),
                }, ensure_ascii=False),
                "output": result_text,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
    return samples
```

Update `after_agent`:

```python
samples = self._extract_lead_samples(messages, thread_id)
samples.extend(self._extract_subagent_samples(messages, thread_id))
if not samples:
    return None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py -v`
Expected: 5 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py \
       packages/agent/backend/tests/test_training_data_middleware.py
git commit -m "feat(training): 录制 subagent tool calling 样本"
```

---

### Task 4: 质量过滤 — 过滤失败/超时/空内容样本

**目标**：按 [checklist §E2](2026-04-15-fine-tuning-data-checklist.md) 要求自动过滤明显低质量样本：空输出、包含 `"error":` 或 `"timed_out"` 的 tool 结果、HTTP 429 消息。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
- Modify: `packages/agent/backend/tests/test_training_data_middleware.py`

**Step 1: Write the failing test**

```python
class TestQualityFilter:
    def test_filters_out_error_tool_messages(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(state={}, runtime=Runtime(context={"thread_id": "t-err"}))
        ai_with_task = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "task", "args": {"subagent_type": "code-executor", "description": "x", "prompt": "y"}}],
        )
        bad_result = ToolMessage(content='{"error": "subagent timed_out"}', tool_call_id="c1")
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="hi"), ai_with_task, bad_result],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-err"}))

        out = tmp_path / "training-data" / "auto-collected" / "t-err.jsonl"
        content = out.read_text() if out.exists() else ""
        assert "code-executor" not in content  # the bad subagent sample was filtered
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py::TestQualityFilter -v`
Expected: FAIL — the bad sample is still written.

**Step 3: Write minimal implementation**

Add module-level constants and filter helper:

```python
_BAD_OUTPUT_MARKERS = ('"error":', '"timed_out"', 'HTTP 429', 'rate_limit_exceeded')

def _is_low_quality(output: str) -> bool:
    if not output or not output.strip():
        return True
    return any(marker in output for marker in _BAD_OUTPUT_MARKERS)
```

Apply filter in both extractors:

```python
# in _extract_lead_samples, before append:
if _is_low_quality(text):
    pending_human = None
    continue

# in _extract_subagent_samples, before append:
if _is_low_quality(result_text):
    continue
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py -v`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py \
       packages/agent/backend/tests/test_training_data_middleware.py
git commit -m "feat(training): 过滤低质量样本（错误/超时/空输出）"
```

---

### Task 5: 在 lead agent 中注册 middleware

**目标**：在 `agent.py` 的 `_build_middlewares()` 里追加 `TrainingDataMiddleware()`。必须在 `ClarificationMiddleware` 之前，在 `MemoryMiddleware` 之后（和 memory 并列都是 after_agent）。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- Create: `packages/agent/backend/tests/test_lead_agent_training_middleware.py`

⚠️ **受保护文件提醒**：这是 DeerFlow fork 的受保护文件，修改后下次上游同步要标记人工判断。改动只 1 行，影响可控。

**Step 1: Write the failing test**

```python
from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware


def test_lead_agent_includes_training_data_middleware(monkeypatch):
    from deerflow.agents.lead_agent.agent import _build_middlewares

    middlewares = _build_middlewares(
        config={"configurable": {"subagent_enabled": False, "is_plan_mode": False}},
        model_name=None,
    )
    types = [type(m).__name__ for m in middlewares]
    assert "TrainingDataMiddleware" in types
    # Must come before ClarificationMiddleware (which must be last)
    assert types.index("TrainingDataMiddleware") < types.index("ClarificationMiddleware")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_lead_agent_training_middleware.py -v`
Expected: FAIL — `TrainingDataMiddleware` not in types list.

**Step 3: Write minimal implementation**

In `agent.py`, add import (alphabetically near other middleware imports):

```python
from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware
```

Add registration. In `_build_middlewares`, right after the `MemoryMiddleware` line (current line ~247):

```python
# Add TrainingDataMiddleware (records every turn for SFT/DPO dataset)
middlewares.append(TrainingDataMiddleware())
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_lead_agent_training_middleware.py -v`
Expected: 1 passed.

Also run full middleware tests to confirm nothing else broke:

```bash
PYTHONPATH=. uv run pytest tests/ -v -k "middleware"
```
Expected: all pass.

**Step 5: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
       packages/agent/backend/tests/test_lead_agent_training_middleware.py
git commit -m "feat(training): 在 lead agent 注册 TrainingDataMiddleware"
```

---

## Phase B: Gateway 反馈 API（Task 6-8）

### Task 6: Feedback router 骨架 + POST 路由

**目标**：新增 `/api/threads/{thread_id}/feedback` 接收 (message_id, verdict ∈ {correct, needs_fix, wrong}, revised_text?, note?) 并落到 `feedback/<thread_id>.jsonl`。

**Files:**
- Create: `packages/agent/backend/app/gateway/routers/feedback.py`
- Create: `packages/agent/backend/tests/test_feedback_router.py`

**Step 1: Write the failing test**

```python
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.gateway.routers import feedback as feedback_mod

    monkeypatch.setattr(feedback_mod, "_base_dir", lambda: tmp_path)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(feedback_mod.router)
    return TestClient(app), tmp_path


def test_post_feedback_correct_verdict(client):
    c, tmp_path = client
    resp = c.post(
        "/api/threads/thread-42/feedback",
        json={"message_id": "m-1", "verdict": "correct"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True}

    out = tmp_path / "training-data" / "feedback" / "thread-42.jsonl"
    record = json.loads(out.read_text().strip().splitlines()[0])
    assert record["message_id"] == "m-1"
    assert record["verdict"] == "correct"


def test_post_feedback_needs_fix_with_revision(client):
    c, tmp_path = client
    resp = c.post(
        "/api/threads/t/feedback",
        json={
            "message_id": "m-2",
            "verdict": "needs_fix",
            "revised_text": "正确的版本",
            "note": "第二段指标解读方向反了",
        },
    )
    assert resp.status_code == 200
    record = json.loads((tmp_path / "training-data" / "feedback" / "t.jsonl").read_text().strip())
    assert record["revised_text"] == "正确的版本"
    assert record["note"] == "第二段指标解读方向反了"


def test_post_feedback_rejects_invalid_verdict(client):
    c, _ = client
    resp = c.post(
        "/api/threads/t/feedback",
        json={"message_id": "m-1", "verdict": "maybe"},
    )
    assert resp.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.gateway.routers.feedback'`.

**Step 3: Write minimal implementation**

Create `app/gateway/routers/feedback.py`:

```python
"""Feedback router for training-data flywheel.

Accepts expert ✅/⚠️/❌ verdicts on assistant messages and appends each
feedback to `.deer-flow/training-data/feedback/<thread_id>.jsonl`.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/threads", tags=["feedback"])


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    verdict: Literal["correct", "needs_fix", "wrong"]
    revised_text: str | None = None
    note: str | None = None


class FeedbackResponse(BaseModel):
    success: bool


def _base_dir() -> Path:
    """Return path to backend/.deer-flow. Overridden in tests."""
    from deerflow.config.paths import get_paths

    return Path(get_paths().base_dir)


@router.post("/{thread_id}/feedback", response_model=FeedbackResponse)
def post_feedback(thread_id: str, req: FeedbackRequest) -> FeedbackResponse:
    out_dir = _base_dir() / "training-data" / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "thread_id": thread_id,
        "message_id": req.message_id,
        "verdict": req.verdict,
        "revised_text": req.revised_text,
        "note": req.note,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / f"{thread_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FeedbackResponse(success=True)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py -v`
Expected: 3 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/app/gateway/routers/feedback.py \
       packages/agent/backend/tests/test_feedback_router.py
git commit -m "feat(feedback): Gateway 新增 feedback 路由"
```

---

### Task 7: 在 Gateway app 中注册 feedback router

**目标**：把 `feedback.router` 加进 `app.gateway.app` 的 FastAPI 实例。

**Files:**
- Modify: `packages/agent/backend/app/gateway/app.py`
- Modify: `packages/agent/backend/tests/test_feedback_router.py`

⚠️ **受保护文件提醒**：同 Task 5。改动 2 行。

**Step 1: Write the failing test**

Append to `test_feedback_router.py`:

```python
def test_feedback_router_mounted_on_gateway_app():
    from app.gateway.app import app as gateway_app

    paths = {route.path for route in gateway_app.routes if hasattr(route, "path")}
    assert "/api/threads/{thread_id}/feedback" in paths
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py::test_feedback_router_mounted_on_gateway_app -v`
Expected: FAIL — path not in gateway_app routes.

**Step 3: Write minimal implementation**

In `app/gateway/app.py`:

1. Add import next to other router imports (look for `from app.gateway.routers import ...`):
   ```python
   from app.gateway.routers import feedback as feedback_router
   ```

2. Register the router where other `app.include_router(...)` calls are:
   ```python
   app.include_router(feedback_router.router)
   ```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py -v`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/app/gateway/app.py packages/agent/backend/tests/test_feedback_router.py
git commit -m "feat(feedback): 在 Gateway 注册 feedback router"
```

---

### Task 8: GET 反馈状态（给前端显示已反馈标记用）

**目标**：`GET /api/threads/{thread_id}/feedback` 返回该 thread 所有已提交反馈列表，用于前端页面刷新后仍能看到哪些消息已打分。

**Files:**
- Modify: `packages/agent/backend/app/gateway/routers/feedback.py`
- Modify: `packages/agent/backend/tests/test_feedback_router.py`

**Step 1: Write the failing test**

```python
def test_get_feedback_list(client):
    c, tmp_path = client
    c.post("/api/threads/t-list/feedback", json={"message_id": "m-1", "verdict": "correct"})
    c.post("/api/threads/t-list/feedback", json={"message_id": "m-2", "verdict": "wrong", "revised_text": "R"})

    resp = c.get("/api/threads/t-list/feedback")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    verdicts = {i["message_id"]: i["verdict"] for i in items}
    assert verdicts == {"m-1": "correct", "m-2": "wrong"}


def test_get_feedback_empty_thread_returns_empty_list(client):
    c, _ = client
    resp = c.get("/api/threads/never-touched/feedback")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py -v -k "test_get_feedback"`
Expected: FAIL — GET route returns 405.

**Step 3: Write minimal implementation**

Add to `feedback.py`:

```python
class FeedbackItem(BaseModel):
    message_id: str
    verdict: str
    revised_text: str | None
    note: str | None
    submitted_at: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackItem]


@router.get("/{thread_id}/feedback", response_model=FeedbackListResponse)
def list_feedback(thread_id: str) -> FeedbackListResponse:
    path = _base_dir() / "training-data" / "feedback" / f"{thread_id}.jsonl"
    if not path.exists():
        return FeedbackListResponse(items=[])
    items: list[FeedbackItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        items.append(FeedbackItem(
            message_id=rec["message_id"],
            verdict=rec["verdict"],
            revised_text=rec.get("revised_text"),
            note=rec.get("note"),
            submitted_at=rec["submitted_at"],
        ))
    return FeedbackListResponse(items=items)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_feedback_router.py -v`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/app/gateway/routers/feedback.py \
       packages/agent/backend/tests/test_feedback_router.py
git commit -m "feat(feedback): 新增 GET 反馈列表接口"
```

---

## Phase C: 前端三按钮 UI（Task 9-12）

### Task 9: 前端 API 客户端封装 feedback 接口

**目标**：在 `api-client.ts` 增加 `submitFeedback` 和 `listFeedback` 两个方法。这是纯网络层，不改 UI。

**Files:**
- Modify: `packages/agent/frontend/src/core/api/api-client.ts`
- Create: `packages/agent/frontend/src/core/api/feedback.test.ts`

**Step 1: Locate current client pattern**

Read `packages/agent/frontend/src/core/api/api-client.ts` first (just for structure — edits should match existing conventions for auth headers, base URL, etc.).

**Step 2: Write the failing test**

Create `packages/agent/frontend/src/core/api/feedback.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { submitFeedback, listFeedback } from "./api-client";

describe("feedback API", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it("submitFeedback POSTs to /api/threads/{id}/feedback", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });
    await submitFeedback("thread-1", {
      message_id: "m-1",
      verdict: "correct",
    });
    const call = (global.fetch as any).mock.calls[0];
    expect(call[0]).toContain("/api/threads/thread-1/feedback");
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({
      message_id: "m-1",
      verdict: "correct",
    });
  });

  it("listFeedback GETs and returns items", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ items: [{ message_id: "m-1", verdict: "correct" }] }),
    });
    const result = await listFeedback("thread-1");
    expect(result.items).toHaveLength(1);
  });
});
```

**Step 3: Run test to verify it fails**

Run (from `packages/agent/frontend`): `pnpm test src/core/api/feedback.test.ts`
Expected: FAIL — `submitFeedback is not exported`.

**Step 4: Write minimal implementation**

Append to `api-client.ts` (follow the file's existing fetch helper pattern — if it uses a base URL constant and auth headers, reuse them):

```typescript
export type FeedbackVerdict = "correct" | "needs_fix" | "wrong";

export interface FeedbackRequest {
  message_id: string;
  verdict: FeedbackVerdict;
  revised_text?: string;
  note?: string;
}

export interface FeedbackItem extends FeedbackRequest {
  submitted_at: string;
}

export async function submitFeedback(
  threadId: string,
  body: FeedbackRequest,
): Promise<{ success: boolean }> {
  const res = await fetch(`/api/threads/${threadId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`submitFeedback failed: ${res.status}`);
  return res.json();
}

export async function listFeedback(
  threadId: string,
): Promise<{ items: FeedbackItem[] }> {
  const res = await fetch(`/api/threads/${threadId}/feedback`);
  if (!res.ok) throw new Error(`listFeedback failed: ${res.status}`);
  return res.json();
}
```

**Step 5: Run test to verify it passes**

Run: `pnpm test src/core/api/feedback.test.ts`
Expected: 2 passed.

**Step 6: Commit**

```bash
git add packages/agent/frontend/src/core/api/api-client.ts \
       packages/agent/frontend/src/core/api/feedback.test.ts
git commit -m "feat(frontend): API client 增加 feedback 方法"
```

---

### Task 10: `<FeedbackButtons>` 可复用组件

**目标**：一个独立 React 组件，props 是 `threadId`, `messageId`, `onSubmit`, `existingVerdict?`，渲染三按钮（✅ ⚠️ ❌）。点 ✅ 直接提交；点 ⚠️/❌ 展开 textarea，提交后调 `submitFeedback`。

**Files:**
- Create: `packages/agent/frontend/src/components/feedback/feedback-buttons.tsx`
- Create: `packages/agent/frontend/src/components/feedback/feedback-buttons.test.tsx`

**Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { FeedbackButtons } from "./feedback-buttons";

vi.mock("@/core/api/api-client", () => ({
  submitFeedback: vi.fn().mockResolvedValue({ success: true }),
}));

describe("<FeedbackButtons>", () => {
  it("renders three buttons", () => {
    render(<FeedbackButtons threadId="t" messageId="m" />);
    expect(screen.getByRole("button", { name: /正确/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /需修正/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /错误/ })).toBeInTheDocument();
  });

  it("clicking ✅ submits immediately", async () => {
    const { submitFeedback } = await import("@/core/api/api-client");
    render(<FeedbackButtons threadId="t" messageId="m" />);
    fireEvent.click(screen.getByRole("button", { name: /正确/ }));
    expect(submitFeedback).toHaveBeenCalledWith("t", {
      message_id: "m",
      verdict: "correct",
    });
  });

  it("clicking ⚠️ opens textarea and submit sends revised_text", async () => {
    const { submitFeedback } = await import("@/core/api/api-client");
    (submitFeedback as any).mockClear();
    render(<FeedbackButtons threadId="t" messageId="m" />);
    fireEvent.click(screen.getByRole("button", { name: /需修正/ }));
    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "修正后的内容" } });
    fireEvent.click(screen.getByRole("button", { name: /提交/ }));
    expect(submitFeedback).toHaveBeenCalledWith("t", {
      message_id: "m",
      verdict: "needs_fix",
      revised_text: "修正后的内容",
    });
  });

  it("shows '已反馈' state after submission", async () => {
    render(<FeedbackButtons threadId="t" messageId="m" />);
    fireEvent.click(screen.getByRole("button", { name: /正确/ }));
    await screen.findByText(/已反馈/);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `pnpm test src/components/feedback/feedback-buttons.test.tsx`
Expected: FAIL — component doesn't exist.

**Step 3: Write minimal implementation**

```tsx
"use client";

import { useState } from "react";
import { submitFeedback, type FeedbackVerdict } from "@/core/api/api-client";

interface Props {
  threadId: string;
  messageId: string;
  existingVerdict?: FeedbackVerdict;
}

export function FeedbackButtons({ threadId, messageId, existingVerdict }: Props) {
  const [verdict, setVerdict] = useState<FeedbackVerdict | null>(existingVerdict ?? null);
  const [expandedVerdict, setExpandedVerdict] = useState<FeedbackVerdict | null>(null);
  const [revisedText, setRevisedText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (verdict) {
    return <div className="text-xs text-muted-foreground mt-2">已反馈（{labelOf(verdict)}）</div>;
  }

  const handleCorrect = async () => {
    setSubmitting(true);
    try {
      await submitFeedback(threadId, { message_id: messageId, verdict: "correct" });
      setVerdict("correct");
    } finally {
      setSubmitting(false);
    }
  };

  const handleExpanded = (v: FeedbackVerdict) => {
    setExpandedVerdict(v);
    setRevisedText("");
  };

  const handleSubmitRevision = async () => {
    if (!expandedVerdict || !revisedText.trim()) return;
    setSubmitting(true);
    try {
      await submitFeedback(threadId, {
        message_id: messageId,
        verdict: expandedVerdict,
        revised_text: revisedText.trim(),
      });
      setVerdict(expandedVerdict);
      setExpandedVerdict(null);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mt-2 flex flex-col gap-2">
      <div className="flex gap-1">
        <button
          type="button"
          onClick={handleCorrect}
          disabled={submitting}
          className="text-xs px-2 py-1 rounded hover:bg-accent"
        >
          ✅ 正确
        </button>
        <button
          type="button"
          onClick={() => handleExpanded("needs_fix")}
          disabled={submitting}
          className="text-xs px-2 py-1 rounded hover:bg-accent"
        >
          ⚠️ 需修正
        </button>
        <button
          type="button"
          onClick={() => handleExpanded("wrong")}
          disabled={submitting}
          className="text-xs px-2 py-1 rounded hover:bg-accent"
        >
          ❌ 错误
        </button>
      </div>
      {expandedVerdict && (
        <div className="flex flex-col gap-1">
          <textarea
            role="textbox"
            value={revisedText}
            onChange={(e) => setRevisedText(e.target.value)}
            placeholder={
              expandedVerdict === "needs_fix"
                ? "请写出修正版（专家版本）"
                : "请写出正确的版本"
            }
            className="text-sm p-2 border rounded min-h-[80px]"
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleSubmitRevision}
              disabled={submitting || !revisedText.trim()}
              className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground"
            >
              提交
            </button>
            <button
              type="button"
              onClick={() => setExpandedVerdict(null)}
              className="text-xs px-2 py-1 rounded"
            >
              取消
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function labelOf(v: FeedbackVerdict): string {
  return v === "correct" ? "✅ 正确" : v === "needs_fix" ? "⚠️ 需修正" : "❌ 错误";
}
```

**Step 4: Run test to verify it passes**

Run: `pnpm test src/components/feedback/feedback-buttons.test.tsx`
Expected: 4 passed.

**Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/feedback/
git commit -m "feat(frontend): 新增 FeedbackButtons 组件"
```

---

### Task 11: 在 assistant text message 下挂 `<FeedbackButtons>`

**目标**：定位到渲染 assistant 最终文本消息的组件（Lead agent 的 AIMessage 以及 report-writer / data-analyst 的最终文本），在末尾挂按钮。**不**给 tool call / tool result / 中间 streaming 消息挂 — 只给完整的最终回复挂。

**Files:**
- Modify: 需要 `grep` 找到具体位置，然后修改
- Create: `packages/agent/frontend/src/components/workspace/messages/assistant-feedback.test.tsx`

**Step 1: Locate the correct file**

Run:
```bash
grep -rn "AIMessage\|assistant.*content" packages/agent/frontend/src/components/workspace/messages/ | head
```

Likely candidate: `message-group.tsx` or a dedicated assistant bubble component. Read it to find where text content is rendered (look for `msg.type === "ai"` or similar).

**Step 2: Write the failing test**

Create `assistant-feedback.test.tsx` (adapt selector to whatever component you identified in Step 1):

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
// import the component under test — adjust path after Step 1
import { AssistantMessage } from "./assistant-message";

describe("assistant message + feedback", () => {
  it("renders FeedbackButtons for completed AI text messages", () => {
    render(
      <AssistantMessage
        threadId="t-1"
        messageId="m-1"
        content="分析完成：组间差异显著 p < 0.05"
        isStreaming={false}
      />,
    );
    expect(screen.getByRole("button", { name: /正确/ })).toBeInTheDocument();
  });

  it("does NOT render buttons while streaming", () => {
    render(
      <AssistantMessage
        threadId="t-1"
        messageId="m-1"
        content="分析中..."
        isStreaming={true}
      />,
    );
    expect(screen.queryByRole("button", { name: /正确/ })).not.toBeInTheDocument();
  });
});
```

If the existing component doesn't take these props directly, write the test against the actual signature — the goal is just: "when message is fully rendered AI text, feedback buttons appear".

**Step 3: Run test to verify it fails**

Run: `pnpm test assistant-feedback.test.tsx`
Expected: FAIL — no feedback button rendered.

**Step 4: Write minimal implementation**

In the assistant-message-rendering component, add at the end of the AI text branch:

```tsx
import { FeedbackButtons } from "@/components/feedback/feedback-buttons";

// ...inside the component, after the text is rendered:
{!isStreaming && threadId && messageId && (
  <FeedbackButtons threadId={threadId} messageId={messageId} />
)}
```

Make sure `threadId` and `messageId` are threaded through from the parent — if not already available, pass them as props from `message-list.tsx` or the workspace container.

**Step 5: Run test to verify it passes**

Run: `pnpm test assistant-feedback.test.tsx`
Expected: 2 passed.

Manual sanity check:
```bash
cd packages/agent && make dev
# Open http://localhost:2026, send a message, wait for full reply,
# confirm ✅ ⚠️ ❌ buttons appear under the assistant text.
```

**Step 6: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/messages/
git commit -m "feat(frontend): assistant 消息下挂 FeedbackButtons"
```

---

### Task 12: 在 SubtaskCard 下挂 `<FeedbackButtons>`（subagent 级别反馈）

**目标**：子 agent（code-executor / data-analyst / report-writer）的 CoT 展示卡片 `subtask-card.tsx` 也挂按钮，这样专家能对每个 subagent 的产出单独打分。

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`
- Create: `packages/agent/frontend/src/components/workspace/messages/subtask-card-feedback.test.tsx`

**Step 1: Write the failing test**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SubtaskCard } from "./subtask-card";

describe("<SubtaskCard> with feedback", () => {
  it("shows feedback buttons when subtask is completed", () => {
    render(
      <SubtaskCard
        threadId="t-1"
        taskId="sub-1"
        subagentType="data-analyst"
        status="completed"
        // ...whatever other required props
      />,
    );
    expect(screen.getByRole("button", { name: /正确/ })).toBeInTheDocument();
  });

  it("hides feedback buttons while running", () => {
    render(
      <SubtaskCard
        threadId="t-1"
        taskId="sub-1"
        subagentType="data-analyst"
        status="running"
      />,
    );
    expect(screen.queryByRole("button", { name: /正确/ })).not.toBeInTheDocument();
  });
});
```

**Step 2: Run test to verify it fails**

Run: `pnpm test subtask-card-feedback.test.tsx`
Expected: FAIL.

**Step 3: Write minimal implementation**

In `subtask-card.tsx`, at the bottom of the card (when status === "completed"):

```tsx
import { FeedbackButtons } from "@/components/feedback/feedback-buttons";

{/* near end of the card JSX */}
{status === "completed" && threadId && taskId && (
  <FeedbackButtons threadId={threadId} messageId={`subtask-${taskId}`} />
)}
```

Note: use `subtask-<taskId>` as message_id so backend can distinguish lead-text vs subtask feedback when joining later.

**Step 4: Run test to verify it passes**

Run: `pnpm test subtask-card-feedback.test.tsx`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add packages/agent/frontend/src/components/workspace/messages/
git commit -m "feat(frontend): SubtaskCard 下挂 FeedbackButtons"
```

---

## Phase D: 后处理 + 仪表板（Task 13-15）

### Task 13: `extract_e2e_sessions.py` — 把录制数据转 Fireworks JSONL

**目标**：读 `auto-collected/*.jsonl` + `feedback/*.jsonl`，join 成最终训练集：
- 每条 lead/subagent 样本加上对应的 feedback（若有）
- `verdict=correct` → 进 SFT（正样本）
- `verdict=needs_fix/wrong` 且有 `revised_text` → SFT 用 revised_text + DPO `(chosen=revised, rejected=original)`
- 输出：`training-data/processed/sft.jsonl`, `dpo.jsonl`, `stats.json`

**Files:**
- Create: `packages/agent/backend/scripts/extract_e2e_sessions.py`
- Create: `packages/agent/backend/tests/test_extract_e2e_sessions.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path

from scripts.extract_e2e_sessions import extract_sessions


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def test_extract_joins_lead_samples_and_correct_feedback(tmp_path):
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-1.jsonl",
        [
            {"role": "lead", "thread_id": "t-1", "input": "hi", "output": "ok",
             "thinking": "", "recorded_at": "2026-04-23T00:00:00+00:00"},
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-1.jsonl",
        [{"thread_id": "t-1", "message_id": "m-1", "verdict": "correct",
          "revised_text": None, "note": None, "submitted_at": "2026-04-23T00:01:00+00:00"}],
    )

    stats = extract_sessions(tmp_path)

    sft = [json.loads(l) for l in (tmp_path / "training-data" / "processed" / "sft.jsonl").read_text().splitlines()]
    assert len(sft) >= 1
    assert stats["sft_count"] >= 1
    assert stats["dpo_count"] == 0


def test_extract_generates_dpo_pair_from_needs_fix(tmp_path):
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-2.jsonl",
        [
            {"role": "lead", "thread_id": "t-2", "input": "分析",
             "output": "泛泛回答", "thinking": "", "recorded_at": "2026-04-23T00:00:00+00:00"},
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-2.jsonl",
        [{"thread_id": "t-2", "message_id": "m-1", "verdict": "needs_fix",
          "revised_text": "专家级解读", "note": "加上 p 值",
          "submitted_at": "2026-04-23T00:01:00+00:00"}],
    )

    stats = extract_sessions(tmp_path)

    dpo = [json.loads(l) for l in (tmp_path / "training-data" / "processed" / "dpo.jsonl").read_text().splitlines()]
    assert len(dpo) == 1
    assert dpo[0]["chosen"] == "专家级解读"
    assert dpo[0]["rejected"] == "泛泛回答"
    assert stats["dpo_count"] == 1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_extract_e2e_sessions.py -v`
Expected: FAIL — `scripts/extract_e2e_sessions.py` missing.

**Step 3: Write minimal implementation**

Create `scripts/extract_e2e_sessions.py`:

```python
"""Post-process training data: join auto-collected samples with feedback.

Outputs:
- training-data/processed/sft.jsonl — Fireworks ChatML
- training-data/processed/dpo.jsonl — preference pairs
- training-data/processed/stats.json — counts for daily dashboard
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sft_record(sample: dict, output_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": output_text, "thinking": sample.get("thinking", "")},
        ],
        "metadata": {
            "thread_id": sample["thread_id"],
            "role": sample["role"],
        },
    }


def extract_sessions(base_dir: Path) -> dict:
    collected_dir = base_dir / "training-data" / "auto-collected"
    feedback_dir = base_dir / "training-data" / "feedback"
    out_dir = base_dir / "training-data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-thread feedback index: {thread_id: [items]}
    feedback_by_thread: dict[str, list[dict]] = {}
    if feedback_dir.exists():
        for f in feedback_dir.glob("*.jsonl"):
            feedback_by_thread[f.stem] = _read_jsonl(f)

    sft: list[dict] = []
    dpo: list[dict] = []

    if collected_dir.exists():
        for f in collected_dir.glob("*.jsonl"):
            thread_id = f.stem
            samples = _read_jsonl(f)
            feedbacks = feedback_by_thread.get(thread_id, [])
            # Naive join: feedback without message_id match → apply to all lead samples in thread
            # (v0.1: improve once frontend sends real message_id)
            for sample in samples:
                # Find first feedback for this sample; for v0.1 we accept all feedback as applying to the thread
                fb = next((f for f in feedbacks), None)
                if fb is None:
                    # No feedback yet — include as SFT with low-confidence tag
                    sft.append(_sft_record(sample, sample["output"]))
                    continue
                verdict = fb["verdict"]
                if verdict == "correct":
                    sft.append(_sft_record(sample, sample["output"]))
                elif fb.get("revised_text"):
                    # Both SFT (with revised) AND DPO pair
                    sft.append(_sft_record(sample, fb["revised_text"]))
                    dpo.append({
                        "prompt": sample["input"],
                        "chosen": fb["revised_text"],
                        "rejected": sample["output"],
                        "metadata": {"thread_id": thread_id, "verdict": verdict},
                    })
                # verdict == "wrong" with no revision → skip entirely

    (out_dir / "sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sft) + ("\n" if sft else ""),
        encoding="utf-8",
    )
    (out_dir / "dpo.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in dpo) + ("\n" if dpo else ""),
        encoding="utf-8",
    )

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sft_count": len(sft),
        "dpo_count": len(dpo),
        "threads_processed": len(list(collected_dir.glob("*.jsonl"))) if collected_dir.exists() else 0,
        "threads_with_feedback": len(feedback_by_thread),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


if __name__ == "__main__":
    default_base = Path(__file__).resolve().parent.parent / ".deer-flow"
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else default_base
    stats = extract_sessions(base)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_extract_e2e_sessions.py -v`
Expected: 2 passed.

**Step 5: Commit**

```bash
git add packages/agent/backend/scripts/extract_e2e_sessions.py \
       packages/agent/backend/tests/test_extract_e2e_sessions.py
git commit -m "feat(training): 新增 extract_e2e_sessions 后处理脚本"
```

---

### Task 14: `make training-stats` 命令 + 每日仪表板

**目标**：加一个 `make training-stats` 命令，跑 `extract_e2e_sessions.py` 并打印人类可读的日报（累计样本数、反馈率、各 verdict 比例、距离 800 条目标还差多少）。

**Files:**
- Modify: `packages/agent/backend/Makefile`
- Create: `packages/agent/backend/scripts/training_dashboard.py`
- Create: `packages/agent/backend/tests/test_training_dashboard.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.training_dashboard import format_dashboard


def test_format_dashboard_renders_progress_bar():
    stats = {"sft_count": 150, "dpo_count": 20, "threads_processed": 30, "threads_with_feedback": 18}
    output = format_dashboard(stats, target_sft=800)
    assert "150" in output
    assert "800" in output
    assert "18%" in output or "18.75%" in output or "19%" in output  # progress


def test_format_dashboard_handles_zero():
    stats = {"sft_count": 0, "dpo_count": 0, "threads_processed": 0, "threads_with_feedback": 0}
    output = format_dashboard(stats, target_sft=800)
    assert "0" in output
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_training_dashboard.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

Create `scripts/training_dashboard.py`:

```python
"""Human-readable training data dashboard. Called via `make training-stats`."""
import sys
from pathlib import Path

from scripts.extract_e2e_sessions import extract_sessions


def format_dashboard(stats: dict, target_sft: int = 800) -> str:
    sft = stats.get("sft_count", 0)
    dpo = stats.get("dpo_count", 0)
    threads = stats.get("threads_processed", 0)
    feedback_threads = stats.get("threads_with_feedback", 0)
    pct = round(sft * 100 / target_sft) if target_sft else 0
    feedback_rate = round(feedback_threads * 100 / threads) if threads else 0

    bar_width = 30
    filled = int(bar_width * min(sft, target_sft) / target_sft) if target_sft else 0
    bar = "█" * filled + "░" * (bar_width - filled)

    return f"""
训练数据飞轮日报
================================
SFT 样本:  {sft:>4} / {target_sft}  [{bar}]  {pct}%
DPO 对:    {dpo:>4}
线程处理:  {threads:>4}
有反馈的:  {feedback_threads:>4}  ({feedback_rate}%)
================================
距离 Phase 1 目标还差 {max(0, target_sft - sft)} 条 SFT 样本
"""


if __name__ == "__main__":
    default_base = Path(__file__).resolve().parent.parent / ".deer-flow"
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else default_base
    stats = extract_sessions(base)
    print(format_dashboard(stats))
```

Add to `Makefile`:

```makefile
training-stats:
	PYTHONPATH=. uv run python scripts/training_dashboard.py
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_training_dashboard.py -v`
Expected: 2 passed.

Also smoke-test: `make training-stats` (should print the dashboard without error, all zeros on fresh repo).

**Step 5: Commit**

```bash
git add packages/agent/backend/scripts/training_dashboard.py \
       packages/agent/backend/tests/test_training_dashboard.py \
       packages/agent/backend/Makefile
git commit -m "feat(training): 新增 make training-stats 日报命令"
```

---

### Task 15: 文档更新 — CLAUDE.md + 使用 SOP

**目标**：(1) 在项目 `CLAUDE.md` 和 `backend/CLAUDE.md` 里记录飞轮的存在和命令；(2) 新建行为学同事使用 SOP 文档。

**Files:**
- Modify: `CLAUDE.md` (项目根)
- Modify: `packages/agent/backend/CLAUDE.md`
- Create: `docs/sop/training-data-flywheel-sop.md`

**Step 1: No tests for docs — just verify after**

(Docs have no unit tests; we verify by reading.)

**Step 2: Update project root `CLAUDE.md`**

Under "重要注意事项"，append:

```markdown
7. **训练数据飞轮已启动** — 每次 agent 会话自动录制到 `packages/agent/backend/.deer-flow/training-data/auto-collected/`；专家反馈走 `/api/threads/{id}/feedback` API + 前端三按钮。查看累计进度：`cd packages/agent/backend && make training-stats`。详见 [docs/sop/training-data-flywheel-sop.md](docs/sop/training-data-flywheel-sop.md)。
```

**Step 3: Update `backend/CLAUDE.md`**

In the middleware chain section (line ~1-12 in the middleware list), add between MemoryMiddleware and ViewImageMiddleware:

```markdown
10. **TrainingDataMiddleware** - Records every turn (lead + subagent) as Fireworks JSONL to `.deer-flow/training-data/auto-collected/<thread_id>.jsonl` for Phase 1 SFT/DPO. Filters low-quality (errors, timeouts, empty).
```

Renumber subsequent middleware items.

**Step 4: Create SOP**

Write `docs/sop/training-data-flywheel-sop.md`:

```markdown
# 训练数据飞轮 — 行为学同事使用 SOP

> 2026-04-23 启动。目标：两个月内攒够 800 条 SFT + 300 对 DPO。

## 一句话说明

**你正常使用 EthoInsight 就在贡献训练数据**。每次对话系统自动录制。你在 assistant 回复下打 ✅/⚠️/❌，就在帮我们教未来的自研模型。

## 使用步骤

1. 打开 http://localhost:2026
2. 新建对话，上传你的 EthoVision 数据，像平时一样做分析
3. Agent 每输出一段 assistant 回复或 subtask 卡片，下面都会有三个按钮：
   - **✅ 正确** — 回复没问题，一键提交
   - **⚠️ 需修正** — 基本对但某些点需要改，点开后写出修正版
   - **❌ 错误** — 整体错了，写出正确版本
4. 一次会话结束，所有反馈自动存档

## 你不用做的事

- 不用手动导出任何数据
- 不用记录任何元信息
- 不用担心"反馈不够专业"—即使简单的 ✅ 也是有价值的信号

## 隐私承诺

- 所有数据存在 `packages/agent/backend/.deer-flow/training-data/` 本地目录
- 不上传任何外部服务
- 你可以随时删除某次会话的录制（删除 `auto-collected/<thread_id>.jsonl`）

## 飞轮状态查看（给工程）

```bash
cd packages/agent/backend
make training-stats
```

显示累计样本数、DPO 对数、反馈率、距离目标进度。

## 反馈聚合时机

每周一上午工程跑一次 `scripts/extract_e2e_sessions.py`，合成当周的 SFT/DPO 数据集。届时会发周报给所有同事。

## 推荐节奏

每位同事每周做 2-3 次完整分析，每次打 5-10 条反馈。3-4 位同事 × 2 个月 = 达标。
```

**Step 5: Commit**

```bash
git add CLAUDE.md packages/agent/backend/CLAUDE.md docs/sop/training-data-flywheel-sop.md
git commit -m "docs: 训练数据飞轮 SOP + CLAUDE.md 更新"
```

---

## Phase E: 集成验证（Task 16-17）

### Task 16: 端到端 E2E 验证

**目标**：真实跑一次 agent 会话，打几个反馈，跑 `make training-stats`，验证数据完整且格式合规。

**Files:**
- Create: `packages/agent/backend/tests/test_training_flywheel_e2e.py`

**Step 1: Write the e2e test**

```python
"""End-to-end smoke test: middleware records → feedback API writes → extract produces valid JSONL."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware
from scripts.extract_e2e_sessions import extract_sessions


def test_flywheel_e2e(tmp_path, monkeypatch):
    # 1. Record a lead sample
    mw = TrainingDataMiddleware(base_dir=str(tmp_path))
    sb = mw.before_agent(state={}, runtime=Runtime(context={"thread_id": "t-e2e"}))
    state = {
        "training_data_path": sb["training_data_path"],
        "messages": [
            HumanMessage(content="分析斑马鱼"),
            AIMessage(content="泛泛的分析结果"),
        ],
    }
    mw.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-e2e"}))

    # 2. Submit feedback via API
    from app.gateway.routers import feedback as feedback_mod
    monkeypatch.setattr(feedback_mod, "_base_dir", lambda: tmp_path)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(feedback_mod.router)
    client = TestClient(app)

    resp = client.post(
        "/api/threads/t-e2e/feedback",
        json={"message_id": "m-1", "verdict": "needs_fix", "revised_text": "专家级分析"},
    )
    assert resp.status_code == 200

    # 3. Extract
    stats = extract_sessions(tmp_path)

    # 4. Verify
    assert stats["sft_count"] >= 1
    assert stats["dpo_count"] == 1

    sft = [json.loads(l) for l in (tmp_path / "training-data" / "processed" / "sft.jsonl").read_text().splitlines()]
    assert sft[0]["messages"][0]["role"] == "user"
    assert sft[0]["messages"][1]["role"] == "assistant"
    assert sft[0]["messages"][1]["content"] == "专家级分析"  # revised version

    dpo = [json.loads(l) for l in (tmp_path / "training-data" / "processed" / "dpo.jsonl").read_text().splitlines()]
    assert dpo[0]["chosen"] == "专家级分析"
    assert dpo[0]["rejected"] == "泛泛的分析结果"
```

**Step 2: Run**

Run: `PYTHONPATH=. uv run pytest tests/test_training_flywheel_e2e.py -v`
Expected: 1 passed.

**Step 3: Manual smoke**

```bash
cd packages/agent && make dev
# Open http://localhost:2026
# Do a full shoaling analysis using demo data
# Click ✅ on lead response, ⚠️ on one subtask with revision text
cd backend && make training-stats
```

Expected dashboard shows `sft_count >= 2`, `dpo_count >= 1`.

**Step 4: Commit**

```bash
git add packages/agent/backend/tests/test_training_flywheel_e2e.py
git commit -m "test(training): 飞轮端到端集成测试"
```

---

### Task 17: 全量回归 + handoff 文档

**目标**：跑全量后端测试 + lint，确认没回归。写本次工作的 handoff 文档。

**Files:**
- Create: `docs/handoffs/2026-04-23-training-data-flywheel-done.md`

**Step 1: Run full regression**

```bash
cd packages/agent/backend
source .venv/bin/activate
make test
```

Expected: all previously passing tests still pass, plus new tests. Baseline before this plan was 1670 — new should be ≥ 1680+ (roughly 16 new tests added across tasks).

```bash
make lint
```

Expected: no lint errors. Fix any ruff complaints.

Frontend:
```bash
cd packages/agent/frontend
pnpm test
```

Expected: all tests pass.

**Step 2: Write handoff**

Create `docs/handoffs/2026-04-23-training-data-flywheel-done.md` following the style of `2026-04-23-m01-remaining-items-done.md`. Include:

- Summary of all 17 tasks
- Files created/modified list
- Test count delta
- How to start using the flywheel (link SOP)
- Remaining work (Phase 1 training itself — separate plan)

**Step 3: Commit**

```bash
git add docs/handoffs/2026-04-23-training-data-flywheel-done.md
git commit -m "docs: 训练数据飞轮 handoff 文档"
```

---

## 总计

- **17 tasks** across 5 phases
- Phase A 后端中间件：5 tasks
- Phase B Gateway API：3 tasks
- Phase C 前端 UI：4 tasks
- Phase D 后处理 + 文档：3 tasks
- Phase E 集成验证：2 tasks
- 预计总时长 **1.5-2 周**（单人全职）
- 每个 task 2-5 分钟的步骤，一般 1-3 小时工程时间

## 验收标准（全部完成后）

1. ✅ `make test` 全绿（后端 + 前端）
2. ✅ `make training-stats` 输出非零日报
3. ✅ 在 UI 里真跑一次分析，三按钮可用，反馈落盘
4. ✅ `make dev` 启动正常，没有 middleware 冲突
5. ✅ Handoff 文档写好、SOP 可分享给行为学同事
6. ✅ 至少 1 个专家愿意用 — 收到首份真实反馈（计划之外的验收，但这是飞轮真正"在转"的标志）

完成后飞轮进入**自驱动状态**。每周跑一次 `extract_e2e_sessions.py` 观察数据累积，2 个月达标 → 进 Phase 1 训练。
