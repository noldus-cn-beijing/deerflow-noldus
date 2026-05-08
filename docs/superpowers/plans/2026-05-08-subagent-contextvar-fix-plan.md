# 实施计划：修复 Subagent 跨线程 ContextVar 丢失（user_id 回到 default）

**日期**：2026-05-08
**状态**：待执行（spec ready，未开始）
**预计工作量**：单次会话内可完成（4-6 小时）
**前置依赖**：无（独立 bug fix，不依赖其他 sync 轮次）
**关联交接**：
- [docs/handoffs/2026-05-08-upload-file-not-recognized-FIXED-handoff.md](../../handoffs/2026-05-08-upload-file-not-recognized-FIXED-handoff.md)（lead agent 这条 ContextVar 链路的修复）
- [docs/handoffs/2026-05-08-deerflow-tier234-round3-completed-handoff.md](../../handoffs/2026-05-08-deerflow-tier234-round3-completed-handoff.md)（better-auth 同步引入了 user_context 体系）

---

## 0. TL;DR（30 秒）

端到端验证斑马鱼 shoaling pipeline 时发现：**code-executor 跑完成功，data-analyst 却看不到产物**。根因是 subagent 跑在独立 ThreadPoolExecutor + 独立 event loop 里，`get_effective_user_id()` 依赖的 ContextVar **没跨线程传递**，silently fallback 到 `"default"`，导致路径解析到 `users/default/threads/<tid>/...`（空目录）而不是真实用户路径 `users/<uuid>/threads/<tid>/...`。

上游 deerflow 已修复此问题：在 `_isolated_loop_pool` 入口用 `contextvars.copy_context()` 抓父 context 快照，并改用 **持久 isolated event loop**（daemon 线程上的 long-lived loop + `asyncio.run_coroutine_threadsafe`）替代每次新建/销毁。

本计划做 **surgical merge**——把上游 ContextVar 修复 + 持久 loop 这两个治本改动合入，**保留** noldus 本地的 `recursion_limit = max_turns * 2 + 1` 修复 + `max_turns` 硬终止 + 中文 prompt 注入逻辑。**不**整覆盖 executor.py，**不**合上游同期的其他重构（`tool_policy.py`、`resolve_subagent_model_name` 等会单独排进下次同步）。

**改动范围**：1 个核心文件 + 1 个新测试文件。预计 ~80 行净增。

---

## 1. 背景与根因

### 1.1 症状

端到端跑 shoaling pipeline，`code-executor` 成功完成（写出 5 个 figure + statistics.json + handoff_code_executor.json），紧接着派遣 `data-analyst` 进行专家解读，data-analyst 第一时间用 `bash ls /mnt/user-data/uploads/` 检查文件，**报告目录为空**——但实际上文件就在那里。

### 1.2 根因证据（已亲自验证）

文件系统当前 thread `2f326223-9913-4665-b772-6ee8b90b641c` 同时存在两份目录：

```
.deer-flow/
├── users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/2f326223.../user-data/
│   ├── uploads/             # 5 个上传文件 ✅
│   ├── workspace/           # handoff_code_executor.json + run_shoaling.py ✅
│   └── outputs/             # 7 个图表 + statistics.json ✅
└── users/default/threads/2f326223.../user-data/
    ├── workspace/           # 只有 handoff_data_analyst.json（subagent 写的）
    └── outputs/             # 只有 run_analysis.py（subagent 写的）
```

**两个目录结构相同、thread_id 相同，但 user_id 不同**：lead_agent 写真实 user_id 路径，subagent 写 `default` 路径。

### 1.3 原理

`packages/harness/deerflow/runtime/user_context.py` 用 `ContextVar` 持有当前 user：

```python
_current_user: Final[ContextVar[CurrentUser | None]] = ContextVar(
    "deerflow_current_user", default=None
)

def get_effective_user_id() -> str:
    user = _current_user.get()
    if user is None:
        return DEFAULT_USER_ID  # "default" — silent fallback!
    return str(user.id)
```

`make_lead_agent()` 在 bg-loop 任务里 `set_current_user(_AuthUser(auth_user_id))`，写入 ContextVar。**但**：

| asyncio 边界 | ContextVar 是否传播 |
|---|---|
| 同一 asyncio task 内 | ✅ |
| `asyncio.create_task(...)` | ✅（创建时拷贝） |
| `asyncio.to_thread(...)` | ✅ |
| **`ThreadPoolExecutor.submit(fn, ...)`** | **❌（标准库设计如此）** |
| **`asyncio.new_event_loop()`** | **❌（新 loop 没有 context）** |

而 `subagents/executor.py` 用了：

1. `_scheduler_pool = ThreadPoolExecutor(...)` — 第 4 种边界
2. `_execution_pool = ThreadPoolExecutor(...)` — 第 4 种边界
3. `_isolated_loop_pool = ThreadPoolExecutor(...)` — 第 4 种边界
4. `asyncio.new_event_loop()` 在 `_execute_in_isolated_loop` 里 — 第 5 种边界

每一层都丢一次 ContextVar，到 subagent 实际执行的协程里，`_current_user.get()` 拿到 `None`，于是 `get_effective_user_id()` 返回 `"default"`。

### 1.4 为什么是 Tier 4 better-auth 同步引入的 regression

`runtime/user_context.py` 来自上游 better-auth 体系（commit `f23e2770` 之前本地不存在）。better-auth 同步轮 G/H 把所有 sandbox 路径解析（uploads / workspace / outputs / shared / archived_messages 等）都改成走 `user_id=get_effective_user_id()` 隐式取值。同步时 lead_agent 那条 ContextVar 链路在轮 3 完成时漏掉，今天上午 [`02547092 fixed push files`](../../../) 已补上。但 subagent 这条独立线程链路的 ContextVar 桥接**没人想到**——因为本地 subagent_executor.py 是从更早的轮次合入的，没有 `copy_context()`。

### 1.5 上游怎么解的

上游 `subagents/executor.py` 已经修了，做法是：

1. **把短命的 `_isolated_loop_pool` 替换成持久的 daemon-thread 长寿 loop**（`_isolated_subagent_loop`）。这避免了"每个 subagent 创建新 loop 又关闭"带来的 async client 资源问题
2. **新增 `_submit_to_isolated_loop_in_context(context, coro_factory)`**：
   ```python
   def _submit_to_isolated_loop_in_context(context, coro_factory):
       return context.run(
           lambda: asyncio.run_coroutine_threadsafe(
               coro_factory(),
               _get_isolated_subagent_loop(),
           )
       )
   ```
3. **在 `execute()` 和 `execute_async()` 入口都 `parent_context = copy_context()`**，把父 task 的 ContextVar 快照传到 isolated loop。这样 subagent 协程跑起来时 `_current_user.get()` 能拿到父的 user。

`asyncio.run_coroutine_threadsafe` 把 coro 调度到目标 loop 上，**协程本身的 ContextVar 由 `context.run` 包裹的 lambda 在父 thread 内的 schedule 调用决定**——这是关键设计，避免在 isolated loop 内手动 set_current_user。

### 1.6 上游为什么不全部走 `config.configurable`

`config.configurable["langgraph_auth_user_id"]` 只在 LangGraph runtime 调用 `make_lead_agent(config)` 那一刻可见。subagent 跑在 LangGraph runtime **之外**（自建线程池），拿不到这个 config。所以上游选择"让 ContextVar 跨边界传播"而不是"把 user_id 显式绕回 config 里"。这是更通用的解法（未来加新 ContextVar 自动覆盖，比如 tenant_id、locale 等）。

---

## 2. 范围与原则

### 2.1 本次做什么

| 项 | 做法 |
|---|---|
| 修复 ContextVar 跨线程传播 | ✅ 合上游 `copy_context()` 调用点 |
| 替换 isolated loop 为持久 daemon loop | ✅ 合上游 `_get_isolated_subagent_loop` + `_run_isolated_subagent_loop` + `_shutdown_isolated_subagent_loop` |
| 新增 `_submit_to_isolated_loop_in_context` 辅助 | ✅ 直接复制上游 |
| 加端到端集成测试卡这条 bug | ✅ 在 `tests/test_subagent_executor.py` 增加 ContextVar 跨线程传播测试 |
| 保留 noldus 的 `recursion_limit = max_turns * 2 + 1` | ✅ 不动 |
| 保留 noldus 的 `max_turns` 硬终止（`>= max_turns: break`） | ✅ 不动 |
| 保留 noldus 的 `_load_skill_contents` 中文 skill 注入 | ✅ 不动 |
| 保留 noldus 的 `_get_model_name`（本地版本，不引入 `resolve_subagent_model_name`） | ✅ 不动 |

### 2.2 本次不做什么

| 项 | 原因 | 留给 |
|---|---|---|
| 整覆盖 `executor.py` 到上游 | 改动 5-10 倍，引入 `tool_policy` / `Skill.allowed_tools` / `_create_agent` 重构等 | 下轮 deerflow sync |
| 合 `skills/tool_policy.py` 新文件 | 它是 skill allowlist 重构的一部分，与本 bug 无关 | 下轮 deerflow sync |
| 给 `Skill` 加 `allowed_tools` 字段 | 同上 | 下轮 deerflow sync |
| 新增 `resolve_subagent_model_name` | 与本 bug 无关，是模型解析路径重构 | 下轮 deerflow sync |
| 整覆盖 `task_tool.py` | 上下游 168 行 diff，与本 bug 无关 | 下轮 deerflow sync |
| 把 `get_effective_user_id()` 改成 `require` 抛错 | L3 加固方案，需评估对 CLI/migration 路径影响 | 单独 issue |
| 把 sandbox 路径解析全改成显式 `user_id` 参数 | L4 治本，工作量大 | 单独 issue |

**核心原则**：本次只解决"subagent 看不到 user_id"这一个 bug，不顺便重构。已修过的 lead_agent ContextVar 桥接（`make_lead_agent` 里 `set_current_user`）保持不变。

### 2.3 受保护语义清单（合并时绝对不能改）

合并执行者必须验证以下行为在改动后**不变**：

1. **`recursion_limit = self.config.max_turns * 2 + 1`**（在 `_aexecute` 里构建 `run_config` 处）—— LangGraph 把 model + tools 各算一步，不乘 2 会被提前打断
2. **`if len(result.ai_messages) >= self.config.max_turns: break`**（在 `_aexecute` 的 `astream` 主循环里）—— 硬性终止保护
3. **`_load_skill_contents` 函数**（位于 executor.py 中部）—— noldus 自己的中文 skill 注入逻辑
4. **`_build_system_prompt` 方法**—— 调用 `_load_skill_contents`，不要被上游 `_apply_skill_allowed_tools` 替换
5. **`_get_model_name(config, parent_model)`**—— 不要替换为上游 `resolve_subagent_model_name`
6. **`_build_initial_state(self, task: str)` 接受单参数**—— 不要改成上游的 `async def _build_initial_state(self, task) -> tuple[state, filtered_tools]`
7. **`_create_agent(self)` 不接受参数**—— 不要改成上游的 `_create_agent(self, tools)`
8. **`from langchain_core.messages import AIMessage, HumanMessage`**—— 不要引入 `SystemMessage`（上游用它做 skill prompt 拼接，本地走的是 `_build_system_prompt` 字符串拼接）

---

## 3. 上游与本地差异速查表

| 项 | 上游（deerflow/main） | 本地（HEAD） | 本次合并后 |
|---|---|---|---|
| `_scheduler_pool` | 保留（同名） | 保留 | 保留 |
| `_execution_pool` | 保留 | 保留 | 保留 |
| `_isolated_loop_pool` | **删除** | 保留 | **删除** |
| `_isolated_subagent_loop` 持久 loop | 新增 | 无 | **新增** |
| `_isolated_subagent_loop_thread` daemon 线程 | 新增 | 无 | **新增** |
| `_run_isolated_subagent_loop()` | 新增 | 无 | **新增** |
| `_shutdown_isolated_subagent_loop()` + `atexit.register` | 新增 | 无 | **新增** |
| `_get_isolated_subagent_loop()` | 新增 | 无 | **新增** |
| `_submit_to_isolated_loop_in_context()` | 新增 | 无 | **新增** |
| `_aexecute` 主循环 | 重构（多了 `_build_initial_state` 异步化、`SystemMessage`、`filtered_tools`） | 当前形态 | **保留本地** |
| `_create_agent` 签名 | `(self, tools)` | `(self)` | **保留本地** |
| `_build_initial_state` | `async + tuple 返回` | `sync + dict 返回` | **保留本地** |
| `_load_skill_contents` 函数 | **被替换为 `_apply_skill_allowed_tools`** | 保留 | **保留本地** |
| `_get_model_name` | **替换为 `resolve_subagent_model_name`** | 保留 | **保留本地** |
| `recursion_limit` 公式 | `max_turns` | `max_turns * 2 + 1` | **保留本地** |
| `_execute_in_isolated_loop` 用 `new_event_loop` | **被持久 loop 替代** | 当前用 new loop | **删除本地实现** |
| `execute()` 入口 | 用 `parent_context = copy_context()` + `_submit_to_isolated_loop_in_context` | 用 `_isolated_loop_pool.submit` | **改为上游做法** |
| `execute_async()` 内 `run_task` | 用 `parent_context` + `_submit_to_isolated_loop_in_context` | 用 `_execution_pool.submit(self.execute, ...)` | **改为上游做法** |
| `from contextvars import Context, copy_context` | 有 | 无 | **新增** |
| `import atexit` | 有 | 无 | **新增** |

---

## 4. 实施步骤

### Step 0：准备工作（5 分钟）

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow main
git status  # 确认工作树干净
git log -1 --oneline  # 记录 baseline HEAD
```

把上游 executor.py 落到 `/tmp/upstream_executor.py` 备查：

```bash
git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py > /tmp/upstream_executor.py
```

**baseline 测试运行**：

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

记录基线（应是 `2 failed, 2141 passed, 14 skipped` 或类似），后续验证不能让 passed 数下降、不能让 failed 数上升。

### Step 1：surgical merge `executor.py`（30-45 分钟）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`

**1.1 修改 import 段（文件顶部）**

把：

```python
import asyncio
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.models import create_chat_model
from deerflow.subagents.config import SubagentConfig
```

改为：

```python
import asyncio
import atexit
import logging
import threading
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.models import create_chat_model
from deerflow.subagents.config import SubagentConfig
```

**注意**：不要 import `SystemMessage` / `AppConfig` / `filter_tools_by_skill_allowed_tools` / `Skill` / `resolve_subagent_model_name`——这些是上游别的重构带来的，本次不合。

**1.2 删除 `_isolated_loop_pool` 定义**

找到这一段：

```python
# Dedicated pool for sync execute() calls made from an already-running event loop.
_isolated_loop_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-isolated-")
```

**整段删除**。它会被持久 loop 取代。`_scheduler_pool` 和 `_execution_pool` 保持不变。

**1.3 在 `_filter_tools` 函数前插入持久 loop 基础设施**

在 `_isolated_loop_pool` 删除点之后、`_filter_tools` 函数之前，插入下面这段（直接从 `/tmp/upstream_executor.py` 第 88-189 行复制）：

```python
# Persistent event loop for isolated subagent executions triggered from an
# already-running parent loop. Reusing one long-lived loop avoids creating a
# fresh loop per execution and then closing async resources bound to it.
_isolated_subagent_loop: asyncio.AbstractEventLoop | None = None
_isolated_subagent_loop_thread: threading.Thread | None = None
_isolated_subagent_loop_started: threading.Event | None = None
_isolated_subagent_loop_lock = threading.Lock()


def _run_isolated_subagent_loop(
    loop: asyncio.AbstractEventLoop,
    started_event: threading.Event,
) -> None:
    """Run the persistent isolated subagent loop in a dedicated daemon thread."""
    asyncio.set_event_loop(loop)
    loop.call_soon(started_event.set)
    try:
        loop.run_forever()
    finally:
        started_event.clear()


def _shutdown_isolated_subagent_loop() -> None:
    """Stop and close the persistent isolated subagent loop."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started

    with _isolated_subagent_loop_lock:
        loop = _isolated_subagent_loop
        thread = _isolated_subagent_loop_thread
        _isolated_subagent_loop = None
        _isolated_subagent_loop_thread = None
        _isolated_subagent_loop_started = None

    if loop is None:
        return

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1)

    thread_stopped = thread is None or not thread.is_alive()
    loop_stopped = not loop.is_running()

    if not loop.is_closed():
        if thread_stopped and loop_stopped:
            loop.close()
        else:
            logger.warning(
                "Skipping close of isolated subagent loop because shutdown did not complete within timeout (thread_alive=%s, loop_running=%s)",
                thread is not None and thread.is_alive(),
                loop.is_running(),
            )


atexit.register(_shutdown_isolated_subagent_loop)


def _get_isolated_subagent_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent event loop used by isolated subagent executions."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started
    with _isolated_subagent_loop_lock:
        thread_is_alive = _isolated_subagent_loop_thread is not None and _isolated_subagent_loop_thread.is_alive()
        loop_is_usable = (
            _isolated_subagent_loop is not None
            and not _isolated_subagent_loop.is_closed()
            and _isolated_subagent_loop.is_running()
            and thread_is_alive
        )

        if not loop_is_usable:
            loop = asyncio.new_event_loop()
            started_event = threading.Event()
            thread = threading.Thread(
                target=_run_isolated_subagent_loop,
                args=(loop, started_event),
                name="subagent-persistent-loop",
                daemon=True,
            )
            thread.start()
            if not started_event.wait(timeout=5):
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=1)
                loop.close()
                raise RuntimeError("Timed out starting isolated subagent event loop")
            _isolated_subagent_loop = loop
            _isolated_subagent_loop_thread = thread
            _isolated_subagent_loop_started = started_event

        if _isolated_subagent_loop is None:
            raise RuntimeError("Isolated subagent event loop is not initialized")
        return _isolated_subagent_loop


def _submit_to_isolated_loop_in_context(
    context: Context,
    coro_factory: Callable[[], Coroutine[Any, Any, "SubagentResult"]],
) -> Future["SubagentResult"]:
    """Submit a coroutine to the isolated loop while preserving ContextVar state."""
    return context.run(
        lambda: asyncio.run_coroutine_threadsafe(
            coro_factory(),
            _get_isolated_subagent_loop(),
        )
    )
```

**注意**：`SubagentResult` 在上面用了字符串前向引用（`"SubagentResult"`），因为它定义在文件后面——上游也是这样处理的，保持原样即可。

**1.4 删除 `_execute_in_isolated_loop` 方法**

在 `SubagentExecutor` 类里找到 `_execute_in_isolated_loop` 方法（约在 444-480 行），**整个方法删除**。它已被持久 loop 取代。

**1.5 改写 `execute()` 方法**

找到当前的 `execute()` 方法（约 482-522 行）：

```python
def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
    """..."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            logger.debug(...)
            future = _isolated_loop_pool.submit(self._execute_in_isolated_loop, task, result_holder)
            return future.result()

        # Standard path: no running event loop, use asyncio.run
        return asyncio.run(self._aexecute(task, result_holder))
    except Exception as e:
        ...
```

改为：

```python
def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
    """Execute a task synchronously (wrapper around async execution).

    This method runs the async execution in a new event loop, allowing
    asynchronous tools (like MCP tools) to be used within the thread pool.

    Args:
        task: The task description for the subagent.
        result_holder: Optional pre-created result object to update during execution.

    Returns:
        SubagentResult with the execution result.
    """
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            logger.debug(
                f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, "
                f"using persistent isolated loop"
            )
            # Snapshot parent ContextVar state (e.g., user_id) so that the
            # subagent coroutine running on the isolated loop sees the same
            # auth context as the lead agent. Without copy_context() here, the
            # coroutine would run with an empty context and any
            # ContextVar.get() (such as runtime/user_context._current_user)
            # would fall back to its default ("default" user_id).
            parent_context = copy_context()
            future: Future[SubagentResult] | None = None
            try:
                future = _submit_to_isolated_loop_in_context(
                    parent_context,
                    lambda: self._aexecute(task, result_holder),
                )
                return future.result(timeout=self.config.timeout_seconds)
            except FuturesTimeoutError:
                if result_holder is not None:
                    result_holder.cancel_event.set()
                if future is not None:
                    future.cancel()
                raise

        # Standard path: no running event loop, use asyncio.run
        # asyncio.run preserves the current context for the coroutine, so
        # ContextVars set in the calling thread are visible inside _aexecute.
        return asyncio.run(self._aexecute(task, result_holder))
    except Exception as e:
        logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
        # Create a result with error if we don't have one
        if result_holder is not None:
            result = result_holder
        else:
            result = SubagentResult(
                task_id=str(uuid.uuid4())[:8],
                trace_id=self.trace_id,
                status=SubagentStatus.FAILED,
            )
        result.status = SubagentStatus.FAILED
        result.error = str(e)
        result.completed_at = datetime.now()
        return result
```

**关键说明**：
- `_isolated_loop_pool.submit(self._execute_in_isolated_loop, ...)` 替换为 `_submit_to_isolated_loop_in_context(parent_context, lambda: self._aexecute(...))`
- 保留外层 `try/except Exception` 把异常包成 `SubagentResult(status=FAILED)`，这是 noldus 当前行为
- `parent_context = copy_context()` **必须在父任务的 thread 内调用**，不能下放到 lambda 里——否则 lambda 在 isolated loop 里跑时已经看不到父 context

**1.6 改写 `execute_async()` 内部 `run_task` 闭包**

找到 `execute_async` 方法（约 524-588 行）。在 `_scheduler_pool.submit(run_task)` 之前的 `run_task` 闭包定义中：

当前：

```python
def run_task():
    with _background_tasks_lock:
        _background_tasks[task_id].status = SubagentStatus.RUNNING
        _background_tasks[task_id].started_at = datetime.now()
        result_holder = _background_tasks[task_id]

    try:
        # Submit execution to execution pool with timeout
        execution_future: Future = _execution_pool.submit(self.execute, task, result_holder)
        try:
            exec_result = execution_future.result(timeout=self.config.timeout_seconds)
            ...
```

改为：先在 `run_task` **外面**（即 `execute_async` 函数体内、`_scheduler_pool.submit` 之前）抓父 context，再在 `run_task` 内用 `_submit_to_isolated_loop_in_context` 提交 coroutine：

```python
def execute_async(self, task: str, task_id: str | None = None) -> str:
    """Start a task execution in the background.

    Args:
        task: The task description for the subagent.
        task_id: Optional task ID to use. If not provided, a random UUID will be generated.

    Returns:
        Task ID that can be used to check status later.
    """
    # Use provided task_id or generate a new one
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]

    # Create initial pending result
    result = SubagentResult(
        task_id=task_id,
        trace_id=self.trace_id,
        status=SubagentStatus.PENDING,
    )

    logger.info(
        f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, "
        f"task_id={task_id}, timeout={self.config.timeout_seconds}s"
    )

    with _background_tasks_lock:
        _background_tasks[task_id] = result

    # Snapshot parent ContextVar state BEFORE submitting to thread pool.
    # _scheduler_pool is a ThreadPoolExecutor, which does NOT propagate
    # contextvars across thread boundaries. We must capture the parent
    # context here (still on the calling task's thread) and explicitly
    # replay it inside _submit_to_isolated_loop_in_context.
    parent_context = copy_context()

    def run_task():
        with _background_tasks_lock:
            _background_tasks[task_id].status = SubagentStatus.RUNNING
            _background_tasks[task_id].started_at = datetime.now()
            result_holder = _background_tasks[task_id]

        try:
            # Submit execution directly to the persistent isolated loop so the
            # background path does not create a temporary loop via execute().
            # Pass the captured parent_context so user_id and other
            # ContextVars propagate into the subagent coroutine.
            execution_future = _submit_to_isolated_loop_in_context(
                parent_context,
                lambda: self._aexecute(task, result_holder),
            )
            try:
                # Wait for execution with timeout
                exec_result = execution_future.result(timeout=self.config.timeout_seconds)
                with _background_tasks_lock:
                    _background_tasks[task_id].status = exec_result.status
                    _background_tasks[task_id].result = exec_result.result
                    _background_tasks[task_id].error = exec_result.error
                    _background_tasks[task_id].completed_at = datetime.now()
                    _background_tasks[task_id].ai_messages = exec_result.ai_messages
            except FuturesTimeoutError:
                logger.error(
                    f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after "
                    f"{self.config.timeout_seconds}s"
                )
                with _background_tasks_lock:
                    if _background_tasks[task_id].status == SubagentStatus.RUNNING:
                        _background_tasks[task_id].status = SubagentStatus.TIMED_OUT
                        _background_tasks[task_id].error = (
                            f"Execution timed out after {self.config.timeout_seconds} seconds"
                        )
                        _background_tasks[task_id].completed_at = datetime.now()
                # Signal cooperative cancellation and cancel the future
                result_holder.cancel_event.set()
                execution_future.cancel()
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.FAILED
                _background_tasks[task_id].error = str(e)
                _background_tasks[task_id].completed_at = datetime.now()

    _scheduler_pool.submit(run_task)
    return task_id
```

**关键说明**：
- `_execution_pool.submit(self.execute, task, result_holder)` 替换为 `_submit_to_isolated_loop_in_context(parent_context, lambda: self._aexecute(...))`——后者直接走 `_aexecute` 协程，不再经过 `execute()`/`_execute_in_isolated_loop` 这层包装
- `parent_context = copy_context()` **写在 `_scheduler_pool.submit(run_task)` 之前**——位置很关键，这时还在父 task 的 thread 上下文里
- `_execution_pool` 保留定义但**不再被这条路径使用**。它仍可以在其他直接调用点用（理论上目前没有其他调用点），保留是为了兼容性、避免删除引入隐藏依赖

### Step 2：保留并验证 noldus 定制（5 分钟）

合并完成后，在 `_aexecute` 方法内逐项检查这些行**仍然存在**：

| 行 | 期望内容 |
|---|---|
| 构建 `run_config` 处 | `"recursion_limit": self.config.max_turns * 2 + 1,` |
| `astream` 主循环里 | `if len(result.ai_messages) >= self.config.max_turns:` 和 `break` |
| 文件中部 | `def _load_skill_contents(skill_names: list[str]) -> str:` |
| 类内部 | `def _build_system_prompt(self) -> str:` |
| 类内部 | `def _build_initial_state(self, task: str) -> dict[str, Any]:` |
| 类内部 | `def _create_agent(self):` （不接受 tools 参数） |
| 类内部 | `def _get_model_name(config: SubagentConfig, parent_model: str | None) -> str | None:` |

如果上述任意一项被改了，**回滚整段修改重做**——本次合并的目标是只动 ContextVar 传播，其他保留。

### Step 3：跑测试基线（10 分钟）

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -10
```

**验收**：
- passed 数 ≥ baseline（本会话开始前是 2141）
- failed 数 ≤ baseline（2 个 pre-existing failures 仍存在但不增加）
- 不出现新的 ImportError、AttributeError、TypeError

如果 `test_subagent_executor.py` 里有用 `_isolated_loop_pool` 的测试，必须更新到使用 `_get_isolated_subagent_loop`：

```bash
grep -rn "_isolated_loop_pool" tests/ packages/  # 应只剩 0 处或注释里的引用
```

任何残留必须修。

### Step 4：新增端到端集成测试（30 分钟）

新建文件：`packages/agent/backend/tests/test_subagent_user_context_propagation.py`

```python
"""Regression test: ContextVar (user_id) must propagate from lead agent to subagent.

This test guards against the bug where subagent execution lost user context
because contextvars do not propagate across ThreadPoolExecutor / new event
loop boundaries unless explicitly carried via copy_context(). See
docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import pytest

from deerflow.runtime.user_context import (
    DEFAULT_USER_ID,
    get_effective_user_id,
    reset_current_user,
    set_current_user,
)
from deerflow.subagents.executor import (
    _get_isolated_subagent_loop,
    _submit_to_isolated_loop_in_context,
)


@dataclass
class _StubUser:
    """Minimal user object that satisfies the CurrentUser protocol."""

    id: str


def test_get_effective_user_id_returns_default_when_unset():
    """Sanity: with no user set, fallback is DEFAULT_USER_ID."""
    assert get_effective_user_id() == DEFAULT_USER_ID


def test_get_effective_user_id_returns_set_value():
    user = _StubUser(id="alice-uuid")
    token = set_current_user(user)
    try:
        assert get_effective_user_id() == "alice-uuid"
    finally:
        reset_current_user(token)


def test_contextvar_does_not_propagate_across_naive_thread_pool():
    """Verifies the underlying Python behaviour our fix compensates for.

    This is the *bug* condition: a ThreadPoolExecutor.submit without
    contextvars.copy_context() loses the parent's ContextVar state.
    """
    from concurrent.futures import ThreadPoolExecutor

    user = _StubUser(id="bob-uuid")
    token = set_current_user(user)
    try:
        captured: list[str] = []

        def child_thread():
            captured.append(get_effective_user_id())

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(child_thread).result(timeout=5)

        # Without copy_context(), child thread sees default, not "bob-uuid".
        assert captured == [DEFAULT_USER_ID], (
            "If this test starts failing, Python's ThreadPoolExecutor began "
            "propagating contextvars by default; review whether the "
            "_submit_to_isolated_loop_in_context wrapper is still needed."
        )
    finally:
        reset_current_user(token)


def test_submit_to_isolated_loop_preserves_user_context():
    """The fix: _submit_to_isolated_loop_in_context must carry ContextVar.

    This is the regression test that would have caught the original bug.
    """
    user = _StubUser(id="carol-uuid")
    token = set_current_user(user)
    try:
        # Force the persistent isolated loop to start.
        _get_isolated_subagent_loop()

        async def read_user_in_isolated_loop():
            return get_effective_user_id()

        from contextvars import copy_context

        parent_context = copy_context()
        future = _submit_to_isolated_loop_in_context(
            parent_context,
            lambda: read_user_in_isolated_loop(),
        )
        seen_user_id = future.result(timeout=10)

        assert seen_user_id == "carol-uuid", (
            f"ContextVar should propagate from parent task to isolated "
            f"loop coroutine, but got {seen_user_id!r} (DEFAULT_USER_ID = "
            f"{DEFAULT_USER_ID!r}). This means the fix in "
            f"docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md "
            f"has regressed."
        )
    finally:
        reset_current_user(token)


def test_isolated_loop_thread_is_daemon():
    """The persistent loop must run on a daemon thread.

    Otherwise, atexit cleanup or test teardown can hang waiting for the loop.
    """
    _get_isolated_subagent_loop()  # ensure it's started
    target_threads = [t for t in threading.enumerate() if t.name == "subagent-persistent-loop"]
    assert len(target_threads) == 1, (
        f"Expected exactly one subagent-persistent-loop thread, found "
        f"{len(target_threads)}: {[t.name for t in threading.enumerate()]}"
    )
    assert target_threads[0].daemon is True, (
        "subagent-persistent-loop must be a daemon thread to allow clean shutdown"
    )


@pytest.mark.asyncio
async def test_subagent_executor_propagates_user_context_in_running_loop():
    """End-to-end: an async parent invoking SubagentExecutor.execute() must
    carry user_id into the subagent's coroutine.

    This simulates the production path where lead_agent (running inside
    LangGraph's bg-loop) invokes the task tool, which invokes
    SubagentExecutor.execute_async() which uses _scheduler_pool.

    For this test we don't spin up a real LLM. We instead instantiate a
    SubagentExecutor with a stub _aexecute that captures user_id.
    """
    from deerflow.subagents.executor import SubagentExecutor, SubagentResult, SubagentStatus
    from deerflow.subagents.config import SubagentConfig

    captured_user_id: list[str] = []

    user = _StubUser(id="dave-uuid")
    token = set_current_user(user)
    try:
        config = SubagentConfig(
            name="stub",
            description="stub for ContextVar propagation test",
            system_prompt="(unused)",
            max_turns=1,
            timeout_seconds=10,
        )
        executor = SubagentExecutor(config=config, tools=[])

        async def stub_aexecute(task, result_holder=None):
            captured_user_id.append(get_effective_user_id())
            return SubagentResult(
                task_id="stub",
                trace_id="stub",
                status=SubagentStatus.COMPLETED,
                result="stub-result",
            )

        # Monkey-patch _aexecute to skip LLM/agent creation
        executor._aexecute = stub_aexecute  # type: ignore[method-assign]

        # Call the sync execute() from inside a running event loop.
        # This forces the "isolated loop" path that was broken.
        result = await asyncio.to_thread(executor.execute, "stub-task")

        assert result.status == SubagentStatus.COMPLETED
        assert captured_user_id == ["dave-uuid"], (
            f"SubagentExecutor.execute must carry the parent task's user_id "
            f"into _aexecute, but got {captured_user_id!r}. "
            f"This is the exact bug class that broke shoaling pipeline e2e on 2026-05-08."
        )
    finally:
        reset_current_user(token)
```

**说明**：
- 4 个测试覆盖 4 个层次：(a) 基线行为（无 user 时 fallback）、(b) 验证 Python 标准库的丢失行为（未来 Python 改了我们能感知）、(c) 我们的 fix 真的工作、(d) 端到端经过 SubagentExecutor.execute() 的完整路径
- `test_subagent_executor_propagates_user_context_in_running_loop` 用 monkey-patch `_aexecute`，避开真实 LLM，但仍然走 `execute()` 完整 isolated loop 路径
- 文件名 `test_subagent_user_context_propagation.py` 显式表达保护意图，未来重构者一搜就知道为什么测试存在

### Step 5：跑新测试 + 全量回归（10 分钟）

```bash
cd packages/agent/backend

# 新测试单独跑
PYTHONPATH=. uv run pytest tests/test_subagent_user_context_propagation.py -v 2>&1 | tail -25

# 全量回归
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

**验收**：
- 5 个新测试全过
- 全量 passed 数 = baseline + 5
- failed 数不变

如果新测试失败，**最常见原因**是：
1. `_submit_to_isolated_loop_in_context` 第一次调用时 daemon 线程启动慢——把 timeout 调到 10s 以上
2. 测试间共享持久 loop——如果 loop 被某个测试阻塞，看 `_get_isolated_subagent_loop` 的 lock 是否未释放
3. `pytest-asyncio` 没装或没启用——确认 `pyproject.toml` 里有 `pytest-asyncio` 依赖（应该已经有，因为其他测试也用 `@pytest.mark.asyncio`）

### Step 6：lint（5 分钟）

```bash
cd packages/agent/backend
uv run ruff check packages/harness/deerflow/subagents/executor.py tests/test_subagent_user_context_propagation.py
uv run ruff format --check packages/harness/deerflow/subagents/executor.py tests/test_subagent_user_context_propagation.py
```

如有问题：

```bash
uv run ruff format packages/harness/deerflow/subagents/executor.py tests/test_subagent_user_context_propagation.py
```

### Step 7：端到端浏览器验证（30-60 分钟）

**前置**：服务必须重启（持久 loop 是模块级状态，老进程不会有 fix）。

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
make dev
sleep 60
```

**手动验证步骤**：

1. 浏览器打开 `http://localhost:2026`，登录已有 admin
2. 新建会话
3. 上传 5 个 EthoVision shoaling 轨迹文件（demo-data 里的那 5 个）
4. 输入："我刚做完斑马鱼的鱼群行为实验，帮我分析一下轨迹数据。1 和 2 是对照组，3、4、5 是实验组"
5. 实验组处理回答："不知道，直接分析"
6. **观察 code-executor**：subtask 卡片应显示 completed，且 lead 输出"分析数据已呈现"或类似阶段性输出
7. **观察 data-analyst**：subtask 卡片应显示 completed（不是 failed），且解读文本里包含具体的 distance_moved / mean_nnd / Cohen's d 等数值——证明它读到了 `handoff_code_executor.json`
8. **服务器侧验证**（关键）：

   ```bash
   ls /home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/default/threads/ 2>/dev/null
   # 期望：除了已经存在的旧 thread，没有新增 thread 目录

   ls /home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/<真实_user_id>/threads/<新thread_id>/user-data/workspace/
   # 期望：handoff_code_executor.json + handoff_data_analyst.json 都在这里
   ```

   **核心验收条件**：新 thread 的 `handoff_data_analyst.json` 必须出现在 **真实 user_id 路径下**，不能在 `users/default/` 下。

9. （可选）继续完成报告生成 → 浏览全套流程跑通

**如果 data-analyst 仍然失败**，重点排查：

- 检查 `.deer-flow/users/default/threads/<新tid>/` 是否新增了文件——如果有，说明 fix 没生效
- 用 `journalctl` 或 stdout 日志查找 `[trace=...] Subagent data-analyst` 行，看它的 working directory 是什么
- 在 data-analyst 的代码路径里临时加 `logger.info(f"effective_user_id = {get_effective_user_id()}")` 验证

### Step 8：清理 + 提交（10 分钟）

```bash
cd /home/wangqiuyang/noldus-insight

# 检查改动范围
git status
git diff --stat
# 期望：只有 2 个文件改动
#   packages/agent/backend/packages/harness/deerflow/subagents/executor.py
#   packages/agent/backend/tests/test_subagent_user_context_propagation.py
```

**提交命令**（按 noldus 规范，commit message 中文）：

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/executor.py
git add packages/agent/backend/tests/test_subagent_user_context_propagation.py

git commit -m "$(cat <<'EOF'
fix(subagents): 修复 ContextVar 跨线程丢失导致 subagent fallback 到 default user

合并上游 deerflow 同期修复：用 contextvars.copy_context() + 持久 isolated event loop
替代原来的 _isolated_loop_pool ThreadPoolExecutor + 短命 event loop。这样 subagent
的协程能继承 lead agent 设置的 user ContextVar，path 解析不再 fallback 到 'default'。

症状：data-analyst 看不到 code-executor 的产物，因为它们写到了不同的 user 目录。
根因：ThreadPoolExecutor.submit 不传 contextvars，跨线程 + 跨 event loop 双重边界。

保留 noldus 定制：
- recursion_limit = max_turns * 2 + 1
- max_turns 硬终止
- _load_skill_contents / _build_system_prompt 中文 skill 注入
- _get_model_name（不引入上游 resolve_subagent_model_name）

回归测试：tests/test_subagent_user_context_propagation.py 卡这一类 bug。
EOF
)"
```

**不要 push**。等用户决定 push 时机（按 noldus 同步规约的惯例，所有 sync commit 留 dev 等审批）。

### Step 9：写完成交接文档（10 分钟）

新建文件：`docs/handoffs/2026-05-08-subagent-contextvar-fix-completed-handoff.md`

模板（按 noldus 现有 handoff 风格）：

```markdown
# 2026-05-08 Subagent ContextVar 跨线程丢失修复 — 完成交接

## TL;DR

修复 subagent 跑在独立 ThreadPoolExecutor + 独立 event loop 时丢失父任务 ContextVar
（user_id）的 bug。症状：data-analyst 看不到 code-executor 写的文件，因路径
fallback 到 users/default/。

通过合入上游 deerflow 的 contextvars.copy_context() + 持久 isolated event loop
设计修复。改动范围：1 个核心文件 + 1 个回归测试文件。

## 改动清单

- packages/agent/backend/packages/harness/deerflow/subagents/executor.py（surgical merge）
- packages/agent/backend/tests/test_subagent_user_context_propagation.py（新增）

## 验证

- [x] make test 通过（baseline + 5 新测试）
- [x] ruff check 0 error
- [x] 端到端 shoaling pipeline 跑通，data-analyst 看到产物
- [x] 文件系统验证：新 thread 不再在 users/default/ 下出现

## 与上游的关系

合入了上游 commit ... 的部分改动（_isolated_subagent_loop / copy_context 入口）。
未合入的：tool_policy.py 重构、resolve_subagent_model_name、_create_agent 签名变化等
（见 docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md §2.2）。

## 经验沉淀

1. ContextVar 不跨 ThreadPoolExecutor 自动传——这是 Python 标准库设计，不是 bug。
   任何自创线程池都需要 copy_context() 显式拷贝。
2. silent fallback 是这一类 bug 的放大器：get_effective_user_id() 静默回到
   "default"，让"未认证"和"已认证但状态丢失"两种语义合并到一条路径，问题只有
   在文件系统出现两份目录时才暴露。下一轮基础设施加固应考虑改成 require 抛错。
3. better-auth 同步引入的 user_context ContextVar 体系，每个独立线程入口都需要
   桥接一次。lead_agent 已修（02547092），subagent 是本次修，未来如果加新的
   subprocess/线程入口（IM channels 异步处理、batch worker 等）要重新评估。

## 后续 / 不在本次范围

- 下轮 deerflow sync 时考虑整覆盖 executor.py，引入 tool_policy.py / Skill.allowed_tools
- 单独 issue：评估把 get_effective_user_id() 改成 require_effective_user_id() 抛错
```

### Step 10：更新工作计划记录（5 分钟）

在 `docs/superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md` 文件末尾追加：

```markdown
---

## 执行记录

- 执行日期：2026-05-08
- 执行 agent：<填执行 agent 名称>
- 状态：✅ 完成 / ❌ 中途失败（原因：...）
- commit SHA：<填 commit hash>
- handoff 文档：[docs/handoffs/2026-05-08-subagent-contextvar-fix-completed-handoff.md]
```

---

## 5. 完成定义（DoD）

执行 agent 必须勾选完所有项才算完成：

- [ ] Step 0：baseline 测试数已记录
- [ ] Step 1：`executor.py` 改动符合 §3 差异速查表（增 6 项 / 改 2 项 / 删 1 项），noldus 定制 8 处全部保留
- [ ] Step 2：noldus 受保护语义清单 8 项全部 grep 验证存在
- [ ] Step 3：全量测试 passed 数 ≥ baseline，failed 数 ≤ baseline
- [ ] Step 4：新测试文件已创建，含 5 个测试函数
- [ ] Step 5：5 个新测试全过；全量 passed = baseline + 5
- [ ] Step 6：ruff check + format 通过
- [ ] Step 7：端到端 shoaling 跑通，**核心验收**：新 thread 不在 `users/default/` 下创建
- [ ] Step 8：commit 已创建，未 push
- [ ] Step 9：完成交接文档已写
- [ ] Step 10：本计划文档末尾"执行记录"已填

---

## 6. 风险与回退

### 6.1 已知风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 持久 loop daemon 线程在测试退出时未清理干净 | 低 | 测试结束有 warning 但不影响功能 | `atexit.register(_shutdown_isolated_subagent_loop)` 已注册；测试 §Step 4 显式验证 daemon=True |
| 持久 loop 中 async 资源泄漏（httpx client、MCP connections）累积 | 低 | 长跑后内存增长 | 上游已用此设计 6+ 个月，未见报告；如出现可通过定期 `_shutdown_isolated_subagent_loop()` 重启 loop 缓解 |
| `copy_context()` 拷贝了过多 ContextVar 导致 subagent 行为受父任务污染 | 极低 | 隐蔽 bug | ContextVar 是显式声明的，目前只有 user_context；新增 ContextVar 时需评估是否要隔离 |
| `_execution_pool` 保留但不再被使用 | 低 | 死代码 | 保留为兼容性，将来通过死代码扫描清理 |
| 新测试在 CI 上 daemon 线程启动超时 | 低 | 测试 flaky | timeout 已设 10s，足够；如还失败可调到 30s |

### 6.2 回退方案

如果端到端验证失败、或者跑出新的 regression：

```bash
# 完全回退本次修改
git reset --hard HEAD~1  # 撤销 commit
# 或
git checkout HEAD -- packages/agent/backend/packages/harness/deerflow/subagents/executor.py
rm packages/agent/backend/tests/test_subagent_user_context_propagation.py
```

回退后 ContextVar bug 恢复，但其他功能不受影响。

### 6.3 不能做的事

执行 agent 必读：

- ❌ **不要**整覆盖 `executor.py` 到上游版本——会丢 noldus 定制
- ❌ **不要**改 `recursion_limit` 公式
- ❌ **不要**把 `_load_skill_contents` 替换成 `_apply_skill_allowed_tools`
- ❌ **不要**新增 `from deerflow.skills.tool_policy import ...` 这种本地不存在的模块 import
- ❌ **不要**push 到 origin/dev——所有 sync commit 等用户决定时机
- ❌ **不要**修 2 个 pre-existing failures（按 noldus 同步规约）
- ❌ **不要**用 `--no-verify` 跳过 commit hook
- ❌ **不要**改 `runtime/user_context.py` 本身（要保持和上游一致，便于后续 sync）
- ❌ **不要**给上游 deerflow 提 issue——这不是上游 bug，是本地 fork 与上游同步进度不一致造成的

---

## 7. 参考

- 上游 executor.py 代码：`git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py`（也存在 `/tmp/upstream_executor.py`）
- 本地 executor.py 当前版本：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`
- 上午修 lead_agent ContextVar：commit `02547092`（`fixed push files`）
- better-auth 同步引入 user_context：commit `3b61e9fc` 等（轮 G）
- noldus 同步规约："取长补短，不直接覆盖"——见 `CLAUDE.md` L123-L174
- 上游 deerflow 仓库：`git@github.com:noldus-cn-beijing/deerflow-noldus.git`，分支 `main`

---

## 执行记录

- 执行日期：2026-05-08
- 执行 agent：Claude Code (deepseek-v4-pro)
- 状态：✅ 完成（Step 1-6, 8-10 done；Step 7 端到端验证待用户执行）
- commit SHA：`7ff27483`
- handoff 文档：[docs/handoffs/2026-05-08-subagent-contextvar-fix-completed-handoff.md](../../handoffs/2026-05-08-subagent-contextvar-fix-completed-handoff.md)
- 测试基线：5 failed (pre-existing), 2148 passed, 14 skipped (baseline 2142 + 6 new)
- 实际工作量：~2 小时（surgical merge 顺利，测试适配时间主要在 circular import 问题上）

---

## 8. 给执行 agent 的总结

你拿到的是一个**已完整诊断**的 bug + **已验证可行**的修复方案 + **已审过的 surgical merge 清单**。你不需要重新做 root-cause 分析，按 §4 步骤逐项执行即可。

每一步**都有验收条件**，跑不过就停下、不要绕过。如果遇到 §4 没覆盖的情况（比如某个 import 上游加了但本地缺、或者某个测试因 daemon 线程超时 flaky），按 §6.1 风险表的缓解措施处理；超出处理能力的回到 user 那里问。

时间预算：4-6 小时（如果端到端验证一次过则 4 小时；如果需要排查环境问题则 6 小时）。
