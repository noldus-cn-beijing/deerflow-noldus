# 2026-05-18 DeerFlow Sync PENDING 4 Commit Surgical Merge Handoff

> **目标**：完成第 3 轮 deerflow 上游同步的 4 个 PENDING commit（A-8~A-11）的 surgical merge。
>
> **状态**：第 3 轮 14 个 commit 已合，4 个 PENDING、4 个 BLOCKED、3 个永久 SKIP、9 个推迟 Plan B。
>
> **预读**：本文件 + [2026-05-18 deerflow sync 进度记录](2026-05-18-deerflow-sync-progress.md)（上一 batch 完整记录）。

---

## 1. 背景

第 3 轮 sync 合并了 14 个上游 commit（`272c7570`），剩余的 4 个 PENDING commit：

| Task | Upstream Commit | 摘要 |
|---|---|---|
| A-8 | `9892a7d4` | subagent token bucketing into parent run totals |
| A-9 | `eab7ae3d` | stream subagent token usage to header via terminal task events |
| A-10 | `813d3c94` | consolidate system_prompt + skills into single SystemMessage |
| A-11 | `7de9b582` | Runtime type alias to eliminate Pydantic serialization warning |

**A-9 依赖 A-8**（token 缓存机制基于 A-8 的 collector），必须顺序做：A-8 → A-9。

A-10 和 A-11 独立，可任意顺序。

---

## 2. 受保护文件速查

| 本地文件 | 行数 | Noldus 定制内容 | 涉及的 PENDING |
|---|---|---|---|
| `task_tool.py` | 314 | `{{handoff://}}` 占位符系统 + `HANDOFF_FILE_REGISTRY` + `HandoffIsolationProvider` 集成 | A-8, A-9, A-11 |
| `executor.py` | 797 | `recursion_limit` 修复 + `max_turns` 硬限制 + `HandoffIsolationProvider` guardrail 挂载 | A-8, A-10 |
| `sandbox/tools.py` | 1658 | `{{shared://}}` 占位符替换 + `extra_env` 参数 + `DEERFLOW_PATH_*` 环境变量 | A-11 |

**铁律**：这三个文件**绝不整文件 cp**。只做 surgical merge——手工合入上游纯 bug fix/改进行，保留所有 Noldus 定制。

---

## 3. A-8：subagent token bucketing（依赖 0）

### 3.1 要干什么

上游给 subagent 执行加了 token collector，把 subagent 的 LLM 调用 token 归并到父 RunJournal。

### 3.2 安全文件（整文件 cp）

| 文件 | 操作 |
|---|---|
| `runtime/journal.py` | **整文件 cp**（本地无 Noldus 定制）|
| `subagents/token_collector.py` | **新建**（复制上游新文件）|
| `backend/tests/test_run_journal.py` | **整文件 cp** |
| `backend/tests/test_subagent_token_collector.py` | **新建** |
| `backend/tests/test_task_tool_core_logic.py` | **整文件 cp** |
| `frontend/tests/unit/core/threads/api.test.ts` | **整文件 cp** |

### 3.3 受保护文件 surgical merge

**executor.py**（Noldus 定制：recursion_limit + max_turns + handoff guardrail）

上游只加了几行，非常局部。手动合入以下内容：

1. 在 import 段加：
```python
from deerflow.subagents.token_collector import SubagentTokenCollector
```

2. 在 `SubagentResult` dataclass 已有的 `cancel_event` 字段后加：
```python
token_usage_records: list[dict[str, int | str]] = field(default_factory=list)
usage_reported: bool = False
```

3. 在 `_execute_async` 方法中，`_build_initial_state` 之后、agent creation 之前加：
```python
collector: SubagentTokenCollector | None = None
```

4. 在 `_create_agent(filtered_tools)` 之后、`run_config` 赋值处加 `callbacks` + `tags`：
```python
collector_caller = f"subagent:{self.config.name}"
collector = SubagentTokenCollector(caller=collector_caller)

run_config: RunnableConfig = {
    "recursion_limit": self.config.max_turns,
    "callbacks": [collector],
    "tags": [collector_caller],
}
```
注意：保留本地 `recursion_limit` 赋值方式（Noldus 定制），只新增 `callbacks` + `tags` 两行。

5. 在 `_execute_async` 中每个 `result.status = SubagentStatus.CANCELLED` / `result.error = ...` 分支（共 2 处）之前加：
```python
if collector is not None:
    result.token_usage_records = collector.snapshot_records()
```

6. 在 completed 路径（`Subagent {name} completed async execution` 日志后）加：
```python
result.token_usage_records = collector.snapshot_records()
```

7. 在 except 兜底（`result.status = SubagentStatus.FAILED`）后加：
```python
if collector is not None:
    result.token_usage_records = collector.snapshot_records()
```

**task_tool.py**（Noldus 定制：handoff 占位符系统 + HandoffIsolationProvider）

上游这段改动**跟 handoff 系统不冲突**——新加的函数（`_is_subagent_terminal`, `_await_subagent_terminal`, `_deferred_cleanup_subagent_task`, `_log_cleanup_failure`, `_schedule_deferred_subagent_cleanup`, `_find_usage_recorder`, `_report_subagent_usage`）是纯粹的 token usage 和 deferred cleanup 逻辑，不接触 handoff 代码路径。手动合入：

1. 在现有 import 段之后加所有新函数（`_is_subagent_terminal` 到 `_report_subagent_usage` 全部）。直接从上流 diff 复制粘贴。

2. 修改 `task_tool` 函数内部的 polling 循环：每个 `result.status == SubagentStatus.XXX` 分支**开头**加 `_report_subagent_usage(runtime, result)`。本地这 4 个分支在 handoff 阶段加过修改——确认每个分支里的 `return f"Task..."` 语句**前面**加上 `_report_subagent_usage`。

3. 修改 `asyncio.CancelledError` 处理器：用上游新写的 `_await_subagent_terminal` + `_report_subagent_usage` 逻辑替代现有的 inline cleanup。关键——本地的 CancelledError 处理器在 sync 第 3 轮可能已有 Noldus 改动，merge 时只替换 deferred cleanup 逻辑部分。

### 3.4 验证

```bash
cd packages/agent/backend && make test
cd packages/agent/frontend && pnpm typecheck
```

---

## 4. A-9：stream subagent usage to header（依赖 A-8）

### 4.1 安全文件（整文件 cp）

| 文件 | 操作 |
|---|---|
| `agents/middlewares/token_usage_middleware.py` | **整文件 cp**（本地无 Noldus 定制）|
| `backend/tests/test_memory_queue_user_isolation.py` | **整文件 cp**（5 行）|
| `backend/tests/test_task_tool_core_logic.py` | **整文件 cp** |
| `backend/tests/test_token_usage_middleware.py` | **整文件 cp** |
| `frontend/src/workspace/messages/message-token-usage.tsx` | **整文件 cp** |
| `frontend/src/core/messages/usage.ts` | **整文件 cp** |
| `frontend/src/core/threads/hooks.ts` | ⚠️ **检查**：本地 hooks.ts drift 大（803 行），看看上游改什么再定 |
| README.md, backend/CLAUDE.md | **整文件 cp** |

### 4.2 受保护文件 surgical merge

**task_tool.py**（Noldus 定制：handoff 系统）

上游改动不接触 handoff 代码路径。手动合入：

1. 在 import 段之后、handoff registry 之前加：
```python
# Cache subagent token usage by tool_call_id so TokenUsageMiddleware can
# write it back to the triggering AIMessage's usage_metadata.
_subagent_usage_cache: dict[str, dict[str, int]] = {}
```

2. 加四个新函数：`_token_usage_cache_enabled`, `_cache_subagent_usage`, `pop_cached_subagent_usage`, `_summarize_usage`。

3. 在 `task_tool` 函数：
   - 开头加 `cache_token_usage = _token_usage_cache_enabled(runtime_app_config)`
   - 每个 poll status branch 的 `writer(...)` 调用前先调 `_cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)`
   - 每个 `writer(...)` 调用加 `"usage": usage` 字段
   - CancelledError 处理器末尾加 `_subagent_usage_cache.pop(tool_call_id, None)`
   - 新增 except Exception 兜底：`_subagent_usage_cache.pop(tool_call_id, None); raise`

---

## 5. A-10：consolidate system_prompt + skills（独立）

### 5.1 安全文件

| 文件 | 操作 |
|---|---|
| `subagents/config.py` | **整文件 cp**（只 1 行改：`system_prompt: str` → `str | None`）|
| `backend/tests/test_subagent_executor.py` | **整文件 cp** |

### 5.2 受保护文件 surgical merge

**executor.py**（Noldus 定制：recursion_limit + max_turns + handoff guardrail）

改动非常小：

1. `_create_agent` 方法中，`system_prompt` 参数值从 `self.config.system_prompt` 改为 `None`。

2. `_build_initial_state` 方法中，原来 `skill_messages` 的 extend 逻辑替换为将 `system_prompt` + `skill_messages` 合并为一个 `SystemMessage`：
```python
system_parts: list[str] = []
if self.config.system_prompt:
    system_parts.append(self.config.system_prompt)
for skill_msg in skill_messages:
    system_parts.append(skill_msg.content)

messages: list[Any] = []
if system_parts:
    messages.append(SystemMessage(content="\n\n".join(system_parts)))
messages.append(HumanMessage(content=task))
```

注意：本地的 `_build_initial_state` 和上游已经接近（第 3 轮已做部分 merge），但这段要手工确认不破坏 handoff guardrail 初始化逻辑。

---

## 6. A-11：Runtime type alias（独立）

### 6.1 安全文件/新建

| 文件 | 操作 |
|---|---|
| `tools/types.py` | **新建**（复制上游新文件）|
| `tools/builtins/present_file_tool.py` | **整文件 cp** |
| `tools/builtins/setup_agent_tool.py` | **整文件 cp** |
| `tools/builtins/update_agent_tool.py` | **整文件 cp** |
| `tools/builtins/view_image_tool.py` | **整文件 cp** |
| `tools/skill_manage_tool.py` | **整文件 cp** |
| `backend/tests/test_tool_args_schema_no_pydantic_warning.py` | **新建** |

### 6.2 受保护文件 surgical merge

**sandbox/tools.py**（Noldus 定制：`{{shared://}}` + `extra_env` + `DEERFLOW_PATH_*`）

这是**纯类型注解改动**——把 `ToolRuntime[ContextT, ThreadState]` 换成 `Runtime`。Noldus 的所有定制在函数体内部（shared 占位符、extra_env、DEERFLOW_PATH），不在类型签名。

手动合入方案（最安全）：
1. 加 import：`from deerflow.tools.types import Runtime`
2. 删 import：`ContextT`, `ToolRuntime`（如果它们只用于类型注解且不再被引用）
3. 逐个函数签名替换：`ToolRuntime[ContextT, ThreadState]` → `Runtime`
   - `_sanitize_error`
   - `get_thread_data`
   - `is_local_sandbox`
   - `sandbox_from_runtime`
   - `ensure_sandbox_initialized`
   - `ensure_thread_directories_exist`
   - `bash_tool`
   - `ls_tool`
   - `glob_tool`
   - `grep_tool`
   - `read_file_tool`
   - `write_file_tool`
   - `str_replace_tool`

**task_tool.py**：同样纯类型——`ToolRuntime[ContextT, ThreadState]` → `Runtime`。只改 import 和函数签名，不碰函数体。

---

## 7. 执行顺序

```
A-8 → A-9 （A-9 依赖 A-8 的 collector + _report_subagent_usage）
A-10     （独立）
A-11     （独立，但改 sandbox/tools.py + task_tool.py 类型标注，注意和 A-8/A-9 的 task_tool.py 冲突——做完 A-8/A-9 再做 A-11）
```

**推荐**：A-8 → A-9 → A-11 → A-10。这样 task_tool.py 先合功能再合类型。

---

## 8. 验证 Gate

每个 commit 完成后：
```bash
cd packages/agent/backend && make lint && make test
```

全部完成后：
```bash
cd packages/agent/backend && make test  # 确认无新增失败
cd packages/agent/frontend && pnpm typecheck && pnpm test
```

最终 dogfood：
```bash
cd packages/agent && make dev
# 上传 EPM 数据 → 端到端分析 → 确认 4 subagent 正常 + handoff 占位符工作
```

---

## 9. 不要做的事

- ❌ 不整文件 cp 到 task_tool.py / executor.py / sandbox/tools.py
- ❌ 不删 `{{handoff://}}` 占位符系统
- ❌ 不删 `{{shared://}}` 替换逻辑
- ❌ 不删 `recursion_limit` / `max_turns` / `extra_env` / `DEERFLOW_PATH_*`
- ❌ 不碰 handoff_isolation_provider.py
- ❌ A-8 和 A-9 不能并行（有数据依赖）
- ❌ 不改 frontend hooks.ts（本地 drift 803 行，留给 frontend 团队）

---

## 10. 和 subagent-role-split spec 的关系

这两个任务是并行的，互不冲突：
- sync PENDING 改的是 deerflow harness 基础设施（token 归并 + type alias + system_prompt merge）→ 所有 agent 都受益
- subagent-role-split spec 改的是 lead prompt + subagent 配置 + skill 结构 → 行为学分析逻辑

但 A-10（system_prompt merge）和 A-11（Runtime type alias）对 subagent-role-split 的实施有正向影响——改了 `SubagentConfig.system_prompt` 类型（`str → str | None`），恰好兼容 Q6 的 `SubagentConfig` 扩展。
