# 上游 Issue 草稿（中文）

**目标仓库**: https://github.com/bytedance/deer-flow/issues/new/choose
**模板**: Runtime Information

---

## Title

```
[runtime] memory updater 触发 RuntimeError: Event loop is closed（langchain provider 全局 client 缓存被跨 loop 复用）
```

## Problem summary

memory updater 通过 `asyncio.run(coro)` 在 daemon thread 中执行 LLM 调用，每次创建/销毁短命 event loop。
langchain provider（如 langchain-anthropic）使用 `@lru_cache` 全局缓存底层 httpx `AsyncClient`，
导致绑定到已关闭 loop 的 transport 留在共享连接池中，
后续主流程（lead agent 流式 LLM 请求）随机复用到这些"僵尸连接"时，
在 httpx/anyio 的连接清理阶段触发 `RuntimeError: Event loop is closed`。

注：本问题与 #2302 / 已有 subagent 路径 issue 同根，但**触发面更广**——memory updater 默认开启，触发频率比 subagent 更高。`BG_JOB_ISOLATED_LOOPS=true` 不能解决（框架级隔离不覆盖 provider 级 lru_cache 全局共享）。

## Expected behavior

memory updater 在 daemon thread 完成 LLM 调用并销毁其短命 event loop 后，主流程后续的 LLM 调用应能正常完成，不应在 httpx 连接清理阶段抛出 `RuntimeError: Event loop is closed`。

## Actual behavior

主 lead agent run 的流式 LLM 调用（节点 `model`）在 anthropic SDK 的 streaming aclose 阶段抛错：

```
RuntimeError: Event loop is closed
```

完整堆栈（节选）：

```
File ".../langchain/agents/factory.py", line 1156, in _execute_model_async
    output = await model_.ainvoke(messages)
File ".../langchain_anthropic/chat_models.py", line 1352, in _astream
    async for event in stream:
File ".../anthropic/_streaming.py", line 194, in _iter_events
    async for sse in self._decoder.aiter_bytes(self.response.aiter_bytes()):
File ".../httpx/_models.py", line 1063, in aiter_raw
    await self.aclose()
File ".../httpx/_transports/default.py", line 276, in aclose
    await self._httpcore_stream.aclose()
File ".../anyio/_backends/_asyncio.py", line 1329, in aclose
    self._transport.close()
File "/usr/lib/python3.12/asyncio/selector_events.py", line 875, in close
    self._loop.call_soon(self._call_connection_lost, None)
File "/usr/lib/python3.12/asyncio/base_events.py", line 541, in _check_closed
    raise RuntimeError('Event loop is closed')
```

错误堆栈完全指向 lead agent 主流程，不出现 memory 相关帧——但**真正闯祸的是上一次 memory update**留下的 transport 引用已关闭 loop。

## 根因分析（Root cause）

三层独立合理的设计叠加构成 race：

### 1. langchain provider 的全局 client 缓存

[`langchain_anthropic._client_utils._get_default_async_httpx_client`](https://github.com/langchain-ai/langchain-anthropic/blob/main/libs/anthropic/langchain_anthropic/_client_utils.py) 用 `@lru_cache` 全局缓存 `_AsyncHttpxClientWrapper`：

```python
@lru_cache
def _get_default_async_httpx_client(...) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(...)
```

进程内**所有 ChatAnthropic 实例共享同一个 httpx.AsyncClient + 同一个连接池**。

### 2. httpx 连接的 loop 亲和性

httpx AsyncClient 内部用 anyio 包裹 SSL transport。一旦在 loop A 上建立 SSL 连接，`transport._loop` 就绑定 loop A——连接对象**不能跨 loop 安全复用**，loop A 关闭后这个连接成为"僵尸"。

### 3. deerflow memory updater 的短命 loop 模式

[`agents/memory/updater.py::_run_async_update_sync`](https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/memory/updater.py) 使用 `asyncio.run(coro)`：

```python
def _run_async_update_sync(coro: Awaitable[bool]) -> bool:
    ...
    return asyncio.run(coro)  # 创建 loop → 跑 coro → 销毁 loop
```

每次 memory update 都创建并销毁一个 event loop。

### 三层叠加触发

```
T0  lead agent 在 loop_main 上调 LLM
    → langchain_anthropic 创建 AsyncClient（缓存到 lru_cache）
    → 在 loop_main 建立 SSL 连接 conn_1，进入连接池

T1  memory queue 触发，daemon thread 中 asyncio.run(coro) 创建 loop_mem
    → 复用 cached AsyncClient，在 loop_mem 上建立新连接 conn_2
    → memory update 完成，loop_mem 关闭
    → conn_2 留在池中，但 loop_mem 已死

T2  下一个 lead run 在 loop_next 上调 LLM
    → 复用 cached AsyncClient
    → 从池中取出 conn_2
    → HTTP 流读完，aclose() 调用 transport.close() → self._loop = loop_mem
    → loop_mem.call_soon(...) → ❌ RuntimeError: Event loop is closed
```

**关键：错误抛在 lead run（T2）但污染源是 memory update（T1），堆栈完全无法定位**。

## 复现路径

1. 配置 `memory.enabled: true`（默认开启）
2. 使用 anthropic 模型（或任何 langchain provider 内部使用 `@lru_cache` 缓存 async http client 的 provider）
3. 触发若干次连续对话使 memory queue 处理 → 等下一次 lead run streaming LLM
4. 在多次试验中可观察到偶发 `Event loop is closed`

触发依赖时序，不是 100% 必现，但在交互密集的 E2E 测试中可稳定复现。

## 已尝试方案

| 方案 | 是否解决 |
|---|---|
| 设置 `BG_JOB_ISOLATED_LOOPS=true` | ❌ 仅隔离 langgraph 框架的 background runs，不影响 deerflow 自己起的 daemon thread；memory updater 仍然走 `asyncio.run` 短命 loop 模式 |
| 在 `llm_error_handling_middleware` 重试列表中添加 `ReadError` | ❌ 重试不能修复僵尸连接（重试时仍可能再次拿到污染连接） |
| **memory update 完成后清空 langchain provider 的 client lru_cache** | ✅ 已在我们的 fork 中验证有效 |

## 建议修法（按改动大小排序）

### 选项 A：memory updater 跑完清 lru_cache（最小改动，已验证有效）

在 `_run_async_update_sync` 的 finally 中调用：

```python
from langchain_anthropic._client_utils import (
    _get_default_async_httpx_client,
    _get_default_httpx_client,
)
_get_default_async_httpx_client.cache_clear()
_get_default_httpx_client.cache_clear()
```

代价：每次 memory update 后第一个 LLM 调用要重做 SSL 握手（约 100-300ms）。memory update 默认 30s debounce，影响可忽略。

局限：依赖 langchain-anthropic 私有模块（`_client_utils`），且只覆盖 anthropic 一家——其他 provider（如 OpenAI、DeepSeek）若有相同 lru_cache 设计需各自处理。

### 选项 B：memory updater 改用长生命周期 loop（根治）

memory queue 的 daemon thread 启动时 `loop = asyncio.new_event_loop()`，所有 update 在同一个 loop 上 `loop.run_until_complete(...)`，永不销毁。彻底消除"短命 loop"模式。

代价：架构改动较大，需要妥善处理 thread 退出时的 loop 清理。

### 选项 C：memory updater 改用同步 LLM 调用

将 `model.ainvoke()` 改为 `model.invoke()`。memory updater 本就在自己的 thread 同步运行，无需 async。

代价：需确认 ChatAnthropic 同步路径覆盖所有特性（cache control、thinking 等），且失去与 async LLM 的统一性。

## 影响范围

- **必现条件**：`memory.enabled=true` + 使用任何 langchain provider 缓存 async client 的模型 provider + 持续多轮对话
- **症状**：lead agent 流式 LLM 调用偶发 `Event loop is closed`，重试可能成功（如果碰巧拿到干净连接）但用户体验受影响
- **配置默认**：memory 在 `config.yaml` 默认启用，受影响用户面较大

## Operating system

Linux

## Platform details

x86_64, bash, Ubuntu

## Python version

Python 3.12.3

## 关键依赖版本

- langgraph-api: 0.7.65
- langgraph: 1.0.9
- langchain: 1.2.3
- langchain-core: 1.2.17
- langchain-anthropic: 1.3.4
- anthropic: 0.84.0
- httpx: 0.28.1

## 相关 issue

- #2302（subagent 路径同根问题）
- 与 [LangGraph #4218](https://github.com/langchain-ai/langgraph/issues/4218) 间接相关（共用 anyio + httpx 异步基础设施）
- LiteLLM 团队也遇到过类似的 httpx client cache eviction 问题，[事后分析](https://docs.litellm.ai/blog/httpx-cache-eviction-incident) 可参考
