# 上传文件 Agent 看不到 — 跨线程 ContextVar 修复 Handoff

**日期**：2026-05-08
**状态**：✅ 已修复并 e2e 验证通过
**关联前置 handoff**：
- [2026-05-07-upload-file-not-recognized-fix-handoff.md](2026-05-07-upload-file-not-recognized-fix-handoff.md)（同一 bug，前一轮诊断和无效修复尝试）
- [2026-05-08-round3-handoff-pre-e2e-verification.md](2026-05-08-round3-handoff-pre-e2e-verification.md)（Tier 4 Round 3 完成态，本次修复在此基础上）

---

## 一、问题陈述

用户上传 EthoVision XT 数据文件成功（Gateway 200），但 Agent 回答时声称没有看到任何文件。

**实际表现**：

文件系统中同一 thread 同时存在两个目录：

```
users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/<tid>/user-data/uploads/  ← 有文件（Gateway 写入）
users/default/threads/<tid>/user-data/                                       ← 空（Agent 读取这里）
```

Gateway 用真实 user_id 写入，Agent 在 bg-loop 上 fallback 到 `"default"`。

---

## 二、根源（一句话）

**LangGraph 已经把 user_id 端到端传到 bg-loop 了，但 deerflow 没在 bg-loop 上把它从 `config.configurable` 拷进自己的 ContextVar，导致 `get_effective_user_id()` 永远 fallback 到 `"default"`。**

详细数据流：

```
请求 (MainThread)                              bg-loop-1_0 (agent 执行)
─────────────────────                          ─────────────────────────
authenticate()                                 make_lead_agent(config)
  decode JWT → user.id = cd95effa-...            cfg["langgraph_auth_user_id"]
  set_current_user(user)  ❌ 错的线程            ✅ 在这里就能拿到
                                                  (LangGraph 自动塞进来的)

                                               UploadsMiddleware.before_agent()
                                                 get_effective_user_id()
                                                 → ContextVar 读不到 → "default"
                                                 → 路径拼成 users/default/... ❌
```

**关键认识**：Python ContextVar 是 **task-local** 的（asyncio 任务局部）。`authenticate()` 跑在 MainThread/ThreadPoolExecutor 的请求处理任务里，bg-loop-1_0 上的 agent 执行是另一个独立 asyncio task，ContextVar 跨不过去。

**LangGraph 的官方传递通道**：

[`langgraph_api/models/run.py:253`](packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph_api/models/run.py#L253) 在创建 run 时就把 `langgraph_auth_user_id` 写进 `configurable` 并持久化到 run kwargs，worker 在 bg-loop 上反序列化时它还在（[worker.py:68](packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph_api/worker.py#L68)）。这才是设计好的跨线程通道。

---

## 三、为什么前一轮修复跑偏

[2026-05-07 那轮 handoff](2026-05-07-upload-file-not-recognized-fix-handoff.md) 试过两个方向，**都跑在错的线程上**：

1. 在 `langgraph_auth.py` 的 `authenticate()` 里加 `set_current_user(user)`
   - 跑在 ThreadPoolExecutor-1_0（请求处理线程）
   - bg-loop-1_0 上的 agent 看不到

2. 在 `@auth.on add_owner_filter` 里也加 `set_current_user(user)`
   - 同样跑在请求处理线程，不是 agent 执行线程
   - 一样无效

诊断方向错了：以为是「认证没传过去」，**实际上 LangGraph 已经传过去了**，只是 deerflow 没在 bg-loop 上从 `configurable` 读出来。

---

## 四、修复方案（最小改动）

3 处编辑 + 1 个新单测文件：

### 1. `make_lead_agent` 开头加 ContextVar 拷贝

[`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py)

```python
from deerflow.runtime.user_context import set_current_user


class _AuthUser:
    """Minimal duck-typed user satisfying the CurrentUser Protocol."""
    __slots__ = ("id",)
    def __init__(self, user_id: str) -> None:
        self.id = user_id


def make_lead_agent(config: RunnableConfig):
    ...
    cfg = config.get("configurable", {})

    # Copy LangGraph's auth user_id into the deerflow ContextVar.
    # make_lead_agent runs on the bg-loop worker task that subsequently
    # invokes the middlewares (UploadsMiddleware, ThreadDataMiddleware, …),
    # so a ContextVar set here is task-local and visible to every middleware
    # in this run.
    auth_user_id = cfg.get("langgraph_auth_user_id")
    if auth_user_id:
        set_current_user(_AuthUser(str(auth_user_id)))

    ...
```

**为什么有效**：
- `make_lead_agent` 在 bg-loop-1_0 上跑（log 已确认，line 92：`thread_name=bg-loop-1_0`）
- 跟所有 middleware 共享同一个 asyncio task context
- asyncio 的 ContextVar.set 是 task-local，每个 run 自动隔离，不会跨用户污染

### 2. 清理 `langgraph_auth.py` 里没用的 set_current_user

[`packages/agent/backend/app/gateway/langgraph_auth.py`](packages/agent/backend/app/gateway/langgraph_auth.py)

删除 `authenticate()` 和 `@auth.on` 里的 `set_current_user(...)` 调用（连同 `_AuthUser` 内部类）。它们在错的线程上跑，本来就没用，留着会误导后续 debug。

文件 docstring 里加一段说明，记下来「这文件不负责 ContextVar；ContextVar 由 make_lead_agent 在 bg-loop 上设置」，避免后人重蹈覆辙。

### 3. 删除一个调试遗留的未使用 import

[`packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py`](packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py#L13)

`DEFAULT_USER_ID` 之前作为 fallback 调试用，已不需要。

### 4. 新增单测

[`packages/agent/backend/tests/test_lead_agent_user_context.py`](packages/agent/backend/tests/test_lead_agent_user_context.py)

4 个测试覆盖：
- `_AuthUser` 满足 CurrentUser Protocol
- `langgraph_auth_user_id` 存在 → ContextVar 正确设置
- `langgraph_auth_user_id` 缺失 → fallback 到 `DEFAULT_USER_ID`
- UUID 类型自动 coerce 到 str

---

## 五、不需要修改的地方

- `UploadsMiddleware`、`ThreadDataMiddleware`、`MemoryMiddleware`、`ArchivingSummarizationMiddleware` **不动** — 它们继续调用 `get_effective_user_id()`，现在能在 bg-loop 上拿到正确的 user_id
- `user_context.py` 的接口契约（`get_effective_user_id` 三态语义）不变
- 持久化层（repository、SQLite）不动

---

## 六、验证

### 单测

- 新单测 4/4 通过
- 相关测试（user_context / uploads_middleware_core / owner_isolation / threads_router / harness_boundary）80/80 通过
- 完整 backend 套件：2142 通过 / 5 失败
  - 5 个失败均与本次改动**无关**，stash 验证为预存在（3 个 JWT secret 警告捕获、1 个 skill 启用状态、1 个 prompt 措辞）

### E2E

用户在真实环境验证：上传 EthoVision XT 文件 → Agent 正确读取并分析。问题消失。

---

## 七、为什么 Tier 4 user-context 系统留着是对的

CLAUDE.md 里把 `runtime.user_context` 标为 Tier 4（per-user filesystem isolation），警告同步上游时不要直接拉。但**本仓库已经在用它**：

- `UploadsMiddleware.before_agent` 调用 `get_effective_user_id()`
- `ThreadDataMiddleware.before_agent` 调用 `get_effective_user_id()`
- `MemoryMiddleware.after_agent` 调用 `get_effective_user_id()`（在 enqueue 时显式捕获，因为 Timer 跑在另一个线程）
- `ArchivingSummarizationMiddleware` 调用 `get_effective_user_id()`
- `app.gateway.auth_middleware` 在 Gateway 进程里 set/reset

修复后这个体系完整运转。删掉它会破坏 4 个 middleware 的多用户隔离能力，违背产品定位（v0.1 要支持研究员个人数据隔离）。

---

## 八、如果以后还出现 fallback 到 `default` 的情况

排查清单：

1. **看 log 的 `thread_name`**：
   - 如果 fallback 发生在 `bg-loop-*`：检查 `make_lead_agent` 是不是被绕过了（比如有别的 graph 入口没经过这个 factory）
   - 如果 fallback 发生在请求线程：检查 Gateway 的 `AuthMiddleware` 是不是在那个 router 上跑了

2. **查 `configurable` 里有没有 `langgraph_auth_user_id`**：
   - 在 `make_lead_agent` 里临时加 `logger.info("cfg keys: %s", list(cfg.keys()))`
   - 如果没有这个 key：说明请求没带 auth cookie 或 langgraph_auth.py 的 `@auth.authenticate` 没正常跑

3. **subagent 能否复用主 agent 的 ContextVar**：
   - 当前 `subagents/executor.py` 的 ThreadPoolExecutor 不会传播 ContextVar
   - 如果 subagent 也需要访问 user_id：要么在 SubagentExecutor 里显式捕获并 reapply，要么 subagent 不再依赖 ContextVar 改走 explicit user_id 参数
   - 现在 subagent 走 sandbox 路径，路径已经被 lead agent 解析好（包含 user_id），所以暂时没问题

---

## 九、相关引用

- LangGraph 源码确认 user_id 传递通道：
  - [`langgraph_api/models/run.py:223-265`](packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph_api/models/run.py#L223-L265) — 创建 run 时把 `langgraph_auth_user_id` 写进 configurable
  - [`langgraph_api/worker.py:60-83`](packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph_api/worker.py#L60-L83) — bg-loop 上从 run kwargs 读出来
  - [`langgraph_api/auth/custom.py:74-129`](packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph_api/auth/custom.py#L74-L129) — `@auth.on` 在请求线程跑，确认与 bg-loop 隔离
- 本次修复 commit（待提交）：`fix(agent): 在 bg-loop 上从 configurable 解析 user_id，修复跨线程 ContextVar 不传递导致的上传文件丢失`

---

## 十、待办（可选 follow-up）

- [ ] 把 `2026-05-07-upload-file-not-recognized-fix-handoff.md` 标记为「已被本 handoff 取代」
- [ ] 考虑加一条 e2e 集成测试：起 LangGraph Server + Gateway，用真 cookie 上传文件，断言 agent 看到了 — 单测覆盖不到跨线程语义
- [ ] CLAUDE.md 第 11 条之后，加一条关于「ContextVar 必须在 bg-loop task 上 set」的血泪经验，避免后人再次踩坑
