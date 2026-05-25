# 2026-05-25 ECS Docker 部署 Channel 'todos' 错误 handoff

## 问题

ECS 上 `make up`（Docker 生产模式）启动后，浏览器访问时报错：

```
76717fb1cb5c1993.js:59 ValueError: Channel 'todos' already exists with a different type
    at o.enqueue (76717fb1cb5c1993.js:59:13003)
```

错误来源：Python `langgraph` 库 `graph/state.py:276`，后端报错被前端 JS 日志捕获。

## 已尝试（均无效）

1. 删除 `checkpoints.db` → 错误依旧
2. `docker compose down -v` → 失败（`DEER_FLOW_HOME` 未设，bind mount）
3. 全量 rsync 本地 → ECS + 删除 `checkpoints.db` + 重新 `make up` → 错误依旧

## 关键发现

### 本地 vs ECS 对比

| | 本地 `make dev` | ECS `make up` |
|---|---|---|
| 运行方式 | 直接进程 | Docker 容器 |
| 报错 | **不报错** | 报错 |

### 错误本质

`langgraph/graph/state.py:276` 的含义是：**在同一个 StateGraph 编译时，`todos` channel 被注册了两次，且类型不同**。这与 checkpoint 数据库无关——是图编译阶段的 schema 冲突。

```python
# langgraph/graph/state.py L269-277
for key, channel in channels.items():
    if key in self.channels:
        if self.channels[key] != channel:
            if isinstance(channel, LastValue):
                pass
            else:
                raise ValueError(
                    f"Channel '{key}' already exists with a different type"
                )
```

### 根因假设

今天 5/25 的 DeerFlow sync 合入了 `merge_todos` reducer（commit `e2e22e3a`，上游 `8785658a`），将 `ThreadState.todos` 从：

```python
todos: list | None  # 无 reducer
```

变为：

```python
todos: Annotated[list | None, merge_todos]  # 有 reducer
```

这改变了 channel 类型。如果图编译时有两个 schema 都定义了 `todos`（如 Input/Output schema + ThreadState），且其中一个还没更新、仍用旧类型，就会触发此错误。

**但为什么本地不报错、只有 Docker 报错？** 可能原因：
- Docker 构建时可能缓存了旧的 `.pyc` / 旧依赖版本
- Dockerfile 里拉取的 langgraph 版本与本地不同
- 图编译路径在 Docker vs 本地有差异

## 需要调查的文件

| 文件 | 用途 |
|---|---|
| `backend/packages/harness/deerflow/agents/lead_agent/agent.py` | `make_lead_agent()` — 图构建入口，检查 `StateGraph` 的 input/output schema 定义 |
| `backend/packages/harness/deerflow/agents/thread_state.py:67` | `todos: Annotated[list \| None, merge_todos]` — 当前定义 |
| `backend/Dockerfile` 或 `docker/Dockerfile` | Docker 镜像构建，检查是否有 build cache 污染 |
| `docker/docker-compose.yaml` | 检查 langgraph 容器的构建上下文 |

## 建议接手步骤

1. **先在 ECS 上确认 Docker 镜像是否真用了最新代码**：

```bash
# 强制无缓存重建（在 ECS 上）
docker compose -f docker/docker-compose.yaml build --no-cache
docker compose -f docker/docker-compose.yaml up -d
```

2. 如果还不行，在 ECS 容器内直接 grep 确认 `todos` channel 的运行时定义：

```bash
docker exec deer-flow-langgraph python -c "
from deerflow.agents.thread_state import ThreadState
print(ThreadState.__annotations__['todos'])
"
```

3. 搜索 `agent.py` 中是否有多重 schema 注册（Input/Output state 也带 `todos`）：

```bash
grep -n "StateGraph\|input_schema\|output_schema\|add_node.*input\|add_node.*output\|class.*State" \
  backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

4. 对比 ECS 容器内的 langgraph 版本和本地：

```bash
# ECS 容器内
docker exec deer-flow-langgraph pip show langgraph | grep Version
# 本地
uv run pip show langgraph | grep Version
```

## ECS 信息

- ECS 路径：`/root/noldus-insight/packages/agent/`
- `DEER_FLOW_HOME` = `/root/noldus-insight/packages/agent/backend/.deer-flow`
- 容器名：`deer-flow-langgraph`, `deer-flow-gateway`, `deer-flow-frontend`, `deer-flow-nginx`
