# 2026-05-26 Channel 'todos' 根因诊断 + 修复方向

## 当前任务目标

修复 ECS 生产容器中 `is_plan_mode: True` 时 lead agent 报 `ValueError: Channel 'todos' already exists with a different type` 导致所有 FST 端到端请求失败的 bug。

---

## 问题现象

```
ValueError: Channel 'todos' already exists with a different type
    at StateGraph.__init__ → _add_schema(input_schema)
```

- **只在 ECS Docker 容器里发生，本地 `make dev` 不报错**
- **只在 `is_plan_mode: True` 时发生**（触发 `TodoMiddleware`）
- ECS 版本：langgraph 1.0.9 / langchain 1.2.3（与本地 lockfile 完全一致）

---

## 本次诊断进展（核心发现）

### 已确认的事实

1. ECS 容器内直接测试 channel 类型：
   - `_resolve_schema(state_schemas, "StateSchema", None)` → `todos: LastValue`
   - `_resolve_schema(state_schemas, "InputSchema", "input")` → `todos: BinaryOperatorAggregate`
   - 本地结果相反（两者都是 `BinaryOperatorAggregate`，且 `==` True）

2. `StateGraph.__init__` 逻辑：
   ```python
   self._add_schema(self.state_schema)        # 先注册 todos → LastValue
   self._add_schema(self.input_schema, ...)   # 再注册 todos → BinaryOperatorAggregate → 冲突！
   ```
   `_add_schema` 里：`if self.channels[key] != channel → raise ValueError`

3. `_add_schema` 有短路保护：`if schema not in self.schemas: skip`，所以 `state_schema is input_schema` 时不会重复处理。但 langchain `factory.py` 生成的是**两个不同的 TypedDict 实例**（`_resolve_schema` 调用了两次）。

### 根因假设（最可能）

`langchain/agents/factory.py` 里：
```python
state_schemas: set[type] = {m.state_schema for m in middleware}
state_schemas.add(base_state)  # ThreadState

resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
```

`_resolve_schema` 内部：
```python
for schema in schemas:  # schemas 是 set，迭代顺序不确定！
    for field_name, field_type in hints.items():
        all_annotations[field_name] = field_type  # 后来者覆盖先来者
```

**Python set 的迭代顺序与元素的哈希值有关，不同机器/Python 进程的哈希种子（`PYTHONHASHSEED`）不同，导致顺序不同。**

在 ECS 上：`state_schemas` 迭代时，某个 schema（可能是 `AgentState` 或别的）定义了 `todos` 为无 reducer 类型，且比 `ThreadState` 后迭代，就把 `ThreadState.todos: Annotated[list|None, merge_todos]` 覆盖掉了。

但 `InputSchema` 模式会 omit `OmitFromInput` 字段（如 `PlanningState.todos`），所以最后只剩 `ThreadState` 的带 reducer 版本。

⚠️ **注意**：这个假设还未完全验证——ECS 上 `_TodoState`（无 todos）+ `ThreadState`（有 todos）+ 其他 schemas 的组合到底是哪个覆盖了哪个，需要进一步确认。

### 另一个可能的根因

`_is_field_binop` 判断 `callable(meta[-1])` 且参数数为 2。如果 `merge_todos` 在某路径下被 Python 的函数签名检查误判（比如有 `*args`），会回退到 `LastValue`。但从代码看 `merge_todos` 签名是 `(existing, new)` 两个参数，这个可能性较低。

---

## 修复方向

### 方案 A（推荐）：在 `_TodoState` 里显式声明 `todos`

当前 `_TodoState` 故意不含 `todos`，让 `ThreadState` 成为唯一来源。但 set 迭代不确定性仍然可能让其他 schema 的 `todos` 覆盖。

**修复**：在 `_TodoState` 里加和 `ThreadState` 完全一致的 `todos` annotation，确保不管顺序如何，合并结果都是 `BinaryOperatorAggregate`：

```python
# todo_middleware.py
from deerflow.agents.thread_state import merge_todos

class _TodoState(AgentState):
    todos: Annotated[list | None, merge_todos]  # 显式声明，和 ThreadState 一致
```

这样 `_resolve_schema` 无论迭代顺序如何，`todos` 最终都是带 `merge_todos` reducer 的 annotation。

⚠️ **注意**：需要验证这不会重新引入 `PlanningState.todos` 的冲突。`PlanningState.todos: Annotated[NotRequired[list[Todo]], OmitFromInput]`——`StateSchema` 处理时 `_TodoState.todos` 和 `ThreadState.todos` 都是同样的 `Annotated[list|None, merge_todos]`，互相覆盖无影响；`PlanningState.todos` 带 `OmitFromInput` 会被 `InputSchema` 的 omit 逻辑剔除，所以不存在 `InputSchema` 里 `todos` 为 `OmitFromInput` 类型的问题。

### 方案 B：固定 `PYTHONHASHSEED=0`

在 Dockerfile 或 docker-compose.yaml 里设 `PYTHONHASHSEED=0`，让 set 迭代顺序确定化。**治标不治本**，不推荐作为最终方案。

### 方案 C：修改 langchain factory.py（不推荐）

改成用 `list` 而非 `set`，按固定顺序合并 schemas。需要 fork 或 monkey-patch，维护成本高。

---

## 修复实施步骤

```bash
# 1. 修改文件
# packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py

# 在 _TodoState 里加：
# from deerflow.agents.thread_state import merge_todos  (已有 import from thread_state 可复用)
# todos: Annotated[list | None, merge_todos]

# 2. 本地验证（模拟 ECS 的 set 顺序）
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONHASHSEED=1 .venv/bin/python -c "
import os
os.environ['DEER_FLOW_CONFIG_PATH'] = '/home/wangqiuyang/noldus-insight/packages/agent/config.yaml'
from langchain_core.runnables import RunnableConfig
config = {'configurable': {'thread_id': 'test', 'is_plan_mode': True, 'subagent_enabled': True}}
from deerflow.agents.lead_agent.agent import make_lead_agent
agent = make_lead_agent(config)
print('SUCCESS')
"

# 3. 测试不同 hash seed
for seed in 0 1 2 42 100 999; do
    PYTHONHASHSEED=$seed .venv/bin/python -c "
import os; os.environ['DEER_FLOW_CONFIG_PATH'] = '/home/wangqiuyang/noldus-insight/packages/agent/config.yaml'
from langchain_core.runnables import RunnableConfig
config = {'configurable': {'thread_id': 'test', 'is_plan_mode': True, 'subagent_enabled': True}}
from deerflow.agents.lead_agent.agent import make_lead_agent
make_lead_agent(config)
print('seed=$seed OK')
" 2>&1 | grep -E "OK|Error|ValueError"
done

# 4. 运行测试套件
PYTHONPATH=$PWD/packages/harness:$PWD \
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
.venv/bin/python -m pytest tests/ --tb=line -q

# 5. build + deploy
cd /home/wangqiuyang/noldus-insight/packages/agent
make deploy-tar
```

---

## 关键文件

| 文件 | 说明 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py:110-122` | `_TodoState` 定义，需加 `todos` annotation |
| `packages/agent/backend/packages/harness/deerflow/agents/thread_state.py:49-67` | `merge_todos` reducer + `ThreadState.todos` 定义 |
| `packages/agent/backend/.venv/lib/python3.12/site-packages/langchain/agents/factory.py:852-864` | `_resolve_schema` + `StateGraph()` 调用，根因所在 |
| `packages/agent/backend/.venv/lib/python3.12/site-packages/langgraph/graph/state.py:248` | `_add_schema(input_schema)` 冲突检测 |

---

## 注意事项

1. **不要用 `PYTHONHASHSEED=0` 作为唯一修复**——只是掩盖问题，ECS 重启后 seed 可能变
2. **修复后必须用多个不同 PYTHONHASHSEED 测试**，确认不再依赖 set 顺序
3. **`_TodoState` 加了 `todos` 后**，注意不要 import `PlanningState` 的 `todos` 定义——始终用 `thread_state.merge_todos`
4. ECS 上还遗留有 `test_todo_middleware.py` 和 `todo_middleware.py` 两个调试文件在 `/app/backend/`，需要清理

---

## 仓库状态

- **dev HEAD**: `94867afd`（5/26 session final handoff docs）
- **生产 ECS**: 已部署最新 dev（`919cad1d` merge），bug 复现
- **本地测试**：`PYTHONHASHSEED` 默认值时不报错，其他值未测试

---

## 第一步建议

```bash
# 确认根因：用不同 hash seed 在本地复现
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
for seed in 0 1 2 3 4 5; do
  result=$(PYTHONHASHSEED=$seed .venv/bin/python -c "
import os; os.environ['DEER_FLOW_CONFIG_PATH']='../config.yaml'
from deerflow.agents.lead_agent.agent import make_lead_agent
make_lead_agent({'configurable':{'thread_id':'t','is_plan_mode':True,'subagent_enabled':True}})
print('ok')
" 2>&1)
  echo "seed=$seed: $result" | tail -1
done

# 如果某个 seed 复现了错误，说明根因确认，然后实施方案 A
```
