# Spec A：memory context UUID 加载崩溃修复（后端，一行根治 + 防御）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `5616a73f`
> 性质：🟡 中 · 后端真 bug（lead 全程缺记忆跑，但被静默吞掉）
> 取证：dogfood thread `e9837b33` gateway.log
> 范围：纯后端 Python，**完全独立**，与前端/其它 spec 零耦合，可立即做

---

## 〇、问题（gateway.log 实测）

dogfood 端到端 log 出现 **ERROR**：
```
2026-06-26 09:39:57 - deerflow.agents.lead_agent.prompt - ERROR - Failed to load memory context: expected string or bytes-like object, got 'UUID'
```

**根因（读码坐实）**：
- `lead_agent/prompt.py:838`：`user_id = current_user.id if current_user is not None else None` —— `current_user.id` 是 **UUID 对象**。
- 但下游 `get_memory_data(agent_name, user_id=user_id)` → `storage.load(..., user_id)` 的签名全部声明 `user_id: str | None`（`memory/storage.py:47`、`memory/queue.py:23/46/57`）。**类型契约要求 str，实际传了 UUID 对象**，storage 内部对 user_id 做字符串/正则操作（拼路径或 re）时崩 `expected string or bytes-like object, got 'UUID'`。

**影响**：
- 异常被 `prompt.py:855-857` catch 成 `return ""` → **lead 全程在"无 memory context"下跑**（拿不到用户记忆 / 历史纠正 / resolved facts 中的 memory 部分）。
- **表面"成功"**（catch 后继续），但 lead 实际缺了记忆注入——这是被静默吞掉的退化，dogfood 看 UI 发现不了，只有 log 暴露。属「响亮故障被改成哑故障」反模式。

---

## 一、修复

### 1.1 根治点（必改）

`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:838`：
```python
# 前
user_id = current_user.id if current_user is not None else None
# 后
user_id = str(current_user.id) if current_user is not None else None
```

### 1.2 防御点（建议，根治同类）

调用点 `str()` 只修了这一处。但 `get_memory_data` / `storage.load` 声明 `user_id: str | None` 却没在入口强制——任何未来调用点裸传 UUID 都会复发。**建议在 `get_memory_data`（或 storage 边界）入口加一次归一**：
```python
# memory storage/get_memory_data 入口
if user_id is not None:
    user_id = str(user_id)   # 容错：调用点可能传 UUID 对象，契约要 str
```
这样调用点疏忽也不崩。**实施 agent 二选一或都做**：① 仅改 838（最小）② 838 + 入口防御（根治同类，推荐）。

---

## 二、验证

1. **裸导入两生产入口**（改 agents/ 必跑，CLAUDE.md 铁律）：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
2. **单测**：构造 `current_user.id` 为 UUID 对象的场景，调 `_get_memory_context`，断言**不抛异常且返回非空 memory context**（而非被 catch 成 ""）。这是 `code≠bug` 的回归——不能只看代码改了，要测"UUID 进来不再崩"。
3. **dogfood 复跑**：跑一个有用户记忆的 thread，grep gateway.log 确认**无** `Failed to load memory context` ERROR，且 lead system prompt 真注入了 `<memory>` 块。
4. `make test` + `make lint`。

---

## 三、关键文件

- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:838`（根治点）
- `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py:47` / `memory/queue.py`（防御点 + 契约来源）

---

*依据：gateway.log ERROR `Failed to load memory context: ...got 'UUID'` + 读码坐实 prompt.py:838 user_id=UUID 与下游 `user_id: str` 契约冲突。*
