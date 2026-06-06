# 调查 + 修复 brief：ask_clarification 反问后 lead 重复重问（画图问题循环）

> **面向接手的 AI agent**。本 brief 已由上一会话**现场核实**（41 条消息全解码 + checkpoint + 中间件源码定位），根因已基本锁定，**真根因在 harness 中间件层，不是环境/前端问题**。你的任务：核实结论 → 实施修复（中间件治本 + prompt 兜底）→ 加 red 锚点单测 → 跑全量 → dogfood 复测。
>
> 仓库 HEAD 应含 commit `e5a0d53b`（data-analyst seal 修复）。本问题与 seal 修复**无关**，是独立的新问题。

---

## 1. 现象

O迷宫 dogfood（thread `ca86744f-39e6-4acb-95fa-bda8ec8593ed`，user `cd95effa-...`）：分析全链路正常跑通（code-executor + data-analyst 都第一次派遣即 seal 成功），但走到「E2E_FULL_ASKVIZ 反问要不要出图」时，lead **连续重复问了 3 遍同一个画图问题**，用户答完才继续。用户原话：「为什么会反复问我要不要画图？？？？」

**注意：这不是死循环、不是 ask_clarification 中断坏了、不是 seal 回退。** 最终链路是通的（用户答"画图"后正常 set_viz_choice → 派 chart-maker）。问题是**反问后到用户回答之间，lead 被强制重新唤起、把问题重打了 3 遍**，体验很差。

---

## 2. 现场核实证据（已验证，可直接采信，但建议复核）

### 2.1 消息序列（checkpoint 解码，41 条消息）

用 langgraph 序列化器解码 checkpoint（**注意：用 `JsonPlusSerializer`，不是 msgpack；msgpack 模块在该 venv 没装**）：

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
.venv/bin/python - <<'PY'
import sqlite3
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
con=sqlite3.connect(".deer-flow/checkpoints.db"); cur=con.cursor()
cur.execute("SELECT type, checkpoint FROM checkpoints WHERE thread_id LIKE '%ca86744f%' ORDER BY rowid DESC LIMIT 1")
typ,blob=cur.fetchone(); con.close()
msgs=JsonPlusSerializer().loads_typed((typ,blob)).get('channel_values',{}).get('messages',[])
for i,m in enumerate(msgs):
    if i<30: continue
    cls=type(m).__name__
    tcs=[tc.get('name') for tc in (getattr(m,'tool_calls',None) or [])]
    c=getattr(m,'content','')
    if isinstance(c,list): c=' '.join(str(b.get('text','')) if isinstance(b,dict) else str(b) for b in c)
    print(f"[{i}] {cls} tool_calls={tcs} :: {str(c).replace(chr(10),' ')[:80]}")
PY
```

关键尾段（已实测）：

| msg | 类型 | tool_calls | 内容摘要 |
|-----|------|-----------|---------|
| [31] | AIMessage | `write_todos`, **`ask_clarification`** | 汇报 + 真调 ask_clarification（画图问题） |
| [33] | ToolMessage(ask_clarification) | — | `🔀 📊 ...需要我把结果可视化成图吗？` + 3 选项 ← **中间件 Command(goto=END) 正确中断** |
| **[34]** | **AIMessage** | **[]（无 tool_call）** | `分析暂告一段落，正在等待你对上图方式的选择...` ← **重问 1** |
| **[35]** | **AIMessage** | **[]** | `目前还在等待你对上图方式的选择...` ← **重问 2** |
| **[36]** | **AIMessage** | **[]** | `还在等待你的选择哦～...` ← **重问 3** |
| [37] | HumanMessage | — | 用户真实回答（选"是，画图"） |
| [38] | AIMessage | `set_viz_choice` | `好的，先记录下来` |
| [40] | AIMessage | `write_todos`, `task` | 派 chart-maker |

**决定性观察**：[34][35][36] 是**连续的、无 tool_call 的纯文本 AIMessage，中间没有任何 HumanMessage**。在标准 ReAct 循环里，无 tool_call 的 AIMessage = agent 完成 → 该走 END。但它被连续唤起了 3 次。问"为什么 END 之后还被唤起" = 根因。

### 2.2 真根因（已定位到中间件源码）

`backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py` 的 `TodoMiddleware.after_model`（L265-309）：

```python
@hook_config(can_jump_to=["model"])
def after_model(self, state, runtime):
    base_result = super().after_model(state, runtime)
    if base_result is not None: return base_result
    # 2. 只在 agent 想干净退出时介入
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if not last_ai or _has_tool_call_intent_or_error(last_ai):
        return None
    # 3. todos 全完成 → 放行
    todos = state.get("todos") or []
    if not todos or all(t.get("status") == "completed" for t in todos):
        return None
    # 4. 上限 _MAX_COMPLETION_REMINDERS = 2
    if self._completion_reminder_count_for_runtime(runtime) >= self._MAX_COMPLETION_REMINDERS:
        return None
    # 5. 注入"Please continue working"提醒 + jump_to model（强制回模型节点）
    self._queue_completion_reminder(runtime, _format_completion_reminder(todos))
    return {"jump_to": "model"}
```

**机制链（两根因实为一体）**：
1. lead 调 `ask_clarification` → ClarificationMiddleware `Command(goto=END)` 结束该 turn（这步对）。
2. 但 `反问是否需要出图` 这条 todo 仍是 `in_progress`（未 completed）。
3. lead 被重新进入、产出无 tool_call 的文本 AIMessage（重问）。
4. `TodoMiddleware.after_model` 看到「无 tool_call + 有未完成 todo」→ 注入 `<system_reminder>Please continue working on these tasks...`（L72）+ `jump_to: model`，**强制 lead 回到模型节点**。
5. lead 被"continue working"提醒逼着再问一遍 → 回到 3。
6. 直到撞 `_MAX_COMPLETION_REMINDERS = 2` 上限才停 → 正好 **1 次原始 + 2 次强制 = 3 条重问**，与实测 [34][35][36] 完全吻合。

**两个根因其实是同一机制的两面**：
- **根因①（harness，真根因）**：`TodoMiddleware` 在 agent **正当等待用户澄清**时，仍因 todo 未完成而强制 re-engage。它分不清"agent 偷懒提前退出"（该催）vs"agent 在等用户答 ask_clarification"（不该催）。
- **根因②（prompt，次要）**：lead 被催时，没有 prompt 指引告诉它"我已发 ask_clarification 正在等、该静默"，于是顺着"continue working"把问题重打。lead 的 thinking 已明确写「user hasn't responded yet... I should wait... but system says continue working」——它知道该等，是被中间件逼的。

---

## 3. 修复方案

### 3.1 🔴 根因①治本（harness，主修复）

`TodoMiddleware.after_model` 增加一个**短路条件**：当 agent 处于"等待用户澄清"状态时，**不**注入 completion reminder、**不** jump_to model，直接 `return None`（放行干净退出）。

判定"等待澄清"的可靠信号（择一或组合，**实施前先核实哪个在 state 里可见**）：
- 消息历史里**最近一条 ToolMessage 是 `name=="ask_clarification"`**，且其后没有新的 HumanMessage —— 说明刚发出澄清、还没等到用户答。这是最直接的信号（ClarificationMiddleware 注入的 ToolMessage `name="ask_clarification"`，见 `clarification_middleware.py:145`）。
- 备选：最近一条 AIMessage 的 tool_calls 含 `ask_clarification`（但 [34] 之后 AIMessage 已无 tool_call，所以要往前找最近的 ask_clarification ToolMessage）。

建议实现（在 L293 "clean exit" 判定后、L296 todo 判定前插入）：

```python
# 当 agent 正在等待用户回应 ask_clarification 时，不要催它继续——
# 它的"无 tool_call 输出"是合法的等待态，不是提前退出。强行 jump_to model
# 会让 lead 把澄清问题重复重打（实测连问 3 遍），体验极差。
# 判定：最近一条 ToolMessage 是 ask_clarification 且其后无新 HumanMessage。
if _is_awaiting_clarification(messages):
    return None
```

`_is_awaiting_clarification` 辅助函数：从后往前扫 messages，遇到 `HumanMessage` 先返回 False（用户已答），遇到 `ToolMessage(name="ask_clarification")` 返回 True，都没遇到返回 False。

**注意边界**：
- 该短路只影响"催 todo"逻辑，不动 ClarificationMiddleware 的中断本身。
- 不要误伤正常的"agent 偷懒提前退出"催办（那个机制是有用的，保留）。
- `super().after_model`（base 类的并行 write_todos 检测，L284）保持先跑，短路加在它之后。

### 3.2 🟡 根因②兜底（prompt，次要）

`backend/packages/harness/deerflow/agents/lead_agent/prompt.py` **L464**（`<clarification_system>` 段内）：

当前：
```
- ✅ 调用 ask_clarification 后执行会自动中断，等待用户回复后再继续
```

在其后补一条**正向指令**（遵守 CLAUDE.md §6 deepseek 正面提示原则，**禁用**"不要重复问"这类否定式反向激活）：

```
- ✅ 已发出 ask_clarification、用户尚未回复时：保持静默等待。若你在没有新用户回复的情况下被再次唤起，说明用户还在思考——此时无需任何输出，更无需重述问题；问题已经展示给用户，等待其回复即可。
```

措辞要点：正面描述"该做什么"（静默等待），不写"不要再问"。

### 3.3 验证（必做）

1. **red 锚点单测**（harness）：新建/扩 `backend/tests/test_todo_middleware.py`（若无则建）——构造 state：messages 末尾是 `ToolMessage(name="ask_clarification")`、todos 有 `in_progress` 项、last AIMessage 无 tool_call → 断言 `after_model` 返回 `None`（不 jump_to model）。再构造对照：ask_clarification 后**有** HumanMessage → 仍正常催办。
2. **prompt 契约单测**（可选）：断言 L464 后新指令文本存在，锁防 sync 回退。
3. **跑全量**（铁律：改共享中间件必跑全量，不能只跑新测试）：`cd backend && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`。
   - ⚠️ 已知 flake：全量顺序下 `test_deferred_tool_registry_promotion.py` 2 条 + `test_inspect_gate_guardrail.py::...test_async_delegates_to_sync` + `test_paradigm_identification_gate.py::...test_async_delegates_to_sync` 共 4 条会偶发 fail（event-loop 污染 + 全局 registry 变更的顺序依赖），**单跑这 4 条 4/4 passed**，与改动无关，别误判成回退。
4. **重启 dev 后 dogfood 复测**（铁律：改 harness/prompt 必须 `make stop && make dev` 重启，否则跑旧代码）：
   - 跑 O迷宫（`/home/wangqiuyang/DemoData/newdemodata/O迷宫/` 3 个 xlsx，发"分析数据"）。
   - 走到画图反问时：**期望只问一次**，然后干净挂起等用户；不再有 [34][35][36] 式的连续重打。
   - 用户答"画图"后正常派 chart-maker。

---

## 4. 复测核验命令

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
# 找最新 O迷宫 thread
ls -dt .deer-flow/users/*/threads/*/ | head
TID=<新thread_id>
WS=$(find .deer-flow/users -type d -path "*threads/$TID/user-data/workspace")
# 用 §2.1 的解码脚本（改 thread_id），数 [画图问题] 区段有几条连续无 tool_call AIMessage
# 期望：ask_clarification ToolMessage 之后、用户 HumanMessage 之前，AIMessage 数 ≤ 1
```

判定通过：画图反问只出现一次连续输出，中间不被 todo 中间件强制 re-engage。

---

## 5. 注意 / 红线

- **不要动 ClarificationMiddleware 的 `Command(goto=END)`** —— 中断本身是对的，问题在 TodoMiddleware 不该在等待态催办。
- **不要把 `_MAX_COMPLETION_REMINDERS` 调成 0** 来"治"这个问题 —— 那会废掉正常的"防 agent 提前退出"催办（它对别的场景有用）。要的是**精准短路 ask_clarification 等待态**，不是关掉整个机制。
- **正面措辞铁律**（CLAUDE.md §6）：prompt 改动禁用"不要/禁止/别重复"，用"保持静默/等待即可"正向描述。
- **改共享中间件必跑全量 + grep 调用方**（用户铁律）：`TodoMiddleware` 是 lead 中间件链一环，grep 它在 `agent.py` 中间件链的位置，确认短路不影响其他 hook。
- **改完必重启 dev** 再 dogfood，否则白测。

---

## 6. 关键路径速查

- 中间件（主修复点）：`backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py:265`（`after_model`）+ `:64`（`_format_completion_reminder`，"Please continue working" 文本源）
- 中断机制（**别动**，仅供理解）：`backend/packages/harness/deerflow/agents/middlewares/clarification_middleware.py:153`（`Command(goto=END)`）+ `:145`（注入 `name="ask_clarification"` ToolMessage）
- prompt 兜底点：`backend/packages/harness/deerflow/agents/lead_agent/prompt.py:464`（`<clarification_system>` 段）
- ASKVIZ 反问模板：`prompt.py:323` 起（E2E_FULL_ASKVIZ 反问 + set_viz_choice 落 gate3）
- 失败 thread（回放）：`.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/ca86744f-39e6-4acb-95fa-bda8ec8593ed/`
- checkpoint 解码：用 `langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`，**不是 msgpack**
- 全量回归：`cd backend && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`（基线约 3675 passed + 4 条已知顺序 flake）
