# 2026-06-01 Sprint S6+S7+S8 实施交接

> **本 handoff 用途**：接上本会话做完的 S4.5+S5，交新 agent 继续做 S6、S7、S8。
>
> **dev HEAD（写本文时）**：`ff9a8b85`（S5 feat 刚合入）
>
> **当前工作区**：worktree `sprint-6-7-memory-assumptions`（已 rebase 到 dev，**一行代码都没写**，可直接开工或丢弃重建）

---

## 0. 已完成进度（本会话内）

| Sprint | 状态 | dev commit |
|---|---|---|
| S4.5 `analysis_config_id` | ✅ push 合入 | `d8a1b7af` |
| S5 `DataQualityGuardrailProvider` | ✅ push 合入 | `ff9a8b85` |
| S6 跨会话 memory | ❌ worktree 已建但零代码 | — |
| S7 假设面板 | ❌ 未开始 | — |
| S8 feedback 回流 | ❌ 未开始 | — |
| S3 FST mobility 判据 | ⏭ 卡 #63 同事 | — |
| S4 调参指南内容 | ⏭ 卡同事 SSOT | — |

---

## 1. Brief 原文权威来源

`/home/wangqiuyang/noldus-insight/docs/handoffs/2026-06/2026-06-01-remaining-sprints-impl-brief.md`  
完整实施说明在那里（§6 S6、§7 S7、§8 S8）。**本文只补充代码核实后的精确情况，不重复 brief 内容。**

---

## 2. S6 跨会话 memory — 代码核实真实情况

### 2.1 Brief 的关键假设 vs 代码现实

| Brief 假设 | 代码现实 |
|---|---|
| "seal tool 无 `add_fact`" | `seal_handoff_tools.py` **存在**（309行），在 report-writer builtins 中已注册 |
| "改 `seal_report_writer_handoff` 内部" | **正确**——`seal_report_writer_handoff` 在 `seal_handoff_tools.py:283`，走 `_seal_handoff()` 辅助，是正确的注入点 |
| "`agents/memory/storage.py` 加 `add_fact()`" | `storage.py` 只有 `load/reload/save`，**无 `add_fact`**，需要添加 |

### 2.2 实施锚点（已核实文件:行）

| 用途 | 路径:锚点 |
|---|---|
| S6 注入点 | `tools/builtins/seal_handoff_tools.py:309`（`seal_report_writer_handoff` 末尾，在 `_seal_handoff` 返回前插入） |
| memory storage | `agents/memory/storage.py:160`（`save()` 方法，需新增 `add_fact()` helper）|
| memory facts 结构 | `create_empty_memory():39` → `"facts": []`；fact 字段见 CLAUDE.md Memory System 段 |
| `get_memory_storage()` 公开入口 | `agents/memory/storage.py:196` |
| user_id 来源 | `_seal_handoff()` 内部通过 `runtime.state["thread_data"]["user_id"]`（或 `None`） |
| thread_id 来源 | `runtime.state.get("thread_id", "unknown")` |
| analysis_config_id 来源 | `_read_analysis_config_id(workspace)` 已在 seal_handoff_tools.py:57 |

### 2.3 add_fact() 设计

```python
def add_fact(
    self,
    content: str,
    category: str,
    confidence: float = 1.0,
    source: str = "",
    agent_name: str | None = None,
    *,
    user_id: str | None = None,
) -> bool:
    """Atomically append a fact to memory.json facts list."""
    # 1. load current memory
    # 2. append fact with {id: uuid4, content, category, confidence, createdAt: utc_now_iso_z(), source}
    # 3. save via self.save()
```

不要重复造轮子——用已有的 `load()` + 修改 + `save()`。`max_facts` eviction 已经在 `updater.py` 里——`add_fact` 只做 append，不管 eviction（那是 updater 的职责）。

### 2.4 experiment_summary fact 格式

```python
# 从 handoff_data_analyst.json 和 handoff_code_executor.json 读取数字
# 不经 LLM —— 确定性拼接
content = (
    f"{paradigm} analysis on {date}: "
    f"n_per_group={n_per_group}; "
    f"key_findings_count={len(key_findings)}; "
    f"analysis_config_id={config_id}"
)
category = "experiment_summary"
confidence = 1.0
source = f"thread:{thread_id}, config_id:{config_id}"
```

注意：`n_per_group` 从 `plan_metrics.json` 或 `handoff_code_executor.json.metadata` 读取（优先 code-executor handoff）；若读不到用 `"n_per_group=unknown"`——不要让失败的读取 block memory 写入。

### 2.5 注意事项

- `seal_report_writer_handoff` 是在 **subagent（report-writer）上下文**里调用的，不是 lead 上下文。`runtime.state` 里有 `thread_data`（包含 `workspace_path`、`user_id`、`thread_id`）。
- `add_fact` 必须能处理 `user_id=None`（单用户环境）。
- memory 写失败不能让 seal tool 失败——wrap 在 try/except，失败只 log warning。
- **不改**：MemoryMiddleware / updater / config / prompt（这是 brief 的硬约束）。

---

## 3. S7 假设面板 — 代码核实真实情况

### 3.1 真实文件状态

- `tools/builtins/present_assumptions.py` **不存在**（需新建）
- `tools/builtins/__init__.py` 需要注册新工具
- `tools/tools.py` 注册 builtin tools 的地方也需要确认是否自动扫描

### 3.2 实施锚点

| 用途 | 路径:锚点 |
|---|---|
| 新建工具文件 | `tools/builtins/present_assumptions.py`（新建） |
| 工具注册 | `tools/builtins/__init__.py`，参考其他 builtins 的注册方式 |
| plan_metrics.json 读取 | `catalog/resolve.py` 的 `plan_metrics_to_dict()` 结构参考 |
| handoff_data_analyst.json 结构 | `subagents/handoff_schemas.py:DataAnalystHandoff` |
| experiment-context.json 结构 | `agents/middlewares/experiment_context.py:read_context()` |
| lead prompt 建议性指引位置 | `agents/lead_agent/prompt.py` 的 `<critical_reminders>` 段（约 line 590） |

### 3.3 工具设计

```python
@tool("present_assumptions", parse_docstring=True)
def present_assumptions_tool(
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """聚合分析假设并渲染为可折叠 Markdown 卡片。
    
    Args:
        workspace_dir: 工作目录。默认 "/mnt/user-data/workspace/"。
    
    Returns:
        Markdown 文本，前端渲染为折叠卡片（标题"分析假设摘要"）。
    """
    # 1. 读 experiment-context.json → analysis_config_id, parameter_overrides, gate_completed
    # 2. 读 plan_metrics.json → parameters_in_use（如有）
    # 3. 读 handoff_data_analyst.json → quality_warnings, parameter_audit_findings
    # 4. 拼装 Markdown
    # 5. 返回字符串（或 JSON，看前端约定）
```

输出格式（markdown collapsed card）：

```markdown
<details>
<summary>分析假设摘要 (config_id=a1b2c3d4...)</summary>

### 参数配置
- analysis_config_id: `a1b2c3d4e5f67890`
- parameter_overrides: {...} 或 "（使用 catalog 默认值）"

### 数据质量
- critical warnings: N 条（blocks_downstream: M 条）
- 已确认: 是/否

### 参数审计
- parameter_audit_findings: N 条（critical: M 条）
</details>
```

前端：brief 说"前端新增折叠卡片组件"——这是前端工作，backend tool 只输出 markdown 字符串，前端根据 `<details>` 渲染。如果本 sprint 不碰前端，tool 返回 markdown 即可，lead 把它 present_files 或直接回复给用户。

### 3.4 Lead prompt 指引位置

在 `prompt.py` 约 line 589 的 `<critical_reminders>` 加：

```
- Assumptions Panel: When analysis has critical warnings or parameter overrides, call present_assumptions() to surface the assumption summary.
```

---

## 4. S8 feedback 回流 — 快速概述

Brief §8 详细。关键锚点：
- `app/persistence/feedback_repository.py`（加 `paradigm` 字段）
- `app/gateway/routers/feedback.py`（写入 + 查询端点）
- `agents/lead_agent/agent.py` + `prompt.py`（注入 prior_corrections）

schema migration：SQLite `ALTER TABLE` 加 nullable `paradigm TEXT`。历史数据留 null。

---

## 5. 执行顺序建议

```
S6:
  1. 先跑全量 make test 确认基线（应该 3230+ passed，3 个 PYTHONPATH-only failures）
  2. add_fact() 写 TDD 测试，再实现
  3. seal_report_writer_handoff 注入 memory fact，写集成测试
  4. 全量 make test 验证

S7:
  1. 写 present_assumptions_tool TDD
  2. 实现
  3. 注册到 builtins
  4. lead prompt 加建议性指引
  5. 全量 make test 验证

S8（最低优先级）:
  1. 核实 feedback_repository.py 当前 schema
  2. SQLite migration + 新字段
  3. 新 API endpoint
  4. lead agent/prompt 注入
  5. 全量 make test
```

---

## 6. 关键约束（每个 sprint 都适用）

1. **每次 make test 必须跑全量**——S5 就踩过"只跑新测试导致 3 个旧测试 fail 进 dev"的坑
2. **每个 sprint 新开 worktree**（用户硬要求）：`EnterWorktree → rebase origin/dev → 写代码 → push → ExitWorktree`
3. **deny 消息必须含明确指令**（`feedback_deny_messages_must_direct.md`）
4. **SSOT 唯一**：参数默认值只在 catalog YAML，不双存
5. **add_fact 失败不阻 seal**：memory 写失败只 log warning

---

## 7. 测试运行方式（worktree 特殊 PYTHONPATH）

```bash
# 在 worktree 目录内：
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/<sprint-name>/packages/agent/backend
source /home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/activate
PYTHONPATH="$PWD/packages/harness:$PWD/../../packages/ethoinsight:$PWD" \
  python -m pytest tests/test_<specific>.py -v  # 只跑指定文件（快）

# 全量（验收时）：
PYTHONPATH="..." python -m pytest tests/ -q
# 预期：3230+ passed, 3 failed (PYTHONPATH-only, 不是真实回归)
```

---

## 8. 下一 Agent 第一步

```bash
# 1. 确认当前位置
git log --oneline -3  # 应看到 ff9a8b85 S5 提交

# 2. 确认 S6 worktree
ls /home/wangqiuyang/noldus-insight/.claude/worktrees/sprint-6-7-memory-assumptions/

# 3. 进入 S6 worktree（或新建，按本文说明重建）
# EnterWorktree path=/home/wangqiuyang/noldus-insight/.claude/worktrees/sprint-6-7-memory-assumptions
# 若 worktree 不干净，ExitWorktree action=remove 再 EnterWorktree name=sprint-6-memory

# 4. 先读 seal_handoff_tools.py 全文了解结构
# Read packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py

# 5. 写 TDD 测试 → 实现 add_fact → 注入 seal_report_writer_handoff
```
