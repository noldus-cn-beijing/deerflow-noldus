# 2026-06-01 Sprint S6+S7+S8 实施交接

> **dev HEAD（写本文时）**：`4b1ea0a4`
>
> **本分支**：`worktree-sprint-6-7-memory-assumptions`，2 commits 已 push
> - `c3644546` feat(S6+S7)
> - `77264915` feat(S8)
>
> **PR 链接**：https://github.com/noldus-cn-beijing/noldus-insight/pull/new/worktree-sprint-6-7-memory-assumptions

---

## 0. 本次实施内容

| Sprint | 状态 | 测试 | 改动 |
|---|---|---|---|
| S6 跨会话 memory fact | ✅ 完成 | 21 新测试 | seal_handoff_tools.py |
| S7 假设面板 tool | ✅ 完成 | 14 新测试 | present_assumptions.py (新) + __init__.py + prompt.py |
| S8 feedback 回流 | ✅ 完成 | 12 新测试 | model.py + sql.py + feedback.py + app.py + prompt.py |

**全量测试**：3283 passed / 3 failed (config.yaml PYTHONPATH-only) / 20 skipped
**新增测试**：47 个

---

## 1. S6 实施细节

### 1.1 设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| `add_fact()` 放哪 | 复用 `updater.create_memory_fact()` | 已有 load→append→save，结构匹配 |
| 写入时机 | seal_report_writer_handoff 成功后 | 只在完整分析结束时写 |
| 内容来源 | 从 handoff JSON 确定性读取 | 不经 LLM，避免数字漂移 |
| 失败处理 | try/except + log warning | 不阻 seal |

### 1.2 fact 内容格式

```
{paradigm} analysis on {YYYY-MM-DD}: n_per_group={n}; key_findings_count={count}; analysis_config_id={config_id}
```
category=`experiment_summary`, confidence=`1.0`, source=`thread:{tid}, config_id:{cid}`

### 1.3 新增函数（seal_handoff_tools.py）

- `_extract_n_per_group(workspace)` → 优先 metadata.n_per_group > metrics_summary.n > "unknown"
- `_extract_key_findings_count(workspace)` → key_findings 长度
- `_write_experiment_summary_memory(workspace, paradigm, config_id, thread_id, user_id)`

---

## 2. S7 实施细节

### 2.1 新文件

- `tools/builtins/present_assumptions.py`：
  - `_build_assumptions_markdown(ctx, plan, da_handoff)` — 纯函数
  - `present_assumptions_tool` — @tool 包装
  - 输出 `<details>` Markdown 折叠卡片

### 2.2 渲染内容

```markdown
<details>
<summary>分析假设摘要 (config_id=abc123...)</summary>
### 参数覆盖
### 运行时参数
### 数据质量
### 参数审计
</details>
```

### 2.3 lead prompt

`<critical_reminders>` 加了一行建议性指引（非强制 GateProvider）。

---

## 3. S8 实施细节

### 3.1 数据库 schema 变更

```sql
ALTER TABLE feedback ADD COLUMN paradigm VARCHAR(64) DEFAULT NULL;
```

历史数据 paradigm 为 NULL，不影响现有查询。

### 3.2 FeedbackRow 新增字段

- `paradigm: Mapped[str | None]` — nullable String(64)

### 3.3 新增查询方法

- `FeedbackRepository.list_prior_corrections(paradigm, user_id, limit)`：
  - WHERE paradigm = ? AND verdict IN ('needs_fix', 'wrong')
  - ORDER BY created_at DESC
  - 用于 lead prompt 注入历史纠正

### 3.4 新增 API 端点

- `GET /api/feedback/prior_corrections?paradigm=epm&limit=5`
  - 返回指定范式的最近纠正记录
  - 需要 `require_permission("threads", "read")`

### 3.5 提交时自动读 paradigm

- `submit_feedback` 调用 `_read_paradigm_from_context(thread_id)` 从 experiment-context.json 读 paradigm
- 写入 feedback 表的 paradigm 字段

### 3.6 lead prompt 注入

- `prompt.py` 新增 `_get_prior_corrections_context()`：
  - 读当前 thread 的 experiment-context.json 获取 paradigm
  - 查 FeedbackRepository 获取最近 3 条纠正
  - 渲染为 `<prior_corrections>` 段
  - 所有异常 catch，返回 ""（非阻塞）
- `SYSTEM_PROMPT_TEMPLATE` 加 `{prior_corrections_context}` 占位符

### 3.7 注意事项

- `_get_prior_corrections_context()` 在 prompt 组装时同步调用
  - 如果在 async event loop 内，用 ThreadPoolExecutor offload
  - 否则用 `asyncio.run()`
- SQLite `ALTER TABLE ADD COLUMN` 只在首次运行时需要
  - SQLAlchemy `create_all()` 会自动添加新列到新建表
  - 已有数据库需要手动 migration（或让 ORM 自动处理）
- harness → app boundary：`_get_prior_corrections_context` 通过 lazy import 访问 persistence 层
  - 这是因为 prompt.py 在 harness 层，FeedbackRepository 在 harness/persistence 层
  - 实际没有违反 boundary（deerflow.persistence 是 harness 的一部分）

---

## 4. 测试运行方式

```bash
cd packages/agent/backend
source /home/wangqiuyang/noldus-insight/packages/agent/backend/.venv/bin/activate
PYTHONPATH="$PWD/packages/harness:$PWD/../../packages/ethoinsight:$PWD" python -m pytest tests/ -q

# S6 only: 21 tests
# S7 only: 14 tests
# S8 only: 12 tests
```

---

## 5. 文件改动汇总

| 文件 | Sprint | 改动类型 |
|---|---|---|
| `seal_handoff_tools.py` | S6 | 修改（+113 行） |
| `present_assumptions.py` | S7 | 新建（169 行） |
| `tools/builtins/__init__.py` | S7 | 修改（+2 行） |
| `agents/lead_agent/prompt.py` | S7+S8 | 修改（+96 行） |
| `persistence/feedback/model.py` | S8 | 修改（+3 行） |
| `persistence/feedback/sql.py` | S8 | 修改（+39 行） |
| `app/gateway/routers/feedback.py` | S8 | 修改（+81 行） |
| `app/gateway/app.py` | S8 | 修改（+3 行） |
| `tests/test_s6_memory_fact.py` | S6 | 新建（299 行） |
| `tests/test_s7_present_assumptions.py` | S7 | 新建（193 行） |
| `tests/test_s8_feedback_corrections.py` | S8 | 新建（218 行） |

**Total**：11 文件，~1216 行新增
