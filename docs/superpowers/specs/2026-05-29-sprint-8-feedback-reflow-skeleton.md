# Sprint 8 设计骨架 — feedback verdict 回流到 prompt

**类型**：设计骨架版（非代码核验版——实施前需按 §核验清单升级）
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 8（原 Sprint 7）+ 2026-05-29 三档重排 P2（v0.2，优先级最低）
**估期**：~2 周
**前置**：无硬依赖（独立实施）。但**自标最低优先**——长期靠微调底模型，本 sprint 是短期补丁，微调到位后收益减半

---

## 0. 目标与原则

**目标**：专家三按钮反馈（correct / needs_fix / wrong + revised_text）已落 SQLite，但**没回流到 prompt**。agent 下次遇到同范式时不知道自己上次被纠正过。这是主人"不重复犯同一个错"的闭环。

**核心机制**：lead 构建 prompt 前，查"同 paradigm 上次 verdict ≠ correct 的 revised_text"，作为 `{prior_corrections}` 注入 system prompt。

**铁律**：
1. **短期补丁定位**：本 sprint 是微调到位前的过渡。不要过度工程——简单的"查最近 N 条同范式纠正 → 注入 prompt"即可，别建复杂的纠正知识库
2. **per-user 隔离**：纠正只回流给同一用户（feedback 已有 user_id），不跨用户污染（Tier 4 现有机制）
3. **只读不改训练数据**：本 sprint 读 feedback 表注入 prompt，不动训练数据飞轮（那是 TrainingDataMiddleware 的事，正交）

---

## 1. grill 复审发现（实施前必读）

### 🔴 前置数据缺口：feedback 表无 paradigm 字段（已核实）
roadmap 原文已预判，2026-05-29 核实坐实：`FeedbackRow`（`persistence/feedback/model.py:13`）字段为 `feedback_id / run_id / thread_id / user_id / message_id / rating / comment / verdict / revised_text / created_at`——**没有 paradigm**。

"查同 paradigm 上次 verdict ≠ correct"无法直查，必须先解决：
- **方案 A（推荐）**：feedback 表加 `paradigm: str | None` 字段（Alembic migration + 提交时从 `experiment-context.json` 读 paradigm 一并落盘）。新数据 → 直查
- **方案 B**：表不动，查询时 thread_id → 反查 `experiment-context.json` 取 paradigm。性能差但兼容历史数据

**实施按 A**，历史 feedback 的 paradigm 留 null（不参与回流）。

### 🟡 是正经 persistence 层，不是裸 SQLite
feedback 是 SQLAlchemy + Alembic（`persistence/feedback/{model,sql}.py` + `migrations/versions/`），不是手写 SQLite。加字段**必须走 Alembic migration**（仿 `20260512_1200_feedback_verdict_revised.py` 这个加 verdict 的样板），不能手写 ALTER。

### 🟡 注入机制有现成参考
lead prompt 已有 `{memory_context}` 插槽（`prompt.py:403`）+ `_get_memory_context()`（`:625`，per-user 隔离）。`{prior_corrections}` 注入仿此机制，不要新造一套。

---

## 2. 设计（骨架）

**数据流**：用户提交 feedback（verdict + revised_text）→ 落盘时连 paradigm 一起存 → 下个会话同范式时，lead 构建 prompt 前 fetch 该 user + 该 paradigm 的近 N 条 verdict≠correct 纠正 → 注入 `{prior_corrections}` 插槽。

**改动（骨架，行号待实施时核验）**：

- `persistence/feedback/model.py`：`FeedbackRow` 加 `paradigm: Mapped[str | None]`（`:44` created_at 之前）
- `persistence/migrations/versions/`：新增 Alembic migration 加 paradigm 列（仿 verdict 那个）
- `app/gateway/routers/feedback.py`：
  - `FeedbackRequest`（`:32`）提交时从 `experiment-context.json` 读 paradigm 一并写入
  - 增查询：`GET /api/feedback/prior_corrections?paradigm=epm&user_id=X&limit=3` → 返回同 paradigm verdict≠correct 的 revised_text（按 created_at 倒序）
- `agents/lead_agent/agent.py`：`make_lead_agent` 构建 prompt 前 fetch prior_corrections
- `agents/lead_agent/prompt.py`：
  - 模板加 `{prior_corrections}` 插槽（仿 `{memory_context}` @ `:403`）
  - 加 `_get_prior_corrections(paradigm, user_id)` helper（仿 `_get_memory_context` @ `:625`）
  - 文案正面化（deepseek 正面提示原则）："以下是你过去在 <paradigm> 分析中收到的专家修订，参考它们避免重复同类问题：…"

**注入条件**：仅当 paradigm 已知（Gate 1 已过）且该 user+paradigm 有 verdict≠correct 记录时才注入。无记录 → 插槽渲染空，不占 token。

---

## 3. 实施前核验清单（骨架→核验版升级步骤）

1. `FeedbackRow` 当前字段 + Alembic migration 怎么写（看 `20260512_1200_feedback_verdict_revised.py` 样板）
2. feedback 提交时 experiment-context.json 是否已可读（paradigm 落盘路径）
3. `_get_memory_context` / `{memory_context}` 的真实注入机制（prior_corrections 仿它）
4. FeedbackRepository 查询接口现状（在哪加 prior_corrections 查询）
5. per-user 隔离：fetch prior_corrections 时 user_id 怎么解析（与 memory 同源）
6. prompt token 预算：prior_corrections 注入会不会撑爆（限 N 条 + 截断）

---

## 4. 验收（骨架）

1. 提交 feedback（verdict=wrong + revised_text + 当前是 EPM）→ feedback 表该行 paradigm=epm
2. 同 user 下个会话再做 EPM → lead system prompt 的 `{prior_corrections}` 含上次的 revised_text
3. 不同 paradigm（如 FST）→ 不注入 EPM 的纠正（paradigm 隔离）
4. 不同 user → 不注入别人的纠正（per-user 隔离）
5. 无纠正记录 → 插槽空，不占 token
6. 历史 feedback（paradigm=null）→ 不参与回流，不报错
7. 全量测试不退化（含 Alembic migration 的 up/down 测试）

## 5. 不在范围
- ❌ 复杂的纠正知识库 / 向量检索（本 sprint 是简单"查最近 N 条注入"短期补丁）
- ❌ 动训练数据飞轮（TrainingDataMiddleware 正交，不碰）
- ❌ 跨 user 聚合纠正（per-user 隔离铁律）
- ❌ 自动应用纠正改 agent 行为（只注入 prompt 让 LLM 参考，不强制）

## 6. 与微调的关系（定位提醒）
本 sprint 是**微调到位前的短期补丁**。Qwen3 微调上线后，"不重复犯错"应主要靠底模型，prompt 回流收益减半。所以：**不要在本 sprint 投入过多**，简单可用即可；v0.2 微调到位后评估是否保留/退役。
