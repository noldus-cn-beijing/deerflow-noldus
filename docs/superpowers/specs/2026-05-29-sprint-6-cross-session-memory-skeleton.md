# Sprint 6 设计骨架 — 跨会话范式 memory（确定性 fact 写入）

**类型**：设计骨架版（非代码核验版——实施前需按 §核验清单升级）
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 6（3 周 → 0.5 周，复用 deerflow facts 通道）+ 2026-05-29 grill 复审 🟢（健康，2 个注意）
**估期**：0.5 周
**前置**：Sprint 0（handoff schema + `seal_report_writer_handoff` 已建）✅ + Sprint 4.5（`analysis_config_id`，fact content 引用它——4.5 未实施则 config_id 段留 PENDING 占位，不阻塞）

---

## 0. 目标与原则

**目标**：用户上周做 EPM、这周做 OFT——agent 应能跨会话汇聚证据（"你上次 EPM 也焦虑，OFT 现在也焦虑——汇聚置信度更高"）。这是"主人"的质变能力。

**核心设计（grill 锁定 B）：复用 deerflow 现有 facts 通道，不新建顶层结构。** report-writer 完成时**确定性写入**一条 `category="experiment_summary"` 的 fact（`confidence=1.0`），**完全不走 LLM 抽取**。

**铁律：精确数字不依赖 LLM 抽取。** 审计发现 MemoryMiddleware 的 LLM 抽取通道会把数字漂移成字符串（"immobility=42.3s" → "约 40 秒"）。experiment_summary 直接从 ReportWriterHandoff 的结构化字段拼，绕开 updater。LLM 抽取通道继续做它的事（user preferences/style），两条路径正交。

---

## 1. grill 复审发现（实施前必读）

### 🔴 与 5.8 改同一 seal 路径——需协调顺序
Sprint 6 要在 `seal_report_writer_handoff`（`seal_handoff_tools.py:278`）内部、atomic write + manifest 之后**额外调** `add_fact`。**5.8 seal-resume 已实施**（改的是 executor 调 subagent 的补轮路径，未改 `seal_report_writer_handoff` 函数体本身——已核验 5.8 代码在 `executor.py`，不在 `seal_handoff_tools.py`）。所以**当前无直接冲突**，但实施 agent 必须：
- 开工前 `git grep -n "seal_report_writer_handoff" -- '*.py'` 确认 5.5（内容校验）有没有也插进同一函数（5.5 挂载点在 `executor.py:146`，理论上正交，但要核）
- add_fact 调用放在 atomic write **成功之后**，且**包 try/except**——写 fact 失败绝不能让 seal 失败（seal 是主路径，memory 是增益）

### 🔴 PRD §8 把"跨范式关联"划出 MVP——确认 v0.1 是否要做
roadmap line 111 + line 118 战略提醒：跨会话 memory 属"主人哲学上层建筑"，**PRD 公测标准是"6 范式跑得准、不翻车、端到端"**。golden-cases 仍为空、OFT/LDB/Zero Maze 无端到端 dogfood。**若 v0.1 资源紧张，Sprint 6 应让位于"范式端到端验证 + golden-case"。** 本骨架不擅自重排优先级，仅标记：实施前与用户确认 Sprint 6 是否进 v0.1（写 fact 的地基便宜，但"跨范式汇聚证据"的 lead prompt 编排可能是 v0.2）。

### 🟡 `add_fact()` helper 不存在，需新增
已核验 `storage.py` 只有 `load`/`save`/`reload`/`reload`，**没有 `add_fact` 公共方法**。roadmap line 436 已预判（"若没有则新增"）。所以这不是"复用现成"，是"新增一个薄 helper"：load → append → eviction（沿用 max_facts 逻辑）→ save。实施 agent 别假设能直接调到现成方法。

### 🟡 eviction 不能挤掉 confidence=1.0 的 experiment_summary
现有 eviction 按 confidence 排序、max_facts=100。experiment_summary 用 confidence=1.0（最高），天然不被低 confidence 的 user preference 挤掉。但要**核验现有 eviction 逻辑是否真按 confidence 降序保留**（验收 §3 有这条），别假设。

---

## 2. 设计（骨架）

**核心数据流**：report-writer seal 时 → 从 ReportWriterHandoff 结构化字段拼 fact content → 调新增的 `storage.add_fact(user_id, fact)` → 写入 `{base_dir}/users/{user_id}/memory.json` 的 facts[]。下个会话 lead system prompt 自动注入 top facts（按 confidence 排序，experiment_summary 最优先）。

**改动（骨架，行号待实施时核验）**：

- `agents/memory/storage.py`：
  - **新增** `add_fact(user_id, content, category, confidence, source)` 公共方法（load → append `{id, content, category, confidence, source, created_at}` → eviction → save）
  - id 生成 / created_at 用现有 `utc_now_iso_z()`（storage.py:19，已存在）
  - eviction 复用现有 max_facts 逻辑（**核验它在哪、是否按 confidence 降序**）
- `tools/builtins/seal_handoff_tools.py:278`（`seal_report_writer_handoff`）：
  - atomic write + manifest 之后，**try/except 包裹**调 `add_fact`
  - fact content 模板（从 ReportWriterHandoff 字段拼，**不走 LLM**）：
    ```
    "{paradigm} analysis on {date}: n_per_group={n}, key metrics: {m1}={v1}, {m2}={v2}; effect_size={es}; analysis_config_id={config_id}"
    ```
    - `category="experiment_summary"`，`confidence=1.0`
    - `source=f"thread:{thread_id}, analysis_config_id:{config_id}"`
    - date / paradigm / metrics 从 handoff 结构化字段取（`config_id` 来自 4.5；4.5 未实施则填 `"PENDING"`）
- **不动**：MemoryMiddleware / memory updater / memory config / memory prompt（所有现有路径不变——这是设计铁律，不要顺手改）

**fact content 取数来源**：`ReportWriterHandoff`（`handoff_schemas.py:370`，含 `report_path` 等）+ 上游 handoff 的 paradigm/metrics（实施时核验 report-writer seal 能拿到哪些字段，可能要从 data_analyst handoff 透传摘要数字）。

---

## 3. 实施前核验清单（骨架→核验版的升级步骤）

实施 agent 开工前必须核验（像 5.8 那样，别假设）：
1. `storage.py` 的 eviction 逻辑在哪、是否按 confidence 降序保留（决定 experiment_summary 是否安全）
2. `seal_report_writer_handoff` 函数体当前结构（atomic write 在哪行、5.5 内容校验有没有也插进来）
3. report-writer seal 时能拿到哪些结构化字段（paradigm/n/metrics/effect_size/config_id 是否都在手，还是要从上游 handoff 透传）
4. `analysis_config_id` 是否已实施（4.5）——未实施则 fact content 该字段填 "PENDING"
5. `get_memory_storage()`（storage.py:196）拿到的实例怎么解析 user_id（per-user 隔离路径）
6. **与用户确认 Sprint 6 是否进 v0.1**（PRD 把跨范式划出 MVP，见 §1 🔴）

---

## 4. 验收（骨架）

1. 完成一次 EPM 分析后，`memory.json[users/{user_id}].facts` 出现一条 `category=experiment_summary` 的 fact，content 含**精确数字**（与 handoff 字段一致，非 LLM 抽取/漂移）
2. 第二个会话做 EPM 时，lead system prompt 的 `<memory>` 段含上次 fact（confidence=1.0 排序最优先）
3. 跑 200 次分析后 memory.json 仍只 100 条 facts（max_facts eviction）；confidence=1.0 的 experiment_summary **不被**低 confidence 的 user preference 挤掉
4. **add_fact 失败不让 seal 失败**：mock storage 抛错 → seal 仍 COMPLETED + handoff 正常落盘（try/except 生效）
5. 全量测试不退化

## 5. 不在范围
- ❌ 新建顶层 ExperimentSummary 结构（grill 锁定 Q9/B：走 facts 通道）
- ❌ 改 MemoryMiddleware / updater / LLM 抽取通道（正交，不动）
- ❌ "跨范式汇聚证据"的 lead prompt 编排（写 fact 是地基；汇聚呈现可能是 v0.2，见 §1 🔴）
- ❌ fact content 走 LLM 抽取（铁律：确定性写入）
