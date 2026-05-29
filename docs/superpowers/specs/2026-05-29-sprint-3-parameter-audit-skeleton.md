# Sprint 3 设计骨架 — data-analyst 参数审计

**类型**：设计骨架版（非代码核验版——实施前需按 §核验清单升级）
**对应**：[roadmap v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 3 + 2026-05-29 grill 复审
**估期**：1 周
**前置**：Sprint 2a（catalog 参数）+ 2b（参数管线，`parameters_used` 字段）**均已实施** ✅

---

## 0. 目标与原则

**目标**：参数现在可配置了（2a/2b），但没人告诉用户"你的数据用当前参数不合适"。data-analyst 比对**参数**与**实际数据分布**，发现 mismatch（如 velocity 中位数 5mm/s 但阈值 30mm/s）→ 生成警告。

**铁律：只警告，不调参。** 调参需用户显式确认（这是 roadmap 的"主人≠擅自调参=暴露假设"核心纠正）。data-analyst 多走半步改参 = 伪造证据。

---

## 1. grill 复审发现（实施前必读）

### 🔴 热点警告：data_analyst.py 是 seal bug 同一文件
Sprint 3 要改 `data_analyst.py` 的 workflow（加参数适配性检查段）。**这正是 5.7 seal bug 的文件**——5.7 根因是"加 step 2.5/2.6 后没把原 step 2 改成 2.7，编号冲突致 LLM 漏调 seal"。

**实施 agent 必做**：加 workflow step 后 `grep -n "^\s*[0-9]\+\." data_analyst.py` 验证步骤编号唯一、连续。这是 [[feedback_single_source_of_truth]] 之外的硬纪律（5.7 血泪）。

### 🔴 多 sprint 同文件热点
Sprint **3 / 5 / 5.8 / 6** 都改 `data_analyst.py` 的 workflow 或 handoff 路径。5.8 已实施（改了 executor 调用 data_analyst 的路径，未改 data_analyst.py 本身）。**实施顺序**：3 改 workflow + handoff schema 加字段，要确认与 5（quality gate）/6（memory）的改动不冲突。建议 3 先于 5/6。

### 依赖已就位
`MetricStat.parameters_used`（Sprint 3 要读来比对）由 2b 写入，**已实施** ✅（2b commit 5373b5b7，5 跳链路跳 5 落地）。

---

## 2. 设计（骨架）

**核心数据流**：data-analyst 已经 read handoff_code_executor.json（拿 per_subject 数据分布 + MetricStat.parameters_used）。参数审计 = 在现有 workflow 里加一段：
- 从 `MetricStat.parameters_used` 取本次实际用的参数值（如 velocity_threshold=30）
- 从 handoff 数据分布取实际值（如 velocity 中位数）
- 比对：实际值远低于/高于阈值 → 生成 `parameter_audit` 条目（warning，不改参）

**改动（骨架，行号待实施时核验）**：

- `subagents/builtins/data_analyst.py`：
  - workflow 加"参数适配性检查"段（**插在现有 step 之间，必须重排编号 + grep 验证唯一**）
  - 检查逻辑：遍历 parameters_used，对每个可调参数（velocity_threshold 等），从数据分布取对应实际统计量比对
  - 结果写入新 handoff 字段 `parameter_audit`
- `subagents/handoff_schemas.py`：`DataAnalystHandoff` 增 `parameter_audit: list[...]` 字段（**与 5.5 的 key_findings 非空检查正交，不冲突**；字段 schema 在此定义=single-source）
  - 建议子模型 `ParameterAuditFinding`（参数名 / 实际值 / 当前阈值 / mismatch 方向 / 建议）
- `subagents/builtins/data_analyst.py` gate_signals：增 `parameter_audit_findings_count`

**mismatch 判据**：roadmap 原文提了 5 元枚举 `mismatch_kind`（实施时从 review-packages 或与行为学同事确认具体阈值——**判据是领域知识，不要工程拍脑袋**，呼应 Sprint 4 同款 SSOT 问题）。

---

## 3. 实施前核验清单（骨架→核验版的升级步骤）

实施 agent 开工前必须核验（像 5.8 那样）：
1. `data_analyst.py` 当前 workflow 的步骤编号结构（确认在哪插、怎么重排不冲突）
2. `MetricStat.parameters_used` 的真实结构（2b 写入的格式，决定怎么取参数值）
3. handoff_code_executor.json 里数据分布字段的真实形态（per_subject / metrics_summary 里有哪些统计量可比对）
4. `DataAnalystHandoff` 当前字段（确认加 parameter_audit 不与 5.5/5.8 改动撞）
5. mismatch 阈值判据：grep review-packages 或问行为学同事（不工程拍脑袋）

---

## 4. 验收（骨架）

1. 跑 velocity 中位数远低于阈值的 case → handoff `parameter_audit` 出现该 mismatch 条目，gate_signals.parameter_audit_findings_count > 0
2. **agent 只警告未改参**（参数仍是 catalog default / 用户 override，没被 data-analyst 偷改）
3. 加 workflow step 后 grep 编号唯一（防 seal bug 复发）
4. 全量测试不退化

## 5. 不在范围
- ❌ 自动调参（铁律：只警告）
- ❌ mismatch 阈值的领域判据由工程定义（应来自 review-packages / 同事）
- ❌ 改 data_analyst.py 以外的 subagent
