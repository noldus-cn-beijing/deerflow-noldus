# Sprint 3 设计骨架 — data-analyst 参数审计

**类型**：可实施版（2026-05-29 从骨架升级——锚点已对 dev HEAD `599ec558` 核验，见 §3.5；实施前 git pull 复核行号）
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

## 3.5 实施锚点（已对 dev HEAD `599ec558` 核验 — 本节把骨架升级为可实施）

**上面 §3 的核验清单已由本节代为完成。实施 agent 仍应 `git pull` 后复核行号（会漂）。**

### 3.5.1 data_analyst.py workflow 插入点
`subagents/builtins/data_analyst.py` 的 `<workflow>` 段（约 line 72-119）现有编号：
```
1. 读输出宪法          (line 73)
2. read handoff_code_executor.json  (line 74)
2.5 读 quality warnings (line 76)
2.6 按范式 read 判读文档 (line 87)
2.7 一次性完成核心分析推理 (line 95)   ← 参数审计是"分析"的一部分，并入此步思考
3. 封存 handoff: 调 seal_data_analyst_handoff (line 113)
4. 最终 AIMessage 摘要  (line 117)
```
**插入点**：新增 `2.8 参数适配性审计`，**插在 2.7 之后、step 3 之前**。
**🔴 编号纪律（5.7 seal bug 教训，[[feedback_single_source_of_truth]]）**：插入后必须 `grep -nE "^[0-9]+\.|^[0-9]+\.[0-9]+" data_analyst.py` 确认编号唯一、连续（2/2.5/2.6/2.7/2.8/3），不得与 line 29 input 说明段的 `2.` 混淆（那是另一段，不算 workflow 步骤）。

### 3.5.2 数据来源（参数审计读什么、和什么比）
- **用了什么参数**：`MetricStat.parameters_used`（`handoff_schemas.py:67`，dict 如 `{"velocity_threshold": 30.0}`），从 `handoff_code_executor.json` 的 `metrics_summary`（`:204`，`dict[str, dict[str, MetricStat]]`）逐条取
- **数据实际分布**：`MetricStat` 自身已含统计量（`:47-56`，mean 等；注释 `:51` 说 median/IQR 可加）+ `per_subject`（`:208`，`dict[str, dict[str, Any]]`）取各 subject 的实际值
- 比对：参数值 vs 数据分布统计量 → 生成 `ParameterAuditFinding`

### 3.5.3 mismatch 判据（关键：分两类，一类现成不卡同事，一类用保守默认）

**这是本 sprint 最容易误判"全卡同事"的地方。实测核验结论：判据分两类。**

| 参数类 | 判据现状 | 实施怎么做 |
|---|---|---|
| **FST/TST 的 pendulum 8 参数** | ✅ **同事已写，不卡** | 直接照 `docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md` §3/§4：如 `periodicity` 钟摆段 >0.5 / 挣扎段 <0.3；`ANALYSIS_WINDOW` 覆盖 3~5 个钟摆周期；§7.3 品系/体重差异影响 PERIOD_MIN/MAX。把这些写进比对逻辑 |
| **`velocity_threshold` + 焦虑范式独有阈值** | ❌ 精确判据缺（issue #63 等同事 @Qukoyk） | **用保守数学默认兜底先跑**：`threshold_too_high` = 数据 velocity 中位数 < 参数值 ÷ 3；`threshold_too_low` = 中位数 > 参数值 × 3。在 finding 的 suggestion 里标注"判据为保守默认，精确物种判据待 issue #63"。**不阻塞实施** |

**严禁**：工程为 velocity 编造"鱼类该是 5"这种领域数字（issue #63 归同事）。保守数学判据只判"分布与参数严重背离"这个**统计事实**，不替同事下"该调到多少"的领域结论。

### 3.5.4 schema 改动锚点
- `handoff_schemas.py`：`DataAnalystHandoff`（`:323`）加 `parameter_audit_findings: list[ParameterAuditFinding]`（新子模型，参数名/实际值/当前阈值/mismatch_kind 5元枚举/suggestion/blocks_downstream）；`GateSignals`（`:129`）加 `parameter_audit_findings_count` + `parameter_audit_critical_count`
- `seal_data_analyst_handoff` tool（`tools/builtins/seal_handoff_tools.py`）加 `parameter_audit_findings` 参数
- **正交确认**：与 5.5（key_findings 非空校验）/5.8（seal-resume）改的字段不撞——它们改 executor 的校验/补轮路径，本 sprint 加 DataAnalystHandoff 新字段，正交

> #### ⚠️ schema 细节的权威来源 = 5-28 原始 spec（实施 agent 必读，勿凭推断）
> 本 spec 是增量升级版（补锚点 + 判据分类 + TDD）；`ParameterAuditFinding` 的**完整 schema 定义在原始 spec** [`2026-05-28-sprint-3-data-analyst-parameter-audit-design.md`](2026-05-28-sprint-3-data-analyst-parameter-audit-design.md) §2.1，**以它为准**。关键已定义值（不要自己发明）：
>
> **`mismatch_kind` = 5 元 Literal（原始 spec §2.1 line 68-73）**：
> 1. `threshold_too_high` — 阈值远高于数据上限/中位数
> 2. `threshold_too_low` — 阈值远低于数据下限/中位数
> 3. `window_too_wide` — 窗口超出 trial 时长
> 4. `window_too_narrow` — 窗口过窄无法捕捉事件
> 5. `category_mismatch` — 离散参数取值与 paradigm 不符
>
> 注意：**不是** agent 可能推断的 `window_too_short`/`periodicity_mismatch`/`parameter_inapplicable`。pendulum 的窗口问题归 `window_too_wide/narrow`；periodicity 阈值偏移归 `threshold_too_high/low`；枚举参数不符归 `category_mismatch`。首发只这 5 类，扩枚举需显式改 Literal。
>
> **`severity` = `Literal["critical","warning","info"]`（原始 spec §2.5 line 199-202）**：
> - `critical`（且 `blocks_downstream=true`）：**所有 subject** 的某指标都受影响
> - `warning`：**≥50% subject** 受影响
> - `info`：单纯阈值落在边界值附近
>
> 即 critical 由**受影响 subject 比例（全部）**定义，**不是**由 mismatch_kind 类型定义。`parameter_audit_critical_count = sum(severity==critical AND blocks_downstream==true)`（原始 spec §2.5 line 211）。
>
> **数学判据阈值（原始 spec §2.5 line 192-197，velocity/窗口的保守默认）**：threshold_too_high = `used > p90×3`；threshold_too_low = `used < p10÷3`；window_too_wide = `window > trial_dur×0.9`；window_too_narrow = `window < trial_dur×0.05`。这些是工程保守默认（统计背离判据），velocity 的**精确物种判据**待 issue #63。

---

## 4. 验收（可实施版）

- [ ] `ParameterAuditFinding` schema：5 字段 + 5元枚举 mismatch_kind + Pydantic 校验
- [ ] `DataAnalystHandoff.parameter_audit_findings` + `GateSignals` 两个计数字段
- [ ] `seal_data_analyst_handoff` 透传 findings 到 handoff JSON（atomic write）
- [ ] data_analyst.py 加 step 2.8（pendulum 用同事判据 + velocity 用保守默认）+ **grep 编号唯一**（防 seal bug）
- [ ] **pendulum case**：FST/TST 数据 periodicity 异常 → finding 引用同事文档判据
- [ ] **velocity case**：velocity 中位数 5、阈值 30 → `threshold_too_high` finding（保守判据），suggestion 标注"精确判据待 #63"
- [ ] **只警告未改参**：参数仍是 catalog default/override，data-analyst 没偷改
- [ ] 全量测试不退化（基线实施前 `make test` 取真值）
- [ ] `git grep "parameter_audit"` 在 handoff_schemas / seal_handoff_tools / data_analyst.py 三处就位

## 5. 实施顺序（TDD task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | `ParameterAuditFinding` schema + DataAnalystHandoff/GateSignals 字段 + 6 个 schema 测试 | 0.5 天 |
| T2 | `seal_data_analyst_handoff` 加 parameter_audit_findings 参数 + 3 个 seal 测试 | 0.25 天 |
| T3 | data_analyst.py 加 step 2.8（pendulum 判据照同事文档 + velocity 保守默认）+ grep 编号唯一 | 1 天 |
| T4 | 集成测试：pendulum case + velocity case + "只警告未改参" | 0.5 天 |
| T5 | dogfood + 全量回归 | 0.5 天 |
| **合计** | | **~2.75 天** |

## 6. 与其他工作的关系（实施 agent 必读）

- **issue #63（@Qukoyk）**：velocity + 焦虑范式独有参数的精确判据/物种值在此。本 sprint 用保守默认先跑，#63 回来后再调精度。**不等 #63 也能实施**
- **Sprint 4（调参指南）**：本 sprint 的 finding.suggestion 会指向"参考调参指南"。Sprint 4 未做时，suggestion 文字仍合理（只是用户找不到指南内容）。**3 和 4 是一对，建议协调；demo 若不触发参数审计则两者皆可缺**
- **编排路径 SSOT 阶段 A**：若该 agent 在并发改 `lead_agent/prompt.py`，本 sprint **不碰 lead prompt**（只在 §2.6 的 lead 播报段加一句参数审计播报——若与编排 SSOT agent 撞，留 `# TODO(orchestration-ssot)` 标记，谁先合谁后 rebase）
- **demo 影响**：本 sprint 是"旁支增强"，不在端到端主干。缺它 demo 主流程照常跑通（5-29 FST dogfood 已证）

## 7. 不在范围
- ❌ 自动调参（铁律：只警告）
- ❌ 工程为 velocity 编造领域数字（归 issue #63 / 同事）
- ❌ 改 data_analyst.py 以外的 subagent（lead prompt 播报段除外，且最小改动）
- ❌ 前端 banner 渲染 parameter_audit（Sprint 7 假设面板做）
