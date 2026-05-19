# 2026-05-18 Subagent 角色拆分 + Capability-Exposure 重构 Spec 交接

> **状态**：grill-me 12 轮完成，所有设计分支决策已做。下一步：新 agent 写完整 spec。
>
> **创建者**：上一会话 Claude（grill-me 主持人）

---

## 预读清单（新 agent 按顺序读）

1. **本 handoff**（你正在读）
2. [2026-05-18 Lead Handoff 消费 + 角色拆分设计讨论交接](2026-05-18-lead-handoff-consumption-and-role-split-handoff.md)（诊断原文）
3. `~/.claude/projects/-home-wangqiuyang/memory/MEMORY.md`（auto-memory 索引）
4. `~/.claude/projects/-home-wangqiuyang/memory/project_2026-05-18_lead_not_reading_handoff.md`
5. `~/.claude/projects/-home-wangqiuyang/memory/project_2026-05-14_e2e_trajectory_failure.md`
6. `~/.claude/projects/-home-wangqiuyang/memory/feedback_single_source_of_truth.md`（用户硬性原则）
7. [CLAUDE.md](../../../CLAUDE.md)（仓库全貌）
8. `packages/agent/backend/CLAUDE.md`

---

## 1. 这次 grill-me 做了什么

基于 2026-05-18 handoff 的诊断（lead 不读 handoff/catalog 缺 fallback/code-executor 角色过载等 5 重根因），grill-me 12 轮把设计从"修 5-18 bug"扩展到"重切 subagent 边界 + capability-exposure 架构 + lead 交互哲学"。

12 个决策分支全部走完，用户每轮确认。

---

## 2. 核心设计哲学（用户自己的洞察）

| # | 洞察 | 对应实现 |
|---|---|---|
| A | "lead 大包大揽太多，找不到数据" | Lead 退化为 router，不读 handoff/数据，只做交互 |
| B | "subagent 数 = 用户需求数" | 按用户意图切 subagent，不按工种切 |
| C | "最大化利用 deerflow + 参考 Claude Code 好地方" | Capability-exposure：subagent 宣告 contract，lead 按 contract 调度 |
| D | "lead 持交互 know-how，不持执行 know-how" | Lead 知道如何反问、如何分类意图、如何串调；不知道如何跑脚本/选图/解读 |
| E | "capability 暴露给主 agent，主 agent 设 plan 调用" | SubagentConfig 加 when_to_use / input_contract / output_contract / required_upstream_handoffs |

---

## 3. 终态架构

### 3.1 Lead Agent（router + planner）

**持有**：意图分类 / 用户交互 gate / 范式识别（通过 skill）/ 工具编排

**不持有**：catalog 字段 / 脚本路径 / handoff schema / glob 模式 / 图种知识 / 指标计算逻辑

**形态**：prompt 骨架 ~200 行 + `ethoinsight-lead-interaction` skill（详细交互手册）

**范式识别流程**（Q11 ε-final）：
1. 读 `ethovision-paradigm-knowledge` skill
2. 用 skill 知识 + 用户 prompt + 文件名推断（**不 read raw txt**）
3. 唯一高置信 → `set_experiment_paradigm` 落盘
4. 多候选 / 无法分辨 → `ask_clarification` gate 反问（**禁止默认猜测**）
5. 用户确认 → 落盘 → 派 subagent

### 3.2 5 个 Subagent

| # | 名字 | 用户意图 | 依赖 handoff | 核心工作 |
|---|---|---|---|---|
| 1 | **code-executor**（改） | E2E（第一棒）| 无 | 读 plan_metrics.json → bash metrics + stats → 写 handoff_code_executor.json；**不跑 charts** |
| 2 | **data-analyst**（微量改） | E2E（第二棒）| code-executor | 读 handoff_code_executor.json → 解读 → 写 handoff_data_analyst.json |
| 3 | **chart-maker**（新建） | CHART 单点 / E2E 第三棒 | code-executor | **自己跑** `catalog.resolve --mode charts` → plan_charts.json；读 handoff + skill 决策树 → bash 跑图脚本；含 fallback 处理 |
| 4 | **report-writer**（微量改） | REPORT 单点 / E2E 末棒 | code-executor, data-analyst（+可选 chart-maker）| 读所有 handoff → 6 段骨架 report.md |
| 5 | **knowledge-assistant**（微量改） | QA_FACT / QA_KNOWLEDGE | QA_FACT 时依赖 code-executor + data-analyst；QA_KNOWLEDGE 时无 | 行为学问答 + 追问 |

### 3.3 6 类用户意图

| 意图 | 触发 | 派遣链 |
|---|---|---|
| **E2E_FULL** | 上传数据 + "分析并画图"等复合 | code-executor → data-analyst → chart-maker → ask_clarification(report) |
| **E2E_MIN** | 上传数据 + "分析一下" | code-executor → data-analyst → ask_clarification(看结果/画图/报告/都要) |
| **CHART** | 已有 handoff + "再画几个图" | chart-maker（单派）|
| **REPORT** | 已有 handoff + "出报告" | report-writer（单派）|
| **QA_FACT** | 已有 handoff + 追问具体数据 | knowledge-assistant（授权 handoff）|
| **QA_KNOWLEDGE** | 领域知识/概念问题 | knowledge-assistant（不授权 handoff）|
| **CLARIFY** | 意图模糊 | ask_clarification |

### 3.4 SubagentConfig 扩展（capability metadata）

`SubagentConfig` 新增 4 字段：

```python
@dataclass
class SubagentConfig:
    name: str
    description: str                    # lead 看的简介
    when_to_use: str                    # 适用场景（取代 lead prompt 里的"何时派"）
    input_contract: str                 # lead 派遣时该传什么
    output_contract: str                # subagent 返回什么
    required_upstream_handoffs: list[str]  # [Q9] 自动注入 handoff 授权
    system_prompt: str                  # subagent 内部用，lead 不读
```

### 3.5 关键 Mechanism：自动注入 handoff 授权（Q9）

Lead 派 task 时**不写** `{{handoff://X}}` 占位符。Harness wrapper 查 `SubagentConfig.required_upstream_handoffs` → 自动给 task prompt 附占位符 → `HandoffIsolationProvider` 自动授权。

→ Lead 完全不知道 handoff 存在。

### 3.6 3 个 Guardrail

| Guardrail | 拦截对象 | 触发条件 |
|---|---|---|
| **IntentClassificationGuardrailProvider**（新）| lead 的第一个非 read_file tool call | lead 未输出 `[intent] X` 行 |
| **TaskHandoffAuthorizationProvider**（新）| lead 的 task() 调用 | lead task prompt 缺少 `{{handoff://X}}` 占位符（与 SubagentConfig.required_upstream_handoffs 对比）|
| **HandoffIsolationProvider**（现有，保留）| subagent 的 read_file | subagent 读未经授权的 handoff_*.json |

### 3.7 Catalog 变更（Q5 + Q10）

- **Plan 拆两份**：`PlanMetrics` + `PlanCharts` 两个 dataclass
- **CLI `--mode` 参数**：`--mode metrics`（code-executor 用）vs `--mode charts --user-intent ...`（chart-maker 用）
- **`_common.yaml`**：注册 trajectory_plot + timeseries_plot，`when: total_subjects >= 1`
- **`_evaluate_when` 扩展**：加 `total_subjects` 变量
- **`charts_fallback_available`**：plan_charts.json 自带 fallback 候选列表
- **`prep_metric_plan` tool**：改输出 `plan_metrics.json`（不含 charts）

### 3.8 Skill 变更（Q4 + Q7）

| Skill | 变更 |
|---|---|
| `ethoinsight`（根）| 新增 `references/execution-conventions.md`（通用 bash 约束 + handoff schema + error recovery）|
| `ethoinsight-code` | 删 step 4 charts 循环；引用根 skill execution-conventions |
| `ethoinsight-charts` | 从 lead 用 → chart-maker 专有；lead 不再读 |
| `ethoinsight-chart-maker` | **新建**：用户语义解析 + fallback 选图逻辑 + 引用 charts references/ |
| `ethoinsight-lead-interaction` | **新建**：意图分类决策树 + 必需字段表 + 反问模板 + 意图转移规则 |
| `ethovision-paradigm-knowledge` | 删 default-fallback.md；加强反问模板 |

---

## 4. E2E 完整粒子（核心场景 S1+S2）

### S2：用户上传 EPM 3v3 + "帮我分析一下"

```
Turn 1:
Lead:
  → 意图分类: E2E_MIN
  → 范式识别: 读 skill → 推断 EPM（高置信）→ set_experiment_paradigm
  → 调 prep_metric_plan(file, paradigm="epm") → plan_metrics.json
  → 派 code-executor
  → [code-executor 跑 metrics + stats → handoff + gate_signals]
  → 收 gate_signals → 派 data-analyst
  → [data-analyst 读 handoff → 解读 → handoff_data_analyst.json]
  → ask_clarification: "看结果就够了 / 加图 / 加报告 / 都要"

Turn 2 (用户选"加图"):
Lead:
  → 意图分类: CHART
  → 派 chart-maker
  → [chart-maker 读 handoff + 跑 catalog.resolve --mode charts → plan_charts.json
     → 用 skill 选图 → bash 跑 scripts → handoff_chart_maker.json + png]
  → present_files(*.png)

Turn 3 (ask_clarification 是否加报告):
  → 用户选"是" → 派 report-writer → report.md
```

### S1：5-18 故障复现（单被试 + "画几个图"）

```
前提: code-executor 已跑完，workspace 有 handoff_code_executor.json

Turn N (用户: "再画几个图"):
Lead:
  → 意图分类: CHART
  → 派 chart-maker
  ↓ harness 自动注入 {{handoff://code_executor}}

chart-maker:
  1. read execution-conventions.md
  2. read ethoinsight-charts/SKILL.md + references/spatial-temporal-charts.md
  3. read handoff_code_executor.json → paradigm=epm, 数据列名, n=1
  4. bash catalog.resolve --mode charts --paradigm epm --user-intent "画几个图" ...
     → plan_charts.json: charts=[], fallback=[trajectory_plot, timeseries_plot]
  5. skill 决策: "几个图"=复数 + 单被试 → 选 trajectory_plot + timeseries_plot
  6. bash python -m ethoinsight.scripts._common.plot_trajectory ... → trajectory.png
     bash python -m ethoinsight.scripts._common.plot_timeseries ... → timeseries.png
  7. write handoff_chart_maker.json
  8. OK + gate_signals

→ 5-18 故障消除。关键差异:
  - Lead 不自己指定"画 trajectory" → chart-maker 自主选
  - chart-maker 契约第一步必读 handoff → 知道有什么数据
  - catalog 提供 fallback 候选 → chart-maker 有选项可选
```

---

## 5. 实施计划：双流并行 + 按层依赖

```
Layer 0（无依赖，可并行第一波）
  W1:  SubagentConfig schema 扩展（4 字段）
  W2:  Plan dataclass 拆 PlanMetrics + PlanCharts
  W3:  catalog/_common.yaml 新建
  W7:  execution-conventions.md（根 skill 新 reference）

Layer 1（依赖 L0）
  W4:  catalog CLI --mode + --user-intent + --total-subjects
  W5:  _evaluate_when 扩展 total_subjects
  W6:  scripts/_common/plot_timeseries.py CLI 包装
  W8:  ethoinsight-lead-interaction skill 新建
  W9:  ethoinsight-charts skill 重定位
  W10: ethovision-paradigm-knowledge skill 更新
  W21: ethoinsight-chart-maker skill 新建

Layer 2（依赖 L1）
  W11: code-executor config 改（删 charts 段 + 加 contract 元数据）
  W12: data-analyst config 改（加 contract 元数据）
  W13: chart-maker builtin 新建
  W14: report-writer config 改
  W15: knowledge-assistant config 改
  W20: prep_metric_plan tool 改

Layer 3（依赖 L2）
  W16: Lead prompt 大瘦身（1243 → ~200 行）
  W17: IntentClassificationGuardrailProvider
  W18: TaskHandoffAuthorizationProvider

Layer 4（依赖 L2+L3）
  W19: Harness task() wrapper 自动注入

Layer 5（依赖全部）
  W22: 测试 + dogfood
```

**双流**：
- **流 1**：catalog/schema 线（W1-W6, W20）
- **流 2**：subagent/config/skill 线（W7-W15, W21）
- **汇合**：L3（W16-W18）→ L4（W19）→ L5（W22）

**以 worktree 执行**。

---

## 6. Dogfood 最小通过集（3 场景 smoke）

| # | 场景 | 覆盖决策 |
|---|---|---|
| **S1** | 5-18 复现：单被试 + "画几个图" → chart-maker 读 handoff + fallback | Q5, Q8, Q9, Q10 |
| **S2** | 正常端到端：EPM 3v3 + "分析一下" → E2E_MIN 流水线 | Q3, Q7, Q8 |
| **S8** | 范式模糊：只说"帮我分析" + 文件名不明确 → gate 反问 | Q11 |

---

## 7. 你的任务：写完整 spec

**输出物**：
1. `docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md`

**Spec 内容**：
- 架构总览（本文档 §3 终态架构）
- SubagentConfig 扩展 schema（精确字段定义）
- 5 个 subagent 的完整 contract（每份含 when_to_use / input_contract / output_contract / required_upstream_handoffs / system_prompt 修改要点）
- Lead prompt 重写骨架（意图分类 + 意图状态机 + capability 列表 + 范式识别 + 反问决策表）
- 3 个 GuardrailProvider 的精确拦截逻辑
- Catalog schema 变更（PlanMetrics / PlanCharts dataclass + CLI --mode 参数完整签名）
- Skill 变更清单（每个 skill 的改动内容精确描述）
- 22 工作项的依赖关系图和每项的文件列表
- Dogfood 验证 3 场景的验收标准
- 风险登记

**Spec 不覆盖**（此工作范围外）：
- `report-writer` 和 `knowledge-assistant` 的 prompt 内部逻辑（基本不动）
- `data-analyst` 解读逻辑（不动）
- 前端（streamdown 升级已完成）
- DeerFlow 上游同步

---

## 8. 操作约束（不要做的事）

- ❌ **不写代码**——spec 阶段只写设计文档
- ❌ **不重新走 grill-me 12 轮**——决策已做完
- ❌ **不动 lead prompt 以外的 subagent system_prompt 细节**——只重写 lead prompt + 各 subagent 的 contract metadata；subagent 内部 prompt 微调范围受限
- ❌ **不动 HandoffIsolationProvider**（现有，保留）
- ❌ **不动 ev19_template_provider guardrail**（现有，保留）
- ❌ **碰 spec-phase-1 worktree**——它在跑另一件事

---

## 9. 关键文件路径速查

| 文件 | 用途 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/subagents/config.py` | SubagentConfig — Layer 0 改 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | code-executor — Layer 2 改 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | data-analyst |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py` | **chart-maker 新建** |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` | report-writer |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py` | knowledge-assistant |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | Lead prompt — Layer 3 大改 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | Middleware 链 — 加 guardrail |
| `packages/agent/backend/packages/harness/deerflow/guardrails/` | Guardrail 目录 |
| `packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py` | 现有，保留 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py` | prep_metric_plan — Layer 2 改 |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | task() 执行 — Layer 4 改 |
| `packages/ethoinsight/ethoinsight/catalog/schema.py` | Plan dataclass — Layer 0 改 |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | Resolve — Layer 1 改 |
| `packages/ethoinsight/ethoinsight/catalog/cli.py` | CLI — Layer 1 改 |
| `packages/ethoinsight/ethoinsight/catalog/_common.yaml` | **新建** — Layer 0 |
| `packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py` | **新建** — Layer 1 |
| `packages/agent/skills/custom/ethoinsight/references/execution-conventions.md` | **新建** — Layer 0 |
| `packages/agent/skills/custom/ethoinsight-lead-interaction/` | **新建 skill** — Layer 1 |
| `packages/agent/skills/custom/ethoinsight-chart-maker/` | **新建 skill** — Layer 1 |
| `packages/agent/skills/custom/ethoinsight-charts/SKILL.md` | 重定位 — Layer 1 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/` | 更新 — Layer 1 |

---

## 10. 心智模型（5 条传下去）

1. **Capability-exposure > prompt instructions**：subagent 通过 contract 宣告自己能力，lead 按 contract 调度——不要让 lead prompt 含 subagent 内部知识
2. **Shared state via handoff files, NOT inline context**：subagent 之间通过 per-subagent handoff 文件传 facts；harness 自动注入授权；lead 不参与 facts 中转
3. **Lead 持有交互 know-how，不持有执行 know-how**：范式识别中"用户语言 → 范式映射"归 lead（借助 skill），但"列名 → 指标脚本"归 catalog
4. **Fallback 在 catalog，不在 subagent 判断**：`_common.yaml` + `charts_fallback_available` 注册 fallback；chart-maker 的 skill 不做重复判断
5. **Gate before guess, never default**：分辨不出必反问用户——不在 lead 层做默认猜测（删 default-fallback.md）

---

**最后**：本 handoff 是上下文转移。新 agent 任务 = **读本 handoff + 预读清单 → 写完整 spec → 交用户 review**。不写代码。
