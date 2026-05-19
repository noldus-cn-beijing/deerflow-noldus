# Subagent 角色拆分 + Capability-Exposure 架构设计 Spec

> **状态**: 设计完成，待实施。基于 2026-05-18 grill-me 12 轮决策。
>
> **来源 handoff**: [docs/handoffs/2026-05/2026-05-18-subagent-role-split-spec-handoff.md](../../handoffs/2026-05/2026-05-18-subagent-role-split-spec-handoff.md)
>
> **For implementing agents**: 本 spec 描述**终态架构 + 完整契约**，不含 step-by-step 任务流。每个工作项（W1-W22）含足够的接口签名、文件路径和验收标准供后续 agent 用 superpowers:writing-plans 转换为 step-level implementation plan，再用 superpowers:subagent-driven-development 执行。**禁止在未阅读 §0 操作约束的情况下直接动代码**。

---

## 0. 操作约束（实施前必读）

| # | 约束 | 出处 |
|---|---|---|
| C1 | **single source of truth**：指标定义 / 范式映射 / 展示元数据 / 模板识别表 等只存一份（catalog YAML / Python 常量 / facts 文件），skill 文档只写"怎么读 catalog"，不内嵌结构化知识 | 用户硬性原则；feedback memory `single-source-of-truth` |
| C2 | **不动 HandoffIsolationProvider 和 Ev19TemplateGuardrailProvider** —— 它们职责正交，新 guardrail 与之并列 | handoff §8 |
| C3 | **不动 data-analyst / report-writer / knowledge-assistant 的内部 prompt 主体逻辑** —— 只补 contract metadata 字段、调整 workflow 步骤里读哪些 handoff、补 gate_signals 字段 | handoff §7、§8 |
| C4 | **不碰 worktree `spec-phase-1-handoff`** —— 它在跑另一件事 | handoff §8 |
| C5 | **不重新走 grill-me** —— Q1–Q12 决策已锁，记录在 grill-me memory | handoff §1 |
| C6 | **遵循 catalog 现有 dataclass 命名风格**：`@dataclass(frozen=True)` for catalog 端、`@dataclass` for plan 端（不强加 pydantic 依赖） | `packages/ethoinsight/ethoinsight/catalog/schema.py` 现状 |
| C7 | **同时改 YAML + 写脚本**：加新 chart/metric 时 YAML 注册和 `ethoinsight/scripts/...` 脚本必须在**同一个 commit** 内落地，避免 "YAML 说有但脚本不存在" 的孤儿状态 | feedback memory `single-source-of-truth` |
| C8 | **TDD 强制**：每个工作项都必须先写单元测试（`packages/agent/backend/tests/test_*.py` 或 `packages/ethoinsight/tests/test_*.py`），test 通过再提交 | CLAUDE.md `Testing` 段 |
| C9 | **遵守 deerflow harness boundary**：harness 层 (`packages/harness/deerflow/...`) **不允许** `from app.*` import；`tests/test_harness_boundary.py` 会在 CI 拦截 | `backend/CLAUDE.md` "Harness / App Split" |

---

## 1. 问题陈述（背景压缩，≤30 行）

5-18 dogfood 第二阶段「画几个图」**复现 5-14 同类故障**：

- lead 不读 `handoff_code_executor.json` → 凭印象决策
- code-executor 角色过载（既算指标又画图），在 `plan.charts=[]` 单被试场景下疯狂猜脚本路径，39 次 SandboxAudit、触发 LOOP DETECTED
- lead 兜底自己 `write_file` 写一次性 Python 脚本，**违反"lead 不直接执行"边界**
- catalog 没有 fallback 候选机制；skill 严禁 subagent 写代码但又不给可选脚本清单

**已证伪假设**：「prompt-only fix（加禁令、加文字铁律）能稳定约束 LLM 行为」 —— 4 天复现 2 次。

**架构层结论（用户洞察）**：
- A. lead 退化为 router/planner，不持有执行 know-how
- B. subagent 数 = 用户意图数（按 intent 切，不按工种切）
- C. capability-exposure：subagent 通过 contract 声明自己能力，lead 按 contract 调度
- D. fallback 在 catalog（single source of truth），不在 subagent 判断
- E. gate before guess —— 范式 / 数据不清晰时反问，不默认猜

---

## 2. 终态架构总览

### 2.1 组件图

```
┌──────────────────────────────────────────────────────────────────────┐
│ Lead Agent  (~200-line prompt + ethoinsight-lead-interaction skill)  │
│                                                                      │
│   - 意图分类（6 类）                                                   │
│   - 范式识别（skill + 文件名推断 + gate）                                │
│   - 调度 5 个 subagent（按 capability metadata 派）                     │
│   - 用户交互 gate（ask_clarification）                                 │
│   - 不知道：handoff 文件结构 / 脚本路径 / 图种选择逻辑                       │
└─────────────────────┬────────────────────────────────────────────────┘
                      │  task(subagent_type=..., prompt=..., ...)
                      │  ↓
        ┌─────────────┴──────────────────────────────────────┐
        │ harness task_tool                                  │
        │   - 解析 {{handoff://X}} 占位符（已存在）             │
        │   - **新**：按 required_upstream_handoffs           │
        │     自动注入未声明的占位符                            │
        │   - 进入 SubagentExecutor                          │
        └─────────────┬──────────────────────────────────────┘
                      │
   ┌──────────────────┼────────────────┬───────────────┬───────────────┐
   ↓                  ↓                ↓               ↓               ↓
┌─────────┐    ┌────────────┐    ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│code-    │    │data-analyst│    │chart-maker   │ │report-writer │ │knowledge-assistant│
│executor │    │            │    │   (新建)     │ │              │ │                  │
│         │    │            │    │              │ │              │ │                  │
│metrics+ │    │读上游 hand-│    │读 code 上游  │ │读 code+data  │ │QA_FACT 走 hand-  │
│stats    │→  │off, 解读   │→  │handoff →    │→│handoff →    │ │off / QA_KNOW-   │
│         │    │            │    │自跑 catalog │ │6 段 report   │ │LEDGE 不走        │
│不跑图    │    │            │    │resolve     │ │              │ │                  │
│         │    │            │    │--mode charts │ │              │ │                  │
└─────────┘    └────────────┘    └──────────────┘ └──────────────┘ └──────────────────┘
   │                  │                │               │               │
   ↓                  ↓                ↓               ↓               ↓
handoff_code_     handoff_data_   handoff_chart_   handoff_report_  knowledge_response.md
executor.json     analyst.json    maker.json       writer.json       (可选)
```

### 2.2 数据流（per-subagent handoff 文件）

| 文件 | 写入者 | 授权读者 |
|---|---|---|
| `handoff_code_executor.json` | code-executor | data-analyst, chart-maker, report-writer, knowledge-assistant(QA_FACT) |
| `handoff_data_analyst.json` | data-analyst | report-writer, knowledge-assistant(QA_FACT) |
| `handoff_chart_maker.json` | chart-maker | report-writer (可选), knowledge-assistant(QA_FACT, 可选) |
| `handoff_report_writer.json` | report-writer | knowledge-assistant(QA_FACT, 可选) |
| `plan_metrics.json` | `prep_metric_plan` tool（lead 触发） | code-executor |
| `plan_charts.json` | chart-maker 自跑 `catalog.resolve --mode charts` | chart-maker 自读 |
| `experiment-context.json` | `set_experiment_paradigm` tool | 由 `Ev19TemplateGuardrailProvider` 读取（不变） |

**权限模型**：subagent 看到 `{{handoff://X}}` 占位符 = 授权读 `handoff_X.json`。Lead 派遣 task 时**不写占位符**（C3 哲学）；harness task_tool 按 `SubagentConfig.required_upstream_handoffs` 自动注入占位符（W19）。

### 2.3 文件物理路径全景（sandbox 虚拟路径）

```
/mnt/user-data/workspace/
  ├── experiment-context.json    # set_experiment_paradigm 写
  ├── metric_plan.json           # prep_metric_plan tool 写 — 现状（兼容期保留）
  ├── plan_metrics.json          # prep_metric_plan tool 写 — 新名（W20 改）
  ├── plan_charts.json           # chart-maker 自跑 resolve 写
  ├── handoff_code_executor.json
  ├── handoff_data_analyst.json
  ├── handoff_chart_maker.json
  ├── handoff_report_writer.json
  └── m_*.json / stats.json / plot_*.png / ...
/mnt/user-data/outputs/
  ├── report.md
  └── plot_*.png    # （chart-maker 用 present_files 把图复制到 outputs/）
/mnt/skills/
  ├── ethoinsight/SKILL.md
  ├── ethoinsight/references/execution-conventions.md  # 新建
  ├── ethoinsight-code/SKILL.md
  ├── ethoinsight-charts/SKILL.md       # 重定位 — 不再 lead 用
  ├── ethoinsight-chart-maker/SKILL.md  # 新建
  ├── ethoinsight-lead-interaction/SKILL.md  # 新建
  └── ethovision-paradigm-knowledge/SKILL.md  # 更新
```

---

## 3. SubagentConfig schema 扩展

### 3.1 字段定义

文件：`packages/agent/backend/packages/harness/deerflow/subagents/config.py`

新增 4 个字段，**全部可选**（默认 `None` / 空列表），不破坏现有 `general-purpose` / `bash` subagent。

```python
@dataclass
class SubagentConfig:
    # ---- 现有字段（不变） ----
    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900

    # ---- Capability-exposure 新增（本 spec）----
    when_to_use: str | None = None
        # 给 lead 看的"何时派"指引（1-3 句中文，列出适用场景 + 反例）
        # 取代 lead prompt 里硬编码的"何时派 code-executor"等段落
    input_contract: str | None = None
        # 给 lead 看的"派遣时该传什么"模板（用户语言 + 任何配套元数据）
        # 不包含 {{handoff://X}}（那些由 required_upstream_handoffs 自动注入）
    output_contract: str | None = None
        # 给 lead 看的"subagent 返回什么"简介（最终消息形态 + handoff 文件名 + gate_signals 必备字段）
    required_upstream_handoffs: list[str] = field(default_factory=list)
        # subagent name 列表（与 HANDOFF_FILE_REGISTRY 的 key 对齐：`code_executor` / `data_analyst` / `chart_maker` / `report_writer`）
        # harness task_tool 在派遣前会自动给 prompt 附 {{handoff://X}} 占位符
        # 留空列表 = subagent 不依赖任何 upstream handoff（如 knowledge-assistant QA_KNOWLEDGE 场景）
```

### 3.2 序列化与展示

- `SubagentConfig` 是 dataclass，已经天然 JSON-serializable
- Lead prompt 渲染时（W16）通过 helper 函数 `format_subagent_capability(config: SubagentConfig) -> str` 把 5 字段（name / description / when_to_use / input_contract / output_contract）渲染成 Markdown 表格，注入到 lead 的 system prompt
- **不**在 prompt 暴露 `required_upstream_handoffs` / `system_prompt` —— 这是 harness 内部使用

### 3.3 验收测试（W1）

- `tests/test_subagent_config_capability.py`
  - 现有 `BUILTIN_SUBAGENTS` 中 `code-executor` / `data-analyst` / `report-writer` / `knowledge-assistant` 加上新字段后**仍能正常 instantiate**
  - `when_to_use` / `input_contract` / `output_contract` 为 `None` 时，`format_subagent_capability` 输出 `"(未声明)"` 而非崩溃
  - `required_upstream_handoffs` 中的每个 key **必须**存在于 `HANDOFF_FILE_REGISTRY`（fail-fast）

---

## 4. 6 类用户意图 + 派遣链

### 4.1 意图分类（lead 持有 know-how，写在 ethoinsight-lead-interaction skill）

| Intent | 触发条件 | 派遣链 | 终结条件 |
|---|---|---|---|
| `E2E_FULL` | 上传数据 + 用户语义包含"分析并画图/出报告/全套" | code-executor → data-analyst → chart-maker → ask_clarification(report?) → 视用户选择派 report-writer | report.md 已生成 / 用户选不要 report |
| `E2E_MIN` | 上传数据 + 用户语义为"分析一下"且未明示要图/报告 | code-executor → data-analyst → ask_clarification("看结果就够了 / 加图 / 加报告 / 都要") → 按选择再派 | ask_clarification 已响应 + 后续派遣完成 |
| `CHART` | 已有 `handoff_code_executor.json` + 用户要求图（"再画几个图" / "补充 trajectory") | chart-maker（单派） | chart-maker 返回 + present_files(*.png) |
| `REPORT` | 已有 `handoff_code_executor.json` + `handoff_data_analyst.json` + 用户要求报告 | report-writer（单派） | report.md 已写 |
| `QA_FACT` | 已有 handoff + 用户追问具体数据/统计 | knowledge-assistant（授权 handoff 占位符） | 文本回答返回 |
| `QA_KNOWLEDGE` | 领域知识 / 概念问题（不依赖具体数据） | knowledge-assistant（**不**授权 handoff） | 文本回答返回 |
| `CLARIFY` | 意图模糊 / 缺范式信息 / 缺数据列 | `ask_clarification`（不派 subagent） | 用户响应 |

### 4.2 意图状态机（写入 lead prompt 骨架）

```
[ANY] ──[ user 上传数据 + 复合 ]──→ E2E_FULL ──→ chart-maker 完 ──→ ask(report?) ──→ [ANY]
[ANY] ──[ user 上传数据 + 单语义 ]──→ E2E_MIN ──→ ask(four-choice) ──→ {CHART | REPORT | E2E_FULL[skip-data] | END}
[ANY+handoff] ──[ user 要图 ]──→ CHART ──→ [ANY+handoff]
[ANY+handoff] ──[ user 要报告 ]──→ REPORT ──→ [ANY+handoff]
[ANY+handoff] ──[ user 追问数据 ]──→ QA_FACT ──→ [ANY+handoff]
[ANY] ──[ user 问知识 ]──→ QA_KNOWLEDGE ──→ [ANY]
[ANY] ──[ 信息缺失 ]──→ CLARIFY ──→ [ANY]
```

### 4.3 意图分类的 hard rule

- **第一个非 read_file tool call 之前**，lead 必须在某条 message 中输出 `[intent] <INTENT_NAME>` 行（被 `IntentClassificationGuardrailProvider` 拦截校验，详见 §6.1）
- 意图未明确时 → `CLARIFY` 而非默认猜

---

## 5. 5 个 Subagent 的完整 Contract

### 5.1 code-executor（修改）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

**改动总览**：
- 加 4 个 capability metadata 字段
- 删除 system_prompt 中 `<workflow>` 第 4 步 "for chart in plan.charts" 段（chart-maker 接手）
- 删除 `<critical_rules>` 中提到的 charts 相关警告（plan.charts=[] 不再是其关注点）
- `skills` 删 `ethoinsight-charts`（不再读图相关 skill）
- `disallowed_tools` 不变

**capability metadata**：
```python
when_to_use = """
适合：
- 用户上传 EthoVision 数据并要求"分析" / "算指标" / "做统计"
- 已经派过本 subagent 后又要"重算某个指标" / "改 include/exclude 重跑"
不适合：
- 画图（派 chart-maker）
- 解读统计结果（派 data-analyst）
- 写报告（派 report-writer）
"""
input_contract = """
派遣 prompt 模板：
  "请按 plan_metrics.json 算指标和统计。范式: <paradigm>"
配套：必须在 prompt 前先调 set_experiment_paradigm + prep_metric_plan tool（lead 自己持有这两个 tool）
"""
output_contract = """
- 写 /mnt/user-data/workspace/handoff_code_executor.json（schema 详见 ethoinsight-code skill templates/output-contract.md）
- 最终 AIMessage 形如 `OK: handoff written\\n[gate_signals]\\n...`
- [gate_signals] 块字段：constitution_acknowledged / data_quality{critical_count, warning_count, critical_items[]} / statistical_validity / errors_count
"""
required_upstream_handoffs = []   # 第一棒，无上游
```

**system_prompt workflow 改动**（diff 风格）：
```diff
 <workflow>
 1. 开工前必读输出宪法...
 2. read ${workspace_path}/plan_metrics.json    # 改名：原 metric_plan.json
 3. for entry in plan.metrics: bash ...
 4. if plan.statistics is not null and skip_reason is null: bash stats
-5. for chart in plan.charts: bash ...
-   注意：plan.charts 是空数组时直接跳过
+   （注：本 subagent 不跑图。plan.charts 字段已从 plan_metrics.json 移除。）
 5. 聚合 → handoff_code_executor.json
 6. write handoff
 7. 输出 [gate_signals]
 </workflow>
```

`skills`：`["ethoinsight-code"]`（删 `ethoinsight-charts`）

### 5.2 data-analyst（微改）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

**改动总览**：
- 加 4 个 capability metadata
- system_prompt 不动主体（contract / workflow / json_writing / principles 等保留）

**capability metadata**：
```python
when_to_use = """
适合：
- code-executor 刚完成、有 handoff_code_executor.json，要对统计结果做专业解读 / 方法学把关 / 离群诊断
不适合：
- 用户问纯领域知识（派 knowledge-assistant）
- 画图（派 chart-maker）
"""
input_contract = """
派遣 prompt 模板（用户语言原话 + 简短引导）：
  "请基于 code-executor 的结果做专业解读，关注效应量和混杂因素。"
"""
output_contract = """
- 写 /mnt/user-data/workspace/handoff_data_analyst.json（schema 详见 system_prompt）
- 最终 AIMessage：2-3 段自然语言摘要 + [gate_signals] 块
- [gate_signals] 字段：constitution_acknowledged / method_warnings_count / outlier_count / excluded_metrics_count / statistical_validity / errors_count
"""
required_upstream_handoffs = ["code_executor"]
```

### 5.3 chart-maker（新建）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py`

**核心职责**：用户语义解析 → 自跑 catalog.resolve --mode charts → 决策图种 → bash 跑图脚本 → 写 handoff。

**capability metadata**：
```python
when_to_use = """
适合：
- 用户要求"画图" / "可视化" / "trajectory" / "再补几个图"
- E2E 流程中 data-analyst 完成后的图表生成阶段
不适合：
- 第一次分析（先派 code-executor 跑指标，本 subagent 依赖 handoff_code_executor.json）
- 解读图意义（派 data-analyst 或 knowledge-assistant）
"""
input_contract = """
派遣 prompt 模板：
  "请基于 code-executor 的结果生成可视化图表。用户原话: <用户语义>"
不需要 lead 指定图种 —— 本 subagent 自己解析用户语义 + 跑 catalog 决定。
"""
output_contract = """
- 写 /mnt/user-data/workspace/handoff_chart_maker.json
- 写 /mnt/user-data/workspace/plot_*.png（图文件）
- 用 present_files 把图复制到 /mnt/user-data/outputs/ 让用户可见
- 最终 AIMessage: `OK: <N> charts generated` + [gate_signals]
- [gate_signals] 字段：constitution_acknowledged / charts_generated_count / charts_skipped_count / fallback_used (bool) / errors_count
"""
required_upstream_handoffs = ["code_executor"]
```

**handoff_chart_maker.json schema**:

```json
{
  "status": "completed" | "failed",
  "charts_generated": [
    {"id": "trajectory_plot", "script": "ethoinsight.scripts._common.plot_trajectory", "output": "/mnt/user-data/workspace/plot_trajectory.png", "reason": "user-requested" | "fallback"}
  ],
  "charts_skipped": [
    {"id": "box_open_arm", "reason": "when_not_satisfied: n_per_group >= 3 (got 1)"}
  ],
  "fallback_used": true,
  "user_intent_parsed": "trajectory + timeseries (单被试 fallback)",
  "errors": [str, ...]
}
```

**system_prompt 框架**（详见 W13 实施时落地，骨架如下）：

```
你是行为数据可视化执行专家。

<environment> 与 code-executor 同 — ethoinsight 预装、bash 调脚本 </environment>

<workflow>
1. 必读 /mnt/skills/ethoinsight/references/execution-conventions.md
2. 必读 /mnt/skills/ethoinsight-chart-maker/SKILL.md
3. read_file /mnt/user-data/workspace/handoff_code_executor.json
   → 拿 paradigm / 数据列名 / n_per_group / n_groups / total_subjects
4. bash python -m ethoinsight.catalog.resolve
     --mode charts
     --paradigm <paradigm>
     --user-intent "<用户语义原话>"
     --total-subjects <N>
     --n-per-group <N>
     --n-groups <N>
     --columns-file <path>
     --raw-files-json <path>
     --workspace-dir <path>
     --output /mnt/user-data/workspace/plan_charts.json
5. read plan_charts.json → 得到 charts[] + charts_fallback_available[]
6. 按 ethoinsight-chart-maker skill 决策树:
     a. 用户语义清晰 + charts[] 非空 → 跑 charts[]
     b. 用户语义清晰 + charts[]=[] + fallback 非空 → 跑 fallback 中匹配用户语义的子集
     c. 用户语义模糊 → 默认跑 charts[] 全部，无 charts 则跑 fallback 全部
     d. 都没有 → 写 status=failed + 让 lead 反问用户
7. for chart in selected: bash python -m <chart.script> --input ... --output ...
8. write handoff_chart_maker.json
9. present_files(<生成的 png 列表>)
10. 输出 OK + [gate_signals]
</workflow>

<bash_constraints>
- 脚本调用：python -m ethoinsight.scripts.<paradigm | _common>.<name> --input ... --output ...
- 仅 catalog CLI 例外：python -m ethoinsight.catalog.resolve --mode charts ...
- 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail
其他形式被 ScriptInvocationOnlyProvider 拦截。
</bash_constraints>
```

**tools / max_turns / timeout / skills**：
```python
tools = ["bash", "read_file", "write_file", "ls", "str_replace", "present_files"]
disallowed_tools = ["task", "ask_clarification", "web_search", "web_fetch", "image_search"]
model = "inherit"
max_turns = 15
timeout_seconds = 600
skills = ["ethoinsight", "ethoinsight-chart-maker"]
```

**注册**：在 `subagents/builtins/__init__.py` 加 `from .chart_maker import CHART_MAKER_CONFIG`、加入 `BUILTIN_SUBAGENTS`、加入 `__all__`。

### 5.4 report-writer（微改）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`

**改动**：加 4 capability metadata；workflow 增加 "可选 read_file handoff_chart_maker.json 取 chart_paths"；其他不动。

**capability metadata**：
```python
when_to_use = """
适合：
- 已有 code-executor + data-analyst handoff，用户要"出报告" / "写 Discussion" / "我要 markdown 报告给导师看"
不适合：
- 没有 data-analyst 解读（先派 data-analyst）
- 只要图（派 chart-maker）
"""
input_contract = """
派遣 prompt 模板：
  "请基于 code-executor 数据 + data-analyst 解读撰写 6 段骨架报告。"
"""
output_contract = """
- 写 /mnt/user-data/outputs/report.md（6 段骨架，详见 system_prompt <structure>）
- 写 /mnt/user-data/workspace/handoff_report_writer.json
- 最终 AIMessage: 报告路径 + 章节摘要 + [gate_signals]
- [gate_signals] 字段：constitution_acknowledged / sections_written_count / sections_missing[] / statistical_validity / errors_count
"""
required_upstream_handoffs = ["code_executor", "data_analyst"]
# 注：chart_maker handoff 是"可选"依赖 —— 不在 required_upstream_handoffs，
# 而是 lead 在派遣时如果走过 chart-maker，额外手动添加 {{handoff://chart_maker}} 占位符
# （这是 capability-exposure 的少数例外。W19 实施时考虑是否扩展 required_upstream_handoffs 支持 optional 标记）
```

### 5.5 knowledge-assistant（微改）

文件：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

**改动**：加 4 capability metadata；system_prompt 不动主体。

**capability metadata**：
```python
when_to_use = """
适合：
- 用户问范式 / 术语 / 方法论概念问题（QA_KNOWLEDGE）
- 已有分析结果，用户追问"为什么 p 不显著" / "NND 偏高说明什么"（QA_FACT）
不适合：
- 用户要重新算指标 / 出新报告（派对应 subagent）
"""
input_contract = """
派遣 prompt 模板：
  QA_KNOWLEDGE: "用户问题: <原话>"
  QA_FACT: "用户问题: <原话>。相关数据见 upstream handoff 文件。"
"""
output_contract = """
- 简单问题：直接在最终 AIMessage 回答
- 深度问题：write_file /mnt/user-data/workspace/knowledge_response.md + 摘要
- 不强制 [gate_signals] 块（QA 不进入 gate 决策路径）
"""
required_upstream_handoffs = []
# 注：QA_FACT 场景下，lead 派遣时手动加 {{handoff://code_executor}} 等占位符
# 不放进 required_upstream_handoffs（区分 QA_KNOWLEDGE 不需要 handoff 的场景）
```

### 5.6 Subagent 字段填充对照表

| Subagent | when_to_use | input_contract | output_contract | required_upstream_handoffs |
|---|---|---|---|---|
| code-executor | ✓ 5.1 | ✓ 5.1 | ✓ 5.1 | `[]` |
| data-analyst | ✓ 5.2 | ✓ 5.2 | ✓ 5.2 | `["code_executor"]` |
| chart-maker | ✓ 5.3 | ✓ 5.3 | ✓ 5.3 | `["code_executor"]` |
| report-writer | ✓ 5.4 | ✓ 5.4 | ✓ 5.4 | `["code_executor", "data_analyst"]` |
| knowledge-assistant | ✓ 5.5 | ✓ 5.5 | ✓ 5.5 | `[]`（QA_FACT 由 lead 手动占位符） |

---

## 6. 3 个 GuardrailProvider（精确拦截逻辑）

### 6.1 IntentClassificationGuardrailProvider（新建）

文件：`packages/agent/backend/packages/harness/deerflow/guardrails/intent_classification_provider.py`

**职责**：lead 在派遣任何 subagent **之前**必须在 message 流中输出 `[intent] <INTENT_NAME>` 行；否则拦截并注入 reminder。

**拦截对象**：lead 调用的所有非 `read_file` tool（task / set_experiment_paradigm / prep_metric_plan / ask_clarification 等）。

**算法**：
- 通过 `ContextVar` + Bridge middleware（类比 `Ev19WorkspaceBridgeMiddleware`），把 thread 的 messages 历史暴露给 provider
- provider 扫描最近的一条 lead message 文本，匹配 `\[intent\]\s+(E2E_FULL|E2E_MIN|CHART|REPORT|QA_FACT|QA_KNOWLEDGE|CLARIFY)`
- 若匹配：allow
- 若未匹配且 tool 不是 `read_file`：deny，理由 `ethoinsight.intent_not_declared`：
  > "请先在 message 中输出 `[intent] <INTENT>` 行（INTENT ∈ {E2E_FULL, E2E_MIN, CHART, REPORT, QA_FACT, QA_KNOWLEDGE, CLARIFY}）"
- `read_file` 永远 allow（lead 在分类前需要 read skill 文档）

**注入位置**：lead_agent 的 middleware 链中，在 `GuardrailMiddleware` 之前加 `IntentBridgeMiddleware`（暴露 messages 到 ContextVar），然后用 `GuardrailMiddleware(provider=IntentClassificationGuardrailProvider(...))`。

**fail-open 边界**：thread 刚开始无 messages 时 allow（避免锁死）。

### 6.2 TaskHandoffAuthorizationProvider（新建）

文件：`packages/agent/backend/packages/harness/deerflow/guardrails/task_handoff_authorization_provider.py`

**职责**：拦截 lead 的 `task(subagent_type=X, prompt=Y)` 调用，校验 `prompt` 是否包含了 `BUILTIN_SUBAGENTS[X].required_upstream_handoffs` 中每个 name 对应的 `{{handoff://X}}` 占位符。

**算法**：
```python
def evaluate(request: GuardrailRequest) -> GuardrailDecision:
    if request.tool_name != "task":
        return GuardrailDecision(allow=True)
    subagent_type = request.tool_input.get("subagent_type", "")
    prompt = request.tool_input.get("prompt", "")
    config = BUILTIN_SUBAGENTS.get(subagent_type)
    if not config or not config.required_upstream_handoffs:
        return GuardrailDecision(allow=True)
    missing = [
        name for name in config.required_upstream_handoffs
        if f"{{{{handoff://{name}}}}}" not in prompt
    ]
    if missing:
        return GuardrailDecision(
            allow=False,
            reasons=[GuardrailReason(
                code="ethoinsight.required_handoff_missing",
                message=(
                    f"subagent '{subagent_type}' 需要 upstream handoff "
                    f"{missing}。在 prompt 中加 {{{{handoff://<name>}}}} 占位符。"
                ),
            )],
            policy_id="task_handoff_authorization",
        )
    return GuardrailDecision(allow=True)
```

**与 W19 自动注入的关系**：
- 终态：W19 在 task_tool 中自动注入占位符 → 本 provider 基本不会真实触发拦截
- 但保留 provider 作为**安全网**：如果 W19 有 bug 或 lead 自己手写 task() 绕过，本 provider 抓住
- 在 W19 完成前，本 provider 是**主要约束**（lead 必须手写占位符）

### 6.3 HandoffIsolationProvider（现有，**不动**）

文件：`packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`

保留现状。需要在 `HANDOFF_FILE_REGISTRY`（在 `task_tool.py`）加新 entry：

```python
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "chart_maker": "handoff_chart_maker.json",     # 新增
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",
}
```

### 6.4 Ev19TemplateGuardrailProvider（现有，**不动**）

保留现状。

### 6.5 ScriptInvocationOnlyProvider（现有，**不动**）

保留现状。chart-maker 的 bash 调用同样受此 provider 约束（只允许 `python -m ethoinsight.*` + 白名单文件操作）。

需要确认：`python -m ethoinsight.catalog.resolve --mode charts` 在 provider 的白名单内（应该是，因为它仍然是 `python -m ethoinsight.*` 形式）。W22 dogfood 时验证。

### 6.6 lead middleware 链插入位置

**指示性顺序**（W16/W17/W18 实施时以 `lead_agent/agent.py` 当前实际链为准，下面只标注新中间件的插入位置）：

```
ThreadDataMiddleware
UploadsMiddleware
SandboxMiddleware
DanglingToolCallMiddleware
Ev19WorkspaceBridgeMiddleware        # 现有
IntentBridgeMiddleware               # 新（W17 配套）
GuardrailMiddleware(Ev19TemplateGuardrailProvider)    # 现有
GuardrailMiddleware(IntentClassificationGuardrailProvider)   # 新（W17）
GuardrailMiddleware(TaskHandoffAuthorizationProvider)        # 新（W18）
# code-executor / chart-maker 子链上叠加 GuardrailMiddleware(ScriptInvocationOnlyProvider) 和 GuardrailMiddleware(HandoffIsolationProvider) —— 现有
GateEnforcementMiddleware                                    # 现有
... 其他中间件 ...
ClarificationMiddleware (last)
```

---

## 7. Catalog 变更

### 7.1 Plan dataclass 拆分（W2）

文件：`packages/ethoinsight/ethoinsight/catalog/schema.py`

**当前**：单一 `Plan` dataclass 同时含 `metrics`/`statistics`/`charts`/`skipped`/`notes`。

**目标**：拆出 `PlanMetrics`（含 metrics + statistics + skipped + notes）和 `PlanCharts`（含 charts + charts_fallback_available + notes）两份独立 dataclass + 独立 schema_version。

```python
@dataclass
class PlanMetrics:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    metrics: list[PlanMetric]
    statistics: PlanStatistics | None
    skipped: list[PlanSkipped]
    notes: list[str]
    schema_version: str = "1.0"   # 默认值字段必须放最后


@dataclass
class PlanCharts:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    charts: list[PlanChart]                       # 命中 catalog when 条件的图
    charts_fallback_available: list[PlanChart]    # 单被试等场景的 fallback 候选
    skipped: list[PlanSkipped]                    # 图层面 skip（when 不满足）
    user_intent: str | None                       # 来自 --user-intent
    notes: list[str]
    schema_version: str = "1.0"
```

**Plan（旧）保留为 backward-compatible 别名**或在过渡期临时存活直到 W11/W13 完成 —— 由 W2 实施 agent 决定（推荐：直接删旧 `Plan` dataclass，让 W11+W13 同步改）。

### 7.2 _common.yaml 新建（W3）

文件：`packages/ethoinsight/ethoinsight/catalog/_common.yaml`

```yaml
# 通用 charts — 范式无关，作为单被试 / 组间不可对比场景的 fallback
# 加新 chart 时同步在 packages/ethoinsight/ethoinsight/scripts/_common/ 下添加 CLI 脚本（C7）
common_charts:
  - id: trajectory_plot
    script: ethoinsight.scripts._common.plot_trajectory
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的轨迹可视化

  - id: timeseries_plot
    script: ethoinsight.scripts._common.plot_timeseries
    when: total_subjects >= 1
    rationale: 单被试或组间数据不全时的时间动态
```

### 7.3 loader.py 加载 _common.yaml（W3 配套）

文件：`packages/ethoinsight/ethoinsight/catalog/loader.py`

加 `load_common_catalog() -> CommonCatalog` 函数，返回 `dataclass CommonCatalog(common_charts: list[ChartEntry])`。

### 7.4 _evaluate_when 扩展（W5）

文件：`packages/ethoinsight/ethoinsight/catalog/resolve.py`

`_evaluate_atomic_when` 加 `total_subjects` 变量分支：

```python
def _evaluate_atomic_when(part, *, n_per_group, n_groups, total_subjects):
    ...
    if var == "total_subjects":
        return total_subjects is not None and total_subjects >= val
    ...
```

`_evaluate_when` 签名同步加 `total_subjects` kwarg；`resolve()` 在 charts 段把 `total_subjects` 算出来（`len(set(per-row subject))` 或从 columns_file 取）并传下去。

### 7.5 catalog CLI --mode 参数（W4）

文件：`packages/ethoinsight/ethoinsight/catalog/cli.py`

加 `--mode {metrics, charts}`（默认 `metrics` 以保持兼容）+ `--user-intent <str>` + `--total-subjects <int>`：

```
python -m ethoinsight.catalog.resolve \
  --mode metrics \
  --paradigm epm \
  --columns-file ... --raw-files-json ... --workspace-dir ... \
  --output plan_metrics.json
```

```
python -m ethoinsight.catalog.resolve \
  --mode charts \
  --paradigm epm \
  --user-intent "再画几个图" \
  --total-subjects 1 --n-per-group 1 --n-groups 1 \
  --columns-file ... --raw-files-json ... --workspace-dir ... \
  --output plan_charts.json
```

CLI 在 `--mode metrics` 下调用新函数 `resolve_metrics(...)`（原 resolve 拆出来）；`--mode charts` 调用新函数 `resolve_charts(...)`（新增）。

### 7.6 resolve_charts 函数（W4 配套）

文件：`packages/ethoinsight/ethoinsight/catalog/resolve.py`

```python
def resolve_charts(
    paradigm: str,
    columns: list[str],
    raw_files: list[str],
    workspace_dir: str,
    *,
    user_intent: str | None = None,
    total_subjects: int | None = None,
    n_per_group: int | None = None,
    n_groups: int | None = None,
    groups_file: str | None = None,
    columns_file: str | None = None,
    ev19_template: str | None = None,
    virtual_workspace_dir: str | None = None,
) -> PlanCharts:
    """生成 PlanCharts。Step:
       1. load_catalog(paradigm) → 拿范式 charts
       2. for ch in cat.charts: if _evaluate_when(ch.when, ...) → charts.append
       3. if not charts:  # 触发 fallback 路径
          load_common_catalog() → for ch in common_charts:
             if _evaluate_when(ch.when, ..., total_subjects=...): fallback.append
       4. 组装 PlanCharts dataclass 返回
    """
```

**Fallback 触发条件**：`len(charts) == 0` AND `len(cat.charts) > 0`（范式有图但被 when 全过滤掉）。

**注意**：单被试场景常常 `len(cat.charts) == 0`（catalog 没注册组间图），此时也应该触发 fallback —— 实施时 W4 重新评估触发条件，建议改为 `len(charts) == 0`（无论 cat.charts 是否原本有内容）。

### 7.7 prep_metric_plan tool 改名输出（W20）

文件：`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`

改动：
- 输出文件名从 `metric_plan.json` 改为 `plan_metrics.json`
- 内部调用 `resolve_metrics(...)` 而非旧 `resolve(...)`
- 返回 dict 的 `plan_path` 字段同步改 `/mnt/user-data/workspace/plan_metrics.json`
- 兼容期处理：**不**保留 `metric_plan.json` 别名（前端/agent 都跟着改即可，避免双源）

### 7.8 新建 plot_timeseries.py（W6）

文件：`packages/ethoinsight/ethoinsight/scripts/_common/plot_timeseries.py`

CLI 包装：参考现有 `plot_trajectory.py` 签名，调用 `ethoinsight.charts.timeseries_plot` 库函数。

可选 `--y-col`；未传时按 paradigm 查 `_DEFAULT_Y_COL_BY_PARADIGM` 默认值映射（见 5-18 handoff §3.1 决策 6）。

---

## 8. Skill 变更清单

### 8.1 ethoinsight（根 skill）—— 新加 references/execution-conventions.md（W7）

文件：`packages/agent/skills/custom/ethoinsight/references/execution-conventions.md`

内容（通用执行约束，所有 code-executor / chart-maker 等"执行类" subagent 都 read 此文件）：
- bash 调用约束：`python -m ethoinsight.scripts.<paradigm | _common>.<name> --input X --output Y`
- handoff JSON 写入规则（schema 详见各 subagent 自己的 skill / contract）
- error recovery：脚本失败时把 errors 写进 handoff，不要重试 >2 次
- gate_signals 块的通用格式说明

更新 `ethoinsight/SKILL.md` 在合适位置（如"操作约束"段）加一句"执行类 subagent 必读 references/execution-conventions.md"。

### 8.2 ethoinsight-code（W11 配套）

文件：`packages/agent/skills/custom/ethoinsight-code/SKILL.md` 和 `references/*.md`

改动：
- 删除 "step 4: for chart in plan.charts" 段
- 加 "本 skill 服务于 code-executor，**只跑 metrics + stats**。图表归 chart-maker（见 ethoinsight-chart-maker skill）" 一句
- 引用根 skill execution-conventions.md，避免重复（C1: single source of truth）

### 8.3 ethoinsight-charts（W9）

文件：`packages/agent/skills/custom/ethoinsight-charts/SKILL.md`

**重定位**：从"lead 用的图选择指南"改为"chart-maker 专有"。lead 不再 read 本 skill。

改动：
- SKILL.md 顶部加 "服务对象：chart-maker subagent。lead 不读本 skill"
- 内容上保留现有的"图种 → 适用场景"对照表，作为 chart-maker 决策时的查询源

### 8.4 ethoinsight-chart-maker（W21 新建）

文件：`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md` + references

内容：
- **用户语义解析逻辑**：「画几个图」→ 复数 + 模糊；「画 trajectory」→ 单选明确；「箱线图比较」→ 组间比较 + 要 box_*
- **fallback 选图决策树**：单被试 + 模糊语义 → trajectory + timeseries；单被试 + 明确语义 → 匹配 fallback 中对应项
- references/spatial-temporal-charts.md 等可从 ethoinsight-charts 继承 / 引用（C1：不重复内容，只链接）

### 8.5 ethoinsight-lead-interaction（W8 新建）

文件：`packages/agent/skills/custom/ethoinsight-lead-interaction/SKILL.md` + references

内容：
- **意图分类决策树**（§4.1 + 4.2 的可执行版本）
- **范式识别决策树**：文件名启发 + skill 知识 + 多候选时反问模板
- **必需字段表**：每种意图派遣前 lead 需要从用户拿到的硬性信息
- **反问模板（中文）**：4-choice ask_clarification 模板、范式不确定反问模板、缺数据列反问模板等
- **意图转移规则**：什么样的用户消息会让 lead 重新分类意图

**Lead system prompt 大瘦身（W16）后，绝大多数"如何反问 / 如何分类"细节移到此 skill**。Lead prompt 只保留：意图分类 hard rule、5 subagent capability 注入位置、范式识别开关、3 个 guardrail 的对应错误码 hint。

### 8.6 ethovision-paradigm-knowledge（W10）

文件：`packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md` + references

改动：
- **删除 references/default-fallback.md**（5-18 handoff §3.5 "Gate before guess"，不再有默认猜测路径）
- 加强反问模板：多候选时让 lead 直接拼装"用户上传的数据看起来是 EPM 或 EZM，请确认"这种带证据的反问
- 保留现有 62 变体白名单 / by-template / by-experiment 文档（C1：知识不重复）

### 8.7 ethoinsight-metric-catalog（不动，但需复查）

现状：data-analyst / report-writer 通过这个 skill 读 catalog YAML 元数据。本 spec 不改它，但 W11 / W12 实施时需要确认 catalog YAML 物理路径变更（如果有）同步到此 skill。

---

## 9. 22 个工作项 + 依赖图

### 9.1 工作项详表

| WI | 名称 | 改动文件 | 测试文件 | 依赖 | 验收 |
|---|---|---|---|---|---|
| W1 | SubagentConfig 4 字段扩展 | `subagents/config.py` | `tests/test_subagent_config_capability.py` | — | §3.3 |
| W2 | Plan dataclass 拆 PlanMetrics + PlanCharts | `ethoinsight/catalog/schema.py` | `ethoinsight/tests/test_catalog_schema.py` | — | dataclass 字段齐全 + serialize 可逆 |
| W3 | _common.yaml + loader 加载 | `ethoinsight/catalog/_common.yaml`（新）+ `loader.py` | `ethoinsight/tests/test_common_catalog.py` | — | load_common_catalog 返回 2 个 chart |
| W7 | ethoinsight 根 skill 加 execution-conventions.md | `skills/custom/ethoinsight/references/execution-conventions.md`（新）+ `SKILL.md` | — | — | 文件存在 + SKILL.md 链接到它 |
| W4 | catalog CLI --mode + --user-intent + --total-subjects + resolve_charts 函数 | `ethoinsight/catalog/cli.py` + `resolve.py` | `ethoinsight/tests/test_catalog_cli_modes.py` + `test_resolve_charts.py` | W2, W3, W5 | S1 dogfood 中能生成 plan_charts.json |
| W5 | _evaluate_when 扩展 total_subjects | `ethoinsight/catalog/resolve.py` | `ethoinsight/tests/test_evaluate_when.py` | W2 | total_subjects >= 1 表达式被正确求值 |
| W6 | plot_timeseries.py CLI 包装 | `ethoinsight/scripts/_common/plot_timeseries.py`（新） | `ethoinsight/tests/test_plot_timeseries_cli.py` | — | bash 调用产 png |
| W8 | ethoinsight-lead-interaction skill 新建 | `skills/custom/ethoinsight-lead-interaction/SKILL.md` + references | — | W7 | SKILL.md 含 §8.5 列出的内容 |
| W9 | ethoinsight-charts 重定位 | `skills/custom/ethoinsight-charts/SKILL.md` | — | W7 | 顶部声明"服务 chart-maker" |
| W10 | ethovision-paradigm-knowledge 更新 | `skills/custom/ethovision-paradigm-knowledge/*` | — | W7 | default-fallback.md 删除 |
| W21 | ethoinsight-chart-maker skill 新建 | `skills/custom/ethoinsight-chart-maker/SKILL.md` + references | — | W7, W9 | 含 §8.4 列出的内容 |
| W11 | code-executor config 改 + skill 改 | `subagents/builtins/code_executor.py` + `skills/custom/ethoinsight-code/SKILL.md` | `tests/test_code_executor_config.py` | W1, W2, W8, W7 | metadata 4 字段 + workflow 删 chart 段 + read plan_metrics.json |
| W12 | data-analyst config 改 | `subagents/builtins/data_analyst.py` | `tests/test_data_analyst_config.py` | W1 | metadata 4 字段；prompt 主体不变 |
| W13 | chart-maker builtin 新建 + 注册 | `subagents/builtins/chart_maker.py`（新） + `subagents/builtins/__init__.py` | `tests/test_chart_maker_config.py` | W1, W2, W3, W4, W21, W7 | BUILTIN_SUBAGENTS 含 "chart-maker"；config 可 instantiate |
| W14 | report-writer config 改 | `subagents/builtins/report_writer.py` | `tests/test_report_writer_config.py` | W1 | metadata 4 字段；required_upstream_handoffs=[code_executor, data_analyst] |
| W15 | knowledge-assistant config 改 | `subagents/builtins/knowledge_assistant.py` | `tests/test_knowledge_assistant_config.py` | W1 | metadata 4 字段；required_upstream_handoffs=[] |
| W20 | prep_metric_plan tool 改输出 plan_metrics.json | `tools/builtins/prep_metric_plan_tool.py` | `tests/test_prep_metric_plan.py`（已存在，扩展） | W2 | 输出 plan_metrics.json；调 resolve_metrics |
| W16 | Lead prompt 大瘦身 + capability 注入 | `agents/lead_agent/prompt.py` | `tests/test_lead_prompt_capability_render.py` | W1, W11-W15, W8 | 行数 1243 → ~200；含 capability 表 + intent state machine + handoff 协议 |
| W17 | IntentClassificationGuardrailProvider + Bridge middleware | `guardrails/intent_classification_provider.py`（新）+ `guardrails/middleware.py` | `tests/test_intent_classification_provider.py` | W1 | 缺 [intent] 行时 deny；read_file 永远 allow |
| W18 | TaskHandoffAuthorizationProvider | `guardrails/task_handoff_authorization_provider.py`（新） | `tests/test_task_handoff_authorization.py` | W1, W11-W15 | 派遣 data-analyst 缺 {{handoff://code_executor}} 时 deny |
| W19 | task_tool 自动注入 required_upstream_handoffs 占位符 | `tools/builtins/task_tool.py` + `subagents/executor.py` | `tests/test_task_tool_auto_inject.py` | W1, W11-W15, W18 | lead 派遣不写占位符时 task_tool 自动加；HandoffIsolationProvider 仍能识别 |
| W22 | Dogfood S1 + S2 + S8 验证 | — | E2E manual + automated smoke | 所有 | §10 验收 |

### 9.2 依赖图（DAG）

```
Layer 0 (无依赖)
  W1   W2   W3   W7
            │    │
Layer 1     │    └────┬──── W8 ─────┐
            │         ├──── W9 ─────┤
            │         ├──── W10 ────┤
  W5 ──────┤         └──── W21 ────┤
  W6                                │
            │                       │
  W4 ───── W2,W3,W5                 │
                                    │
Layer 2 (依赖 L0+L1)                 │
  W11 ──── W1,W2,W8,W7                │
  W12 ──── W1                         │
  W13 ──── W1,W2,W3,W4,W21,W7         │
  W14 ──── W1                         │
  W15 ──── W1                         │
  W20 ──── W2                         │
                                      │
Layer 3 (依赖 L2)                      │
  W16 ──── W1, W11-W15, W8            │
  W17 ──── W1                         │
  W18 ──── W1, W11-W15                │
                                      │
Layer 4 (依赖 L3 + L2)                 │
  W19 ──── W1, W11-W15, W18           │
                                      │
Layer 5 (依赖全部)                     │
  W22 ──── 全部                       │
```

### 9.3 双流并行执行建议

- **流 1（catalog/schema 线）**：W2 → W5 → W4 → W6 → W20（Python 数据层，不依赖 deerflow harness）
- **流 2（subagent/skill 线）**：W1 → W7 → (W8 + W9 + W10 + W21 并行) → (W11 + W12 + W14 + W15 并行) → 等流 1 的 W4 完成后 → W13
- **汇合**：W16 + W17 + W18（依赖两流 L2 完成） → W19 → W22

可使用 superpowers:dispatching-parallel-agents 同时跑两流。

---

## 10. Dogfood 3 场景验收

### 10.1 S1：5-18 复现（单被试 + "再画几个图"）

**前提**：已有 thread 跑过一次 EPM 单被试 code-executor，workspace 含 `handoff_code_executor.json`（paradigm=epm, n=1）。

**用户输入**：「再画几个图」

**期望路径**：
1. lead 输出 `[intent] CHART`（被 W17 通过）
2. lead `task(chart-maker, prompt="请基于 code-executor 结果生成可视化图表。用户原话: 再画几个图")` —— W19 自动注入 `{{handoff://code_executor}}`
3. W18 通过（占位符存在）
4. chart-maker 读 `/mnt/skills/ethoinsight/references/execution-conventions.md` + `/mnt/skills/ethoinsight-chart-maker/SKILL.md`
5. read `handoff_code_executor.json`
6. bash `python -m ethoinsight.catalog.resolve --mode charts --paradigm epm --user-intent "再画几个图" --total-subjects 1 --n-per-group 1 --n-groups 1 ... --output plan_charts.json`
7. read `plan_charts.json` → `charts=[]`, `charts_fallback_available=[trajectory_plot, timeseries_plot]`
8. skill 决策："几个图" = 复数 + 模糊 + 单被试 → 选 fallback 全部
9. bash 跑 plot_trajectory + plot_timeseries → 2 个 png
10. write `handoff_chart_maker.json`（fallback_used=true, charts_generated=2）
11. present_files(2 个 png) + `OK: 2 charts generated\n[gate_signals]\n...`

**验收**：
- ✅ 0 次 LOOP DETECTED
- ✅ chart-maker 总 bash 调用 ≤ 6 次（resolve + 2 个 plot + 最多 2-3 次 ls/cat）
- ✅ lead **没有**自己 write_file 写 Python 脚本
- ✅ 用户在前端看到 trajectory.png + timeseries.png

### 10.2 S2：正常端到端（EPM 3v3 + "帮我分析一下"）

**用户输入**：上传 6 个 EPM 文件（3 control + 3 treatment）+ "帮我分析一下"

**期望路径**：
1. lead 输出 `[intent] E2E_MIN`
2. lead 范式识别（skill + 文件名 + 高置信）→ `set_experiment_paradigm(paradigm=epm, ev19_template=...)`
3. lead `prep_metric_plan(uploaded_file=..., paradigm=epm)` → plan_metrics.json
4. lead `task(code-executor, ...)` （W19 自动注入 —— 此 subagent 无上游 handoff，prompt 不会被加占位符）
5. code-executor 跑 metrics + stats → `handoff_code_executor.json` + `[gate_signals]`
6. lead 收 gate_signals → `task(data-analyst, ...)` （W19 注入 `{{handoff://code_executor}}`）
7. data-analyst → `handoff_data_analyst.json` + `[gate_signals]`
8. lead `ask_clarification("看结果就够了 / 加图 / 加报告 / 都要")` — 4 选项
9. 视用户选择继续派 chart-maker / report-writer / 两者

**验收**：
- ✅ 6 个被试的 EPM 默认指标 + 组间 t-test 跑完无错
- ✅ 4-choice ask_clarification 出现在 turn 4 左右
- ✅ 用户选"都要"后，chart-maker 跑出 EPM catalog 注册的 charts（不是 fallback，因 n_per_group >= 3 满足）

### 10.3 S8：范式模糊（"帮我分析一下"+ 文件名不明确）

**用户输入**：上传文件名为 `mouse_data_2026.txt` + "帮我分析一下"（无范式线索）

**期望路径**：
1. lead 输出 `[intent] CLARIFY`（或先 E2E_MIN 然后转 CLARIFY，取决于 prompt 实现）
2. lead 读 `ethovision-paradigm-knowledge` skill
3. 文件名 + 列名（如果有 dump_headers 结果）无法唯一确定范式
4. lead `ask_clarification("我从数据中看到 X / Y / Z 列，可能是 EPM 或 OFT，请确认实验类型")`
5. 用户回答 → lead 继续 E2E_MIN 流程

**验收**：
- ✅ lead **不**默认猜范式（也不调 `set_experiment_paradigm(paradigm="epm")`）
- ✅ ask_clarification 文案带"证据"（列名 / 文件特征），而非空泛"请问范式是什么"
- ✅ `Ev19TemplateGuardrailProvider` 未触发（因为 lead 在分类后没派 code-executor）

---

## 11. 风险登记

| # | 风险 | 严重 | 缓解 |
|---|---|---|---|
| R1 | W16 lead prompt 瘦身后丢失某些关键约束（如"不直接写 Python 脚本"）| 🔴 高 | W22 dogfood 严测 S1；保留 `GateEnforcementMiddleware` 现有 lead 行为约束 |
| R2 | W17 IntentClassification 误拦 lead 的正常 read_file（如读 skill）| 🟡 中 | provider 算法第一条 "tool != read_file" 才校验；W17 单元测试覆盖 |
| R3 | W19 自动注入与 lead 手动写占位符冲突（双注入） | 🟡 中 | task_tool 注入前先 grep 是否已有占位符，已有则跳过 |
| R4 | chart-maker 跑 catalog CLI 时 sandbox 路径翻译异常（已有过 G5 回归）| 🟡 中 | W4 实现时复用 `_resolve_virtual_workspace_dir` 三级 fallback；W22 在 sandbox 中验证 |
| R5 | report-writer 的 `required_upstream_handoffs` 不支持 optional（chart_maker 是可选）| 🟢 低 | 短期：lead 派遣时手动加 `{{handoff://chart_maker}}` 占位符（W14 已说明）；长期：W19 后续 PR 加 `optional_upstream_handoffs` 字段 |
| R6 | Q11 的"lead 不 read raw txt"约束被 prompt 瘦身后丢失，导致 lead 凭文件名乱猜 | 🟡 中 | ethoinsight-lead-interaction skill（W8）明确写入"禁止 read_file raw 数据"；如有需要可加 `RawFileReadProvider` guardrail（不在本 spec 范围） |
| R7 | 22 个 WI 横跨 catalog + subagents + skill + guardrail + harness 5 个子系统，merge conflict 风险 | 🟡 中 | 双流并行（§9.3）+ 每个 WI 单独 commit + 频繁 rebase 到 dev |
| R8 | 单被试 default y_col 选错（如给 distance_moved 而非用户期望的 open arm time）| 🟢 低 | W6 实施时 `_DEFAULT_Y_COL_BY_PARADIGM` 映射保守选择（EPM 默认 open_arm_time_ratio），允许 chart-maker 在 skill 决策时覆盖 |
| R9 | 同一架构问题第 3 次复现（5-14 → 5-18 → ???）| 🔴 高 | 本 spec 用 harness 级 guardrail 硬约束（W17 + W18），不再依赖 prompt 文字；S1 dogfood 必须 0 LOOP DETECTED |

---

## 12. 范围外（Out of Scope）

- ❌ DeerFlow 上游同步（Plan B 等本 spec 合 dev 后再做）
- ❌ 前端改动（streamdown 升级已完成）
- ❌ 微调 / golden-case / SFT 数据飞轮逻辑
- ❌ `ethoinsight/templates/<paradigm>.py` 分析模板（按学术范式组织的库代码保留不动）
- ❌ data-analyst / report-writer 的内部解读 / 写作逻辑（只补 capability metadata）
- ❌ `optional_upstream_handoffs` SubagentConfig 字段（短期手动 workaround，长期再扩展）
- ❌ Plan dataclass 旧字段的 backward-compat 兼容层（直接 W11+W13 同步改）

---

## 13. 心智模型（5 条传给实施 agent）

1. **Capability-exposure > prompt instructions**：subagent 通过 contract（when_to_use/input_contract/output_contract）宣告自己能力，lead 按 contract 调度。不要让 lead prompt 含 subagent 内部知识。
2. **Shared state via handoff files, NOT inline context**：subagent 之间通过 per-subagent handoff 文件传 facts；harness 自动注入授权（W19）；lead 不参与 facts 中转。
3. **Lead 持有交互 know-how，不持有执行 know-how**：范式识别中"用户语言 → 范式映射"归 lead（借助 skill），但"列名 → 指标脚本"归 catalog；"用户语义 → 图种"归 chart-maker。
4. **Fallback 在 catalog，不在 subagent 判断**：`_common.yaml` + `charts_fallback_available` 注册 fallback；chart-maker 的 skill 不做重复判断。
5. **Gate before guess, never default**：分辨不出必反问用户——不在 lead 层做默认猜测（删 ethovision-paradigm-knowledge default-fallback.md）。

---

## 14. 实施起步建议（给下一 agent）

1. **创建独立 worktree**（superpowers:using-git-worktrees）—— 不要碰 `spec-phase-1-handoff`
2. 用 superpowers:writing-plans 把本 spec 转 implementation plan（per-WI 拆成 step-level，每个 step 含 file path + code snippet + test command + commit message）
3. 推荐顺序：W1 → W7 → (W2 W3 W5 并行) → W6 → W4 → W8/W9/W10/W21 → W11/W12/W14/W15 → W13 → W16/W17/W18 → W19 → W22
4. 每个 WI 完成后用 superpowers:requesting-code-review 找 reviewer 把关，然后 commit
5. W22 dogfood 验证 S1/S2/S8 → 全绿后 PR 合 dev
6. 实施过程中如发现本 spec 字段命名 / 边界含糊：**先用 AskUserQuestion 反问用户**（不要默认猜，本 spec 哲学 §13.5）

---

**最后**：本 spec 是 grill-me 12 轮 + 用户硬性原则 + 5-14/5-18 双次故障教训的结晶。哪怕 LLM 实施时觉得某条 contract / guardrail "过度设计"，**不要绕过**——4 天复现 2 次就是它存在的理由（R9）。
