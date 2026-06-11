# AutoResearch 启发：EthoInsight 实验日志、智能约束、快速失败与透明版本化

**日期**：2026-06-06（2026-06-06 Opus review 修正）  
**来源**：[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 分析  
**状态**：设计草案，经 review 修正，待讨论

---

## 一、背景：AutoResearch 的核心模式

AutoResearch（Karpathy）建立了一个极简的自主实验循环：

```
人类迭代 program.md（指令）
    ↓
Agent 修改 train.py（唯一可改文件）
    ↓
固定 5 分钟训练 → val_bpb 评估（不可变指标）
    ↓
记录到 results.tsv → 保留或回滚 → 循环
```

三个文件、一个循环、一个指标。整个系统的力量来自**约束的精心设计**，而非能力的无限扩展。

### 核心启发：不可变评估标准

AutoResearch 最深的架构决策是：**评估代码与实验代码的分离**。`evaluate_bpb()` 在 `prepare.py` 中，标记为 "DO NOT CHANGE"，agent 无论如何修改 `train.py` 都无法影响评估逻辑。这确保了优化目标本身不会被污染。

在 EthoInsight 中，这个"不可变评估标准"由多层组成：

| AutoResearch | EthoInsight 对应 | 当前状态 |
|---|---|---|
| `evaluate_bpb()`（不可变） | `ethoinsight/statistics.py` 确定性决策树（Shapiro-Wilk → 参数/非参数） | 约定不可变 |
| `val_bpb` 指标 | Golden Cases（专家标注的期望分析结论） | 框架就绪，内容待填充 |
| `results.tsv` 追加日志 | Experiment Log（本文 Feature 1） | 待实施 |
| `prepare.py` 不可被 agent 修改 | `ethoinsight/` 库代码不可被 code-executor 修改 | 约定不可变，待强制 |

从中提炼出 5 个可操作启发：

| # | 启发 | AutoResearch 原版 | EthoInsight 映射 |
|---|------|-------------------|------------------|
| 1 | 不可变代码边界 | Agent 只能改 train.py，评估函数不可变 | `ethoinsight/` 库从约定性不可变升级为强制不可变 |
| 2 | SkillOpt 方法论验证 | program.md 迭代 = skill 迭代，val_bpb = golden cases | SkillOpt 路线有先例验证 |
| 3 | 智能资源约束 | 固定 5 分钟训练（不适合我们） | 差异化 max_turns（已有机制）+ subagent 级 LoopDetection |
| 4 | 实验日志 | results.tsv 记录每次实验 | **用户可见的分析历史记录**（核心 feature） |
| 5 | 快速失败 | NaN/loss 爆炸立即 exit(1) | 代码强制检查（NaN/范围）+ LLM 判断（前提一致性） |

启发 2 是文档级验证（在 SkillOpt 设计文档中引用即可），不需要工程实施。其余 4 个启发各自对应一个 feature，以下逐一设计。

> **关于 Git 版本化**：AutoResearch 用 git commit/branch 管理实验状态。我们考虑过对应方案（用户透明的分析版本管理），但由于 sandbox 安全风险（见附录 B），决定 v0.1 不实施，在本文仅做简要讨论（§六）。

---

## 二、Feature 1：实验日志（Analysis Experiment Log）

### 2.1 定位：与现有系统的关系

当前 EthoInsight 有四个"记录系统"，实验日志是第五个——它们服务于完全不同的目的：

| 系统 | 记录内容 | 受众 | 用途 |
|------|----------|------|------|
| **experiment-context.json**（已有） | 每个 thread 的分析上下文：paradigm、ev19_template、column_semantics、gate 状态 | **agent pipeline**（内部） | Gate 决策、column 别名、跨 subagent 共享状态 |
| **Memory**（已有） | LLM 摘要的对话事实、偏好、上下文 | **AI agent**（内部） | 跨 session 记忆用户偏好、领域知识 |
| **Training Data**（已有） | 每 turn 完整轨迹（JSONL） | **模型训练者**（内部） | SFT/DPO 训练数据 |
| **RunRow**（已有） | run_id、thread_id、status、token 用量 | **系统运维**（内部） | 监控、计费、调试 |
| **Experiment Log**（新增） | 数据分析的完整记录：范式、指标、统计决策、发现、用户反馈 | **行为学研究员**（用户可见） | 分析历史、实验室记录本、结果对比、SkillOpt 评估基准 |

**关键区别**：Experiment Log 是**用户可见的 feature**，不是基础设施。行为学研究员打开 EthoInsight 后能看到"我过去做了哪些分析、各自得出了什么结论"。

### 2.2 与 experiment-context.json 的关系（关键架构决策）

`experiment_context.py` 中的 `set_experiment_paradigm_tool` 已经持久化了 paradigm、ev19_template、column_semantics、data fingerprint（`compute_analysis_config_id`）等字段到 `experiment-context.json`。这些字段与 ExperimentRow 的范式识别段高度重叠。

**架构决策：ExperimentRow 是 experiment-context.json + subagent handoff 文件的持久化投影（projection），而非独立的第二条写入路径。**

具体来说：
- **paradigm / ev19_template / groups / data fingerprint**：由已有的 `set_experiment_paradigm_tool` 写入 `experiment-context.json`。ExperimentRow 的对应字段从该 JSON 读取（seal 时），而非要求 lead agent 通过另一个工具重复写入。
- **metrics_computed / statistical_tests / normality_results / findings**：来自 subagent handoff 文件（`handoff_code_executor.json`、`handoff_data_analyst.json` 等），这些文件已有 schema 定义（Pydantic）。ExperimentRow 的对应字段从 handoff 文件提取（seal 时），而非要求 lead agent 解析 handoff 后手动填充——lead 是 relay 拓扑，不读 handoff 内容（MEMORY `feedback_task_context_framework_mismatches_ethoinsight_topology`）。
- **status / user_feedback / resource 统计**：seal 时综合写入。

这避免了：
1. 范式/指纹信息被 lead agent 通过两个不同工具写两遍（违反 SSOT 原则）
2. Lead agent 手动从 handoff 提取 statistical_tests 等字段（lead 不读 handoff 内容，强行要求会导致 seal-deadlock 类故障）
3. handoff 字段定义与 ExperimentRow JSON 字段的第三套分化（MEMORY `feedback_handoff_metrics_field_divergence_mislabels_failed` — metrics/metrics_results/metrics_summary 三分裂导致 28/56 误标 FAILED）

### 2.3 写入机制：Seal-time 投影

不在 pipeline 中间插入新的写入步骤。而是在**分析完成时（seal 时刻）**，由一个 middleware 或 hook 从已有的文件源投影到 ExperimentRow：

```
Seal 时刻触发（报告生成完成 或 用户接受/拒绝反馈 或 thread 关闭）
    ↓
ExperimentSealMiddleware（或 after_agent hook）:
    1. 读取 experiment-context.json → 填充 paradigm, ev19_template, groups, data_fingerprint
    2. 读取 handoff_code_executor.json → 提取 metrics_computed
    3. 读取 handoff_data_analyst.json → 提取 statistical_tests, normality_results, findings
    4. 从 RunRow / runtime 提取 wall_time_s, turns_used, total_tokens
    5. 综合 status（读 gate_signals 判断 completion_reason）
    6. upsert ExperimentRow
```

**为什么不用 `record_experiment` 工具**：让 lead agent 调用工具逐步填充的前提是 lead 能获取这些数据——但 lead 是 relay 拓扑（不读 subagent handoff 内容），且 experiment-context.json 的数据已经被 `set_experiment_paradigm_tool` 写入过一次。新增 `record_experiment` 工具只会制造第二条写入路径和更多字段分化风险。

### 2.4 Schema 设计

JSON 字段引用已有的 Pydantic schema（`handoff_schemas.py`），避免第三套字段定义：

```python
# 新增模型：packages/agent/backend/packages/harness/deerflow/persistence/experiment/model.py

from datetime import datetime, UTC
from sqlalchemy import JSON, DateTime, Index, String, Float
from sqlalchemy.orm import Mapped, mapped_column
from deerflow.persistence.base import Base


class ExperimentRow(Base):
    """一条实验记录 = 一个端到端分析 session 的完整记录。
    
    数据从 experiment-context.json + subagent handoff 文件投影而来，
    在 seal 时刻一次性写入，之后不可变。
    """

    __tablename__ = "experiments"

    # ── 标识 ──
    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # 注意：一个 experiment 可能关联多个 run（用户追问触发新 run），
    # 因此不在 ExperimentRow 上设单一 run_id FK。
    # 反向关联：RunRow 可增加 experiment_id 字段（或通过 thread_id JOIN）
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # ── 数据指纹（来自 experiment-context.json）──
    data_fingerprint: Mapped[str | None] = mapped_column(String(64))
    # sha256 of uploaded raw data files (compute_analysis_config_id)
    data_files: Mapped[dict] = mapped_column(JSON, default=dict)
    # {"filename": "exp1_EPM.xlsx", "md5": "abc123", "size_bytes": 12345}

    # ── 范式识别（来自 experiment-context.json）──
    paradigm: Mapped[str | None] = mapped_column(String(64), index=True)
    # "EPM" | "OFT" | "FST" | "LDB" | "Zero Maze" | "TST" | null
    ev19_template: Mapped[str | None] = mapped_column(String(128))
    # EthoVision XT 模板名称
    groups: Mapped[list] = mapped_column(JSON, default=list)
    # 来自 EV19 文件头 Treatment/Dose 字段（MEMORY feedback_ev19_header_has_treatment_field）
    # [{"role": "control", "label": "WT-Saline", "n": 8}, {"role": "treatment", "label": "WT-Drug", "n": 8}]
    # 用 list 而非 dict 以支持 >2 组设计

    # ── 分析路径（来自 subagent handoff 文件）──
    # JSON 字段的 item 结构引用已有 Pydantic schema：
    # - metrics_computed: 对应 handoff_schemas.py 中 MetricResult
    # - statistical_tests: 对应 handoff_schemas.py 中 StatisticalTestResult
    # - findings: 对应 handoff_schemas.py 中 Finding
    metrics_computed: Mapped[list] = mapped_column(JSON, default=list)
    statistical_tests: Mapped[dict] = mapped_column(JSON, default=dict)
    normality_results: Mapped[dict | None] = mapped_column(JSON)
    findings: Mapped[list] = mapped_column(JSON, default=list)

    # ── 质量信号 ──
    status: Mapped[str] = mapped_column(String(20), default="completed")
    # "completed" | "partial" | "failed" | "user_rejected"
    completion_reason: Mapped[str | None] = mapped_column(String(64))
    # "full_pipeline" | "insufficient_data" | "unknown_paradigm" | "user_aborted"
    user_feedback: Mapped[str | None] = mapped_column(String(20))
    # "accepted" | "revised" | "rejected" | null（未反馈）

    # ── 资源消耗（seal 时从 RunRow 汇总）──
    wall_time_s: Mapped[float | None] = mapped_column(Float)
    turns_used: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()

    # ── 时间戳 ──
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    __table_args__ = (
        Index("ix_experiments_user_paradigm", "user_id", "paradigm"),
        Index("ix_experiments_thread", "thread_id"),
    )
```

**与 v1 Schema 的关键变化**（基于 review 反馈）：

1. **移除 `run_id` 列**：一个 experiment 可关联多个 run（用户追问触发新 run），1:N 关系不能由单一 `run_id` FK 表达。如需从 run 反向查 experiment，在 `RunRow` 加 `experiment_id` 字段或在查询时通过 `thread_id` JOIN。
2. **`groups_detected` → `groups`，dict → list**：原设计用 dict 隐含 2 组假设；EV19 支持 >2 组，且分组信息来自文件头（是已知字段，非 "detected"）。
3. **JSON 字段引用已有 Pydantic schema**：`metrics_computed`/`statistical_tests`/`findings` 的 item 结构不重新定义，引用 `handoff_schemas.py` 中的对应类型，防止字段名三分裂。
4. **移除 `clarification_rounds` / `subagent_failures`**：属于运维遥测，归 `RunRow` 域（§2.1 受众划分）。
5. **`user_id` 改为 `nullable=False`**：多用户系统，列出"当前用户的实验"是核心用例。

### 2.5 实施任务

```
□ 新增 ExperimentRow 模型 + migration 脚本
  ⚠ make deploy-tar 不跑 alembic（MEMORY feedback_deploy_alembic_migration_for_added_columns）
  → migration 脚本需手动在 ECS 执行，或加入 deploy SOP
□ 在 persistence/models/__init__.py 注册 ExperimentRow（Base.metadata 需要）
□ 实现 ExperimentSealMiddleware：
  - 读取 experiment-context.json → 填充范式/指纹/groups
  - 读取 handoff_*.json → 提取 metrics/statistics/findings（复用已有 Pydantic schema 校验）
  - 从 RunRow/runtime 汇总资源消耗
  - upsert ExperimentRow
  - 位置：在 ClarificationMiddleware 之前（确保 seal 在 Command(goto=END) 之前执行）
□ API: GET /api/experiments（列表）+ GET /api/experiments/{id}（详情）
  - v0.1 MVP 只做这两个端点；compare/export/notes 后续迭代
□ 前端: "Analysis History" 列表页 + 详情页（列表+详情足够 MVP，步骤时间线后续迭代）
```

### 2.6 前端 MVP 设计

```
┌────────────────────────────────────────────────────────┐
│ Analysis History                                       │
│                                                        │
│ Filter: [Paradigm: All ▾] [Status: All ▾]             │
├────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐ │
│ │ 2026-06-06 14:32                    EPM ✓ completed │ │
│ │ WT-Saline (n=8) vs WT-Drug (n=8)                  │ │
│ │ Open arm time ↓ p=0.032, Cliff's δ=0.72           │ │
│ │ Conclusion: Anxiolytic-like effect                 │ │
│ │                              [View Details]         │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### 2.7 与 SkillOpt 的关系

Experiment Log 是 SkillOpt 的**补充评估数据源**（golden case 是设计的 benchmark，experiment log 是真实用户反馈）。`user_feedback` 三元信号（accepted/revised/rejected）比 AutoResearch 的 keep/discard 二元信号更丰富。

---

## 三、Feature 2：Harness 级不可变代码边界

### 3.1 现状

`ethoinsight/` 库在概念上是不可变的——code-executor 只通过 CLI 调用脚本。当前保护层次：

1. **ScriptInvocationOnlyProvider**：bash 白名单只放行 `ethoinsight.scripts.*`
2. **Sandbox 路径隔离**：`write_file` / `str_replace` 通过 `_resolve_and_validate_user_data_path`（`sandbox/tools.py:1778`）限定写入 `/mnt/user-data/`。而 `ethoinsight/` 作为已安装的 Python 包在 `.venv/site-packages/`，不在 `/mnt/user-data/` 下，**可能已被现有路径校验拒绝**。

### 3.2 设计（两步验证，避免过度实施）

**第一步：验证现有保护是否已足够。** 写一个 red anchor test——尝试通过 `write_file` 工具写入 `.venv` 下的 `ethoinsight/ethoinsight/metrics.py`——确认是否被 `_resolve_and_validate_user_data_path` 拒绝。如果已拒绝，Feature 2 缩减为"加回归测试 + 优化 deny 消息"。

**第二步：如果现有保护有缺口**，在 ScriptInvocationOnlyProvider 中增加写保护检查（注意：工具参数名是 `path`，不是 `file_path`）：

```python
# 在 script_invocation_only_provider.py 中扩展

# 注意：工具调用参数名是 "path"（sandbox/tools.py:1749 write_file_tool, :1818 str_replace_tool）
# 从 tool_input 读取时必须用 request.tool_input.get("path", "")

READONLY_PATH_PREFIXES = [
    "ethoinsight/ethoinsight/",  # 库代码：只能执行 scripts，不能修改
]

def _is_write_to_readonly_path(tool_input: dict) -> bool:
    """Check if a write_file/str_replace target path is read-only.
    
    注意：使用精确的路径前缀匹配而非子串匹配，避免误判
    workspace 中包含 "ethoinsight/ethoinsight/" 子串的文件。
    """
    path = tool_input.get("path", "")
    for prefix in READONLY_PATH_PREFIXES:
        if path.startswith(prefix) or f"/{prefix}" in path:
            return True
    return False
```

Deny 消息遵循"拒绝消息必须含明确指令"原则（MEMORY `feedback_deny_messages_must_direct`）：

> "Cannot modify `ethoinsight/` library code — it is immutable. To compute metrics, use `python -m ethoinsight.scripts.<paradigm>.<script>` instead."

### 3.3 范围

- v0.1 只保护 `ethoinsight/` 库代码
- 必须先验证现有路径校验是否已覆盖（优先走第一步）
- 无论第一步结果如何，都加 red anchor test

---

## 四、Feature 3：智能资源约束

### 4.1 为什么不用固定时间预算

同原设计（分析任务异构 + 用户等待体验 + 交互性）。我们需要的是**防止资源被无效消耗**的智能约束。

### 4.2 差异化 max_turns（利用已有机制）

`max_turns` 已经存在于 `SubagentConfig`（`subagents/config.py:36`，默认 50），在 `executor.py:934` 通过计数 AI 消息强制执行。**不需要新建 machinery**——只需在 `subagents/builtins/__init__.py` 配置各 subagent 的 `SubagentConfig.max_turns`：

| Subagent | 建议 max_turns | 理由 |
|----------|---------------|------|
| code-executor | 15 | 多指标可能需多次 bash 调用 |
| data-analyst | 10 | 需读取多个文件 + 审核 |
| report-writer | 5 | 单次生成任务 |
| knowledge-assistant | 8 | 可能需要多步检索 |

这些值是 `SubagentConfig` 上的一个字段，支持按范式覆盖。

### 4.3 Subagent 级 LoopDetection（复用，非重造）

当前 `build_subagent_runtime_middlewares`（`tool_error_handling_middleware.py:141`）**不包含** `LoopDetectionMiddleware`——subagent 缺乏 tool-call 去重保护。而 executor 已有 substantial 的 anti-stall 机制：handoff 校验、seal-resume 单次聚焦重试（`_attempt_seal_resume`，executor.py:726）、内容完整性检查（`_HANDOFF_CONTENT_CHECKS`）。

**方案**：将 `LoopDetectionMiddleware` 加入 `build_subagent_runtime_middlewares`（遵循 CLAUDE.md §12 "复用 deerflow 现成功能优先于自造轮子"），而非新建 `detect_stall` + 3 个 bespoke 信号。

放弃原设计的 `consecutive_empty_handoffs` 停滞信号——subagent 只 emit 一次 handoff（在结束时），不存在"连续空 handoff"的场景。

### 4.4 实施路径

- 配置 `SubagentConfig.max_turns`（每个 subagent 一行配置）
- 将 `LoopDetectionMiddleware` 加入 `build_subagent_runtime_middlewares`
- 注意 `ask_clarification` 等待用户回复时不计入停滞（已有修复 d3026f31）

---

## 五、Feature 4：语义快速失败

### 5.1 核心原则：代码能做的不交给 LLM

AutoResearch 的 `if math.isnan(loss): exit(1)` 是在**代码**中执行检查，不是让 LLM "判断 loss 是否合理"。这一原则对 EthoInsight 同样适用：NaN 检测、范围检查等确定性逻辑应该在 Python 代码中执行，LLM 做判断性检查（前提一致性、结论合理性）才是其优势。

### 5.2 检查点分层

```
                      ┌─────────────┐
                      │ 用户上传数据  │
                      └──────┬──────┘
                             │
                  ┌──────────▼──────────┐
                  │ Checkpoint 1: 数据质量 │  ← 代码（inspect_uploaded_file）
                  │ - 列数、样本量、空列    │
                  └──────────┬──────────┘
                             │
                      ┌──────▼──────┐
                      │ 指标计算     │
                      └──────┬──────┘
                             │
                  ┌──────────▼──────────────┐
                  │ Checkpoint 2: 指标合理性   │  ← 代码（compute scripts 或 catalog validator）
                  │ - 百分比 ∈ [0, 100]       │
                  │ - 距离/速度 > 0            │
                  │ - 无 NaN/Inf             │
                  │ 失败 → 标记 partial，不派遣│
                  │ data-analyst              │
                  └──────────┬──────────────┘
                             │
                      ┌──────▼──────┐
                      │ 统计分析     │
                      └──────┬──────┘
                             │
                  ┌──────────▼──────────────┐
                  │ Checkpoint 3: 统计前提    │  ← LLM（data-analyst SKILL.md）
                  │ - 检验方法与正态性结果一致？│
                  │ - 效应量方向正确？         │
                  └──────────┬──────────────┘
                             │
                      ┌──────▼──────┐
                      │ 报告生成     │
                      └──────┬──────┘
                             │
                  ┌──────────▼──────────────┐
                  │ Checkpoint 4: 结论一致性  │  ← LLM（data-analyst SKILL.md）
                  │ - "显著"声明有 p<0.05？   │
                  │ - 无禁止声明（幻觉）？     │
                  │ - 效应量方向与文字一致？   │
                  └─────────────────────────┘
```

### 5.3 代码强制检查（Checkpoint 2）

在 `ethoinsight/` 库中新增一个轻量 validator（或在现有 compute scripts 中增加断言）：

```python
# 伪代码：在 metric computation 完成后自动执行
def validate_metrics(metrics: dict) -> list[str]:
    """Return list of validation errors (empty = pass)."""
    errors = []
    for name, value in metrics.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            errors.append(f"{name}: NaN/Inf")
        if name.endswith("_pct") and not (0 <= value <= 100):
            errors.append(f"{name}: {value} out of [0, 100]")
        # ... 更多检查
    return errors
```

### 5.4 LLM 判断检查（Checkpoint 3 & 4）

在 data-analyst SKILL.md 中增加（保持原设计方向，但移除代码应做的 NaN/范围检查）：

```markdown
## Fast-fail rules

Before full interpretation, verify these. If any hard-fail, report immediately.

### Hard fail（abort interpretation）:
1. **Sample size**: any group n < 3 → mark "descriptive only, no inferential statistics"

### Soft fail（continue with limitation note）:
2. **Normality-test mismatch**: Shapiro-Wilk p < 0.05 but parametric test used → flag

### Warning（continue with caveat）:
3. **Conclusion-statistics inconsistency**: "significant" claimed but p ≥ 0.05
4. **Effect size direction**: effect direction contradicts finding text
5. **Forbidden claims**: hallucinated mechanism, absolute threshold judgment
   (MEMORY: 判读只看组间比较，不用绝对阈值)
```

### 5.5 实施

- P0：Checkpoint 2 的代码 validator（放在 `ethoinsight/` 库中，被 compute scripts 调用或作为独立 check）
- P1：data-analyst SKILL.md 补充 fast-fail 规则
- 不新增 middleware，遵循"智能在节点内·约束在节点间"哲学

---

## 六、Git 版本化：v0.1 不做

AutoResearch 用 git 管理实验状态（每个实验 = 一个 commit）。在 EthoInsight 中考虑过用户透明的分析版本管理（后台 git，前端词汇翻译），但存在以下风险：

### 安全风险

1. **`git add -A` 范围不可控**：如果工作区存在 symlink（MEMORY 已记录过 "stray ethoinsight symlink"），checkout/reset 可能通过 symlink 写入 thread 目录外
2. **`git checkout` / `reset --hard` 是任意写原语**：恢复历史树 = 写文件，但 git 操作在 backend Python 中执行，不受 `ScriptInvocationOnlyProvider` 的 `_is_path_safe` 管控（后者只 gate `bash`/`write_file` 工具）
3. **跨 thread 泄漏**：如果 git lib 被指向 thread 目录外的路径，可能访问/写入其他用户数据

### 决策

- v0.1：Experiment Log（Feature 1）自身提供"分析历史"——每条 experiment record 就是一次分析版本。Phase 1 = CRUD + JSON diff，不需要 VCS
- v0.2+：如果行为学研究员确实需要步骤级撤销（需要 user research 验证），将 git 版本化作为独立 spec 撰写，进行专项安全 review。准入条件包括：
  - working tree 严格限定在 thread 目录内
  - 禁用 symlink following
  - 每次 restore 操作重新校验所有写入路径
  - 使用 `pygit2`/`dulwich`（纯 Python，不依赖系统 git），禁止 shell-out

---

## 七、实施优先级与分期（修正）

```
v0.1（2026 年 9 月前）
├── P0: Feature 1 — Experiment Log（核心 feature）
│   ├── ExperimentRow 模型 + migration 脚本 + models/__init__.py 注册
│   ├── ExperimentSealMiddleware（从 experiment-context.json + handoff 文件投影）
│   ├── API: GET /api/experiments + GET /api/experiments/{id}（MVP 两个端点）
│   └── 前端 "Analysis History" 列表页 + 详情页
├── P1: Feature 2 — 不可变代码边界
│   ├── 第一步：red anchor test 验证现有路径校验是否已覆盖
│   └── 第二步（如需要）：在 ScriptInvocationOnlyProvider 加写保护（注意参数名 path）
├── P1: Feature 4 — 语义快速失败
│   ├── Checkpoint 2 代码 validator（ethoinsight 库内）
│   └── data-analyst SKILL.md fast-fail 规则（只含 LLM 判断检查）
└── P2: Feature 3 — 智能约束
    ├── SubagentConfig.max_turns 按 subagent 差异化配置
    └── LoopDetectionMiddleware 加入 subagent runtime middlewares

v0.2+（用户反馈后）
└── 分析版本管理独立 spec（如果需要步骤级撤销）
```

**注意**：Feature 3 & 4 因为利用了已有机制（`SubagentConfig`、`LoopDetectionMiddleware`、SKILL.md），实施成本比原估计更低。

---

## 八、与 SkillOpt 路线的协同

AutoResearch 分析最关键的验证是：**SkillOpt 路线的方法论有先例支撑**。

```
AutoResearch:  program.md ──迭代──→ 更好的 program.md
                                 ↓
                           Agent 行为改善
                                 ↓
                           val_bpb 下降

SkillOpt:      SKILL.md ──优化──→ 更好的 SKILL.md
                                 ↓
                           Agent 行为改善
                                 ↓
                           Golden Case 通过率上升
```

两者的元模式完全一致：**把指令（而非代码）作为优化对象，用一个不可变的评估标准来驱动迭代**。

Experiment Log 在这个路线中扮演关键角色：它是 golden case 的补充评估数据源；提供 `user_feedback` 信号（accepted/revised/rejected）作为 SkillOpt reward；记录真实分析路径供 skill 优化对比。

AutoResearch 的 `prepare.py`（不可变评估代码）对应 EthoInsight 的 `statistics.py`（确定性决策树）+ Golden Cases（专家标注标准）。Feature 2（保护 `ethoinsight/` 不可变）正是在确保这个评估标准不被污染——与 AutoResearch 把 `evaluate_bpb` 标记为 DO NOT CHANGE 是同一种架构本能。

---

## 九、开放问题

1. **Experiment Log 的 seal 时机**：报告生成后自动 seal？用户显式关闭 session？N 分钟无交互？需要结合前端交互设计确定。
2. **Experiment 与 Thread 的生命周期关系**：用户在同一 thread 中追问（如"重新分析对照组"），算新 experiment 还是 revision？
3. **前端 MVP 范围**：列表 + 详情是否足够？Compare 和 Export 是否 v0.1 实现？
4. **Migration 执行**：`make deploy-tar` 不跑 alembic——新增表的 migration 如何在 ECS 执行？需更新 deploy SOP。
5. **Feature 2 第一步结果**：现有路径校验是否已经阻止 agent 写 `ethoinsight/`？如果已经阻止，Feature 2 缩减为回归测试 + deny 消息优化。

---

## 附录 A：Review 验证过的关键文件

Opus review（2026-06-06）实际读取并验证了以下文件，本文修正基于这些验证：

| 文件 | 验证内容 |
|------|----------|
| `agents/middlewares/experiment_context.py` | `set_experiment_paradigm_tool` + `experiment-context.json` 已持久化 paradigm/fingerprint/groups（Feature 1 重叠） |
| `agents/lead_agent/agent.py:281-481` | 实际中间件链（本文依赖的架构上下文） |
| `persistence/run/model.py` | RunRow schema（ExperimentRow 差异化设计依据） |
| `persistence/models/__init__.py` | Base.metadata 注册要求 |
| `subagents/config.py:36` | `SubagentConfig.max_turns`（Feature 3 已有机制） |
| `subagents/executor.py:726,934` | max_turns 强制执行 + seal-resume anti-stall 机制 |
| `agents/middlewares/subagent_limit_middleware.py` | 只截断并发 task tool calls（不是 max_turns） |
| `agents/middlewares/tool_error_handling_middleware.py:141` | `build_subagent_runtime_middlewares` 不含 LoopDetectionMiddleware |
| `guardrails/script_invocation_only_provider.py:142,220` | `_is_path_safe` 路径校验 + `tool_input["path"]` 参数名 |
| `sandbox/tools.py:1749,1778,1818` | `write_file`/`str_replace` 使用参数 `path`（不是 `file_path`）；`_resolve_and_validate_user_data_path` 路径限定 |
| `guardrails/middleware.py:50` | guardrail middleware 传 `tool_input = tool_call["args"]` |

## 附录 B：AutoResearch 中我们明确不采纳的设计

| AutoResearch 做法 | 不采纳理由 |
|-------------------|-----------|
| 固定 5 分钟时间预算 | 分析任务异构，交互式产品不适合一刀切截断（→ 改用智能约束） |
| Agent 直接修改代码文件 | EthoInsight 的 agent 是分析师不是实验者（→ 保护 `ethoinsight/` 不可变） |
| 单 Agent 单文件架构 | 行为学分析需要多 Agent 流水线（复杂领域知识 + 质量审核） |
| Git 直接暴露给用户 | 行为学研究员不是程序员（→ 如果未来引入，必须完全透明） |
| 纯数值评估指标（val_bpb） | 分析质量是多维的（统计正确性 + 解读质量 + 报告可读性） |
