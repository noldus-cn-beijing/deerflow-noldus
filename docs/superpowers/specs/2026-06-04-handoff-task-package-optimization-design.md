# Handoff 系统优化：Planner-Generator-Evaluator 最佳实践落地

> ⚠️ **已评审，结论需重大修订——实施前务必先读评审意见：**
> [`2026-06-04-handoff-task-package-optimization-design-review.md`](2026-06-04-handoff-task-package-optimization-design-review.md)
>
> 评审摘要：本 spec 问题诊断方向有洞察，但**三个 Phase 全部需重构**——Phase 1（subagent 填 7 字段）与 `feedback_subagent_seal_deadlock`（6-03 实证根因）正面冲突；Phase 2 推荐方案自相矛盾，应走备选；Phase 3 应砍。**且存在一个更根本、已实证的前置 bug：handoff metrics 字段三分裂（56 真实样本中 28 个完整成功的 handoff 被误标 FAILED），应作为 Phase 0 先修。** 下文正文保留原样供对照，**不要直接照此实施**。

> 状态：~~设计文档，待 review~~ → **已 review，需按修订意见重构（见上方横幅）**
> 日期：2026-06-04
> 关联：基于 agent 长任务最佳实践（planner-generator-evaluator 模式）的 handoff 语义升级

---

## 一、问题陈述

### 1.1 背景

EthoInsight 的 agent 流水线是 `lead → code-executor → data-analyst → chart-maker → report-writer`，subagent 之间通过 workspace 下的 `handoff_*.json` 文件传递结果。这套机制在 infra 层已经比较成熟：

- `seal_*_handoff` first-party tool：LLM 填参数 → Pydantic 校验 → atomic write → lineage manifest
- `HandoffPlaceholderInjectionMiddleware`：lead 派遣时自动注入 `{{handoff://X}}` 授权占位符
- `HandoffIsolationProvider`：subagent 只能读被授权的 peer handoff 文件
- `_validate_handoff_emitted`：检查文件存在 + 核心字段非空，空内容触发重派

### 1.2 核心问题

**当前 handoff 是"产出报告"（output report）而非"任务交接包"（task package）。**

以 `CodeExecutorHandoff` 为例，当前字段：

| 字段 | 语义 | 类别 |
|------|------|------|
| `status` | 执行状态 (completed/partial/failed) | 产出元信息 |
| `summary` | 一句话总结 | 产出元信息 |
| `metrics_summary` | 分组 → 指标 → 统计值 | **产出内容** |
| `per_subject` | 每个 subject 的原始数据 | **产出内容** |
| `statistics` | 组间统计检验结果 | **产出内容** |
| `output_files` | 产物文件路径表 | **产出内容** |
| `data_quality_warnings` | 数据质量警告 | 质量标注 |
| `errors` | 错误信息列表 | 异常记录 |
| `gate_signals` | 给 lead 的决策信号 | 调度信号 |

这些字段完整描述了"code-executor 产出了什么"，但下游 agent（data-analyst）拿到 handoff 后，仍然需要自己判断：

- **这个 handoff 是完整的还是残缺的？**（`status=partial` 时，哪些指标没跑、为什么？）
- **上游尝试了什么、放弃了什么？**（`errors` 记录了错误，但没记录"尝试过的恢复路径"）
- **我应该先验证什么再开始消费？**（没有 verify 指令）
- **上游建议我重点看什么？**（`gate_signals` 有质量信号，但没有语义建议）

这些问题当前靠 data-analyst 自己读 handoff + 推理来解决——这正是 agent 长任务中"跑到后边注意力缺失"的典型诱因。每一层 subagent 都在重新推理上游的意图和状态，而非直接接手一个明确的任务包。

### 1.3 与 agent 最佳实践的差距

针对长任务 agent，业界共识（也是本项目实践中反复验证的）三点：

1. **状态外部化**：进度、意图、失败路径不能只存在对话中，必须在文件系统中可检索
2. **任务分段化**：每个 subagent 有明确边界，输入/输出/职责清晰
3. **交接结构化**：handoff 不是聊天摘要，而是下一轮 agent 能直接接受的任务包——目标、约束、已完成项、失败路径、文件变更、验证命令、下一步

当前状态：

| 最佳实践 | 当前水平 | 差距 |
|----------|---------|------|
| 状态外部化 | 🟢 较好 | workspace 文件 + memory 系统 + manifest lineage |
| 任务分段化 | 🟢 较好 | DeerFlow subagent 系统 + required_upstream_handoffs |
| 交接结构化 | 🟡 有基础但语义不足 | seal tool + Pydantic schema 保证了**格式**，但 handoff 的**内容语义**仍是"产出报告"而非"任务包" |
| Evaluator 角色 | 🔴 缺失 | 没有独立的"我们做对了吗"检查，data-analyst 的事后审计不能替代门禁 |

---

## 二、DeerFlow Infra 的支撑（已有基础）

DeerFlow 为这三个最佳实践提供了扎实的 infra，Noldus 在其上做了领域定制：

### 2.1 状态外部化

| 能力 | 提供方 |
|------|--------|
| Per-thread 隔离目录 (`user-data/{workspace,uploads,outputs}`) | DeerFlow `ThreadDataMiddleware` |
| Sandbox 虚拟路径 (`/mnt/user-data/...` ↔ 宿主机) | DeerFlow `SandboxMiddleware` |
| 跨会话记忆持久化 (`memory.json`) | DeerFlow `MemoryMiddleware` |
| 上下文压缩外存化 (`conversation_summary.md`，不塞回 prompt) | **Noldus** `ArchivingSummarizationMiddleware` |
| 实验上下文文件 (`experiment-context.json`) | **Noldus** 定制 |
| 不可变 handoff 文件 + lineage manifest | **Noldus** 定制 |

### 2.2 任务分段化

| 能力 | 提供方 |
|------|--------|
| `task()` 派遣工具（`subagent_type`, `prompt`, `description`） | DeerFlow 原生 |
| `SubagentConfig`（独立 system_prompt/tools/max_turns/timeout/skills） | DeerFlow 原生 |
| `SubagentExecutor`（双线程池、15min 超时、recursion_limit 自动计算） | DeerFlow 原生 |
| `SubagentLimitMiddleware`（MAX_CONCURRENT_SUBAGENTS=3） | DeerFlow 原生 |
| `required_upstream_handoffs`（subagent 声明依赖，harness 自动注入授权） | **Noldus** 定制 |
| 意图状态机（`INTENT → 派遣链`） | **Noldus** 定制 |
| `when_to_use` / `input_contract` / `output_contract`（能力曝光契约） | **Noldus** 定制 |

### 2.3 交接结构化

| 能力 | 提供方 |
|------|--------|
| `SubagentResult`（status/result/error/ai_messages） | DeerFlow 原生 |
| SSE 事件流（task_started → running → completed/failed/timed_out） | DeerFlow 原生 |
| `seal_*_handoff` first-party tools（Pydantic 校验 + atomic write） | **Noldus** 定制 |
| `handoff_schemas.py`（5 个 Pydantic BaseModel） | **Noldus** 定制 |
| `HandoffPlaceholderInjectionMiddleware`（自动注入授权占位符） | **Noldus** 定制 |
| `HandoffIsolationProvider`（subagent 隔离读 peer 文件） | **Noldus** 定制 |
| `_validate_handoff_emitted`（文件存在 + 核心字段非空检查） | **Noldus** 定制 |
| `.lineage/manifest.json`（sha256 + config_id + timestamp） | **Noldus** 定制 |

**关键洞察**：DeerFlow 解决了"agent 之间怎么可靠地交接"的 infra 问题（铁轨、信号系统、集装箱）。Noldus 在集装箱里定义了"货物清单格式"（handoff schema）。**本轮优化要解决的是"货物清单上该写什么才能让下游直接接手"——这是语义层的问题，infra 不需要大改。**

---

## 三、设计方案（经评审修订）

> **修订原则**：subagent prompt 不加任何新内容——`feedback_subagent_seal_deadlock`（6-03 实证）证明给 turn 预算吃紧的 subagent 加"归纳自由文本"指令会制造叙述黑洞。所有 task_context 的填充走三条非 LLM 路径：
> 1. **Seal tool 自动填**（确定性字段：file_changes、verify_commands）
> 2. **Lead 派遣时填**（派遣时已知的语义字段：objective、constraints、next_steps）
> 3. **已有机制度量**（errors 已有 failed 信息，status 已有 partial 信息，不重复）
>
> ⚠️ **前置依赖**：上表中「seal tool 自动提取 file_changes」依赖 handoff 字段可靠——但 Phase 0（见下）实证 FST 系 handoff 的 output_files/metrics 在校验器不认的字段名下。因此 Phase 1b / Phase 2 必须在 Phase 0 完成后才生效，否则 seal tool 在 FST 范式上提取不到东西、evaluator 对 28/56 成功 handoff 误报。

### Phase 0（前置·根本）：Handoff Metrics 字段统一

> 这是 review 过程挖出的、比 Phase 1/2 都更根本的前置问题。详见评审文档 §4.5。**Phase 1b 和 Phase 2 依赖此阶段的干净地基。**

#### 0.1 实证

对 `.deer-flow/**/handoff_code_executor.json` 全量 56 个真实样本统计，顶层 metrics 字段名分布：

| 顶层字段名 | 样本数 | 是否符合 schema |
|-----------|--------|----------------|
| `metrics`（list） | **27** | ❌ schema 未声明，靠 `extra="allow"` 落盘 |
| `metrics_summary`（嵌套 dict） | 24 | ✅ 符合 |
| `metrics_results`（list） | 1 | ❌ 同 metrics |
| 无 metrics 字段 / 解析异常 | 4 | — |

即真实落盘存在**至少三种互不兼容的 metrics 字段名**。schema 只声明 `metrics_summary`。

#### 0.2 后果

校验器 `_check_code_executor_content`（`executor.py:104`）只查 `metrics_summary` 非空。跑全量 56 样本：**28 个 `status=completed`、数据完整的成功 handoff 被判「metrics_summary is empty」**（数据其实在 `metrics`/`metrics_results` 字段）。

抽查 thread `9f77adcc`（FST, completed）：`metrics_summary=None`，但 `metrics` 字段里是完整真数据。这表现为「terminated without emitting handoff」——与 seal deadlock 同形但根因独立。即使 subagent 完美调了 seal、数据完整落盘，只要字段名是 `metrics`/`metrics_results` 而非 `metrics_summary`，校验照样判 FAILED。

#### 0.3 修复方向

1. **统一 handoff metrics 落盘格式**：FST 系（`metrics`/`metrics_results`）与 EPM 系（`metrics_summary`）统一到一个字段名
2. **schema + 校验器对齐**：`CodeExecutorHandoff.metrics_summary` 与统一后的字段名一致，`_check_code_executor_content` 只认统一后的字段
3. **灰盒修复**：不改变 seal 子工具接口，只修 code-executor 脚本的 JSON 输出字段 + executor 校验器认的字段名
4. **red 锚点已就绪**：`tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence`（2 xfail strict，修复后自动转红 → 摘 xfail）

#### 0.4 为什么这个 bug 一直绿着

`test_handoff_content_validation.py::test_nonempty_metrics_summary_passes` 用合成 fixture `{"metrics_summary":{...}}`，真实的 `metrics` 字段数据从没进过测试——合成 fixture 测理想世界、真实落盘没覆盖（同 `feedback_pr_merge_must_run_full_suite_on_shared_logic` 盲区）。

#### 0.5 改动范围

| 文件 | 改动 |
|------|------|
| `ethoinsight/scripts/fst/compute_*.py` 等 | 输出字段名从 `metrics`/`metrics_results` 统一到 `metrics_summary` |
| `subagents/handoff_schemas.py` | 确认 `CodeExecutorHandoff.metrics_summary` 是唯一 metrics 字段 |
| `subagents/executor.py` | `_check_code_executor_content` 确认只查 `metrics_summary` |
| `tests/test_handoff_content_validation.py` | 摘除 `TestCodeExecutorFieldNameDivergence` 的 2 个 xfail（strict=True，修复后自动转红 → 手动摘 xfail → 变绿） |

---

### Phase 1：Handoff Schema 增强 — TaskContext（被动数据结构）

> ⚠️ Phase 1b（seal tool 自动提取 file_changes）依赖 Phase 0 完成。FST 系 handoff 的 `output_files` 在校验器认得的字段名下才有东西可提取。

#### 3.1 新增模型

在 `handoff_schemas.py` 中新增一个 `TaskContext` 模型。**关键约束：subagent 不填此字段，不由 LLM 归纳。** 字段由 seal tool 自动填充或 lead 派遣时注入。

```python
class TaskContext(BaseModel):
    """任务包元数据——被动数据结构，不由 subagent 的 LLM 填充。

    字段来源：
    - seal tool 自动填充: file_changes, verify_commands
    - lead 派遣时注入: objective, constraints, next_steps
    - v0.1 保留 schema 定义但不强制填充: attempted_paths, failed_paths, pending_items
    """
    model_config = ConfigDict(extra="allow")

    # === seal tool 自动填充 ===
    file_changes: list[str] = Field(
        default_factory=list,
        description="seal tool 自动从 output_files + handoff 自身路径提取，不需要 LLM 手写。"
    )
    verify_commands: list[str] = Field(
        default_factory=list,
        description="seal tool 按固定模板生成，如 JSON 语法校验 + 产物文件存在检查。"
    )

    # === lead 派遣时注入 ===
    objective: str = Field(
        default="",
        description="派遣目标，lead 在 task() prompt 中声明。"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="约束清单，lead 从 experiment-context.json + 意图状态机提取。"
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="建议下游动作，lead 根据意图状态机已知下一步。"
    )

    # === v0.1 保留定义，不强制填充 ===
    attempted_paths: list[str] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    pending_items: list[str] = Field(default_factory=list)
```

给 4 个 handoff model 各加一个字段：

```python
task_context: TaskContext | None = Field(default=None, description="任务包元数据（被动填充）")
```

#### 3.2 设计决策

**为什么不让 subagent 填？**

`feedback_subagent_seal_deadlock_is_prompt_not_budget`（2026-06-03，commit `4caa78b8`）实证：subagent 卡死「terminated without emitting handoff」的真根因是 prompt 让模型多起草结构化内容，烧光 turn 预算写不到 seal。`attempted_paths`（"read plan → bash x6 → …"）、`failed_paths`（"方法: 原因"）这类需要回顾会话再归纳的字段，正是会制造"叙述黑洞"的内容。

且 `code_executor.py:79-82` 的 `<critical_rules>` 整段在哀求模型省 turn、别多做、优先写 handoff。给同一个 subagent 加 task_context 填写步骤是反向操作。

**TaskContext 的语义信息大部分是派遣时已知的，不需要 subagent 归纳：**

| 字段 | 谁知道 | 怎么填 |
|------|--------|--------|
| `objective` | **Lead** | 派遣 prompt 里本来就有 |
| `constraints` | **Lead** | paradigm/n/分组来自 experiment-context.json |
| `file_changes` | **Seal tool（确定性）** | seal 时自动从 `output_files` 提取 |
| `verify_commands` | **Seal tool（确定性）** | 固定模板——JSON 语法校验 + 文件存在检查 |
| `next_steps` | **Lead** | 派遣时根据意图状态机就知道下一步 |

#### 3.3 改动范围

| 文件 | 改动 | 说明 |
|------|------|------|
| `subagents/handoff_schemas.py` | 新增 `TaskContext` 模型；4 个 handoff model 各加 `task_context: TaskContext \| None = None` | 被动 schema，零风险 |
| `tools/builtins/seal_handoff_tools.py` | `_seal_handoff` helper 自动从 `output_files` 提取 `file_changes`，按固定模板生成 `verify_commands` | seal tool 不改签名，subagent 无感知 |
| `agents/lead_agent/prompt.py` | lead 在 task() prompt 模板中结构化声明 objective/constraints/next_steps | lead 本来就在 prompt 里写这些，只是结构化 |
| 4 个 subagent prompts | **不改** | 不往 prompt 加任何新内容 |

---

### Phase 2：Evaluator Gate（确定性纯函数，挂 executor）

#### 3.4 设计思路

当前 pipeline 中，data-analyst 承担了部分 evaluator 职责（step 2.8 审计 parameters_used vs 数据分布），但这是**事后审计**而非**门禁**。一个独立 evaluator 的价值在于：

- **确定性检查不需要 LLM**：交叉校验（metrics_summary 非空但 status=completed? partial 但没有 pending_items?）是纯逻辑判断，不应消耗 LLM turn
- **挂 executor 不进 LLM**：在 `_validate_handoff_emitted` 的"核心字段非空"检查之后跑，产 warning 日志，完全不进 LLM 上下文
- **失败不阻断，但必须记录**：evaluator 返回的 issues 作为 warning 日志；阻塞级问题（如 critical+blocks_downstream）由现有的 `_validate_handoff_emitted` → FAILED 路径处理

#### 3.5 新增模块

`subagents/evaluator.py` — 4 个纯函数，收 parsed handoff dict → 返回 `list[str]`（空 = 通过）：

| 函数 | 检查项 | Phase 0 依赖 |
|------|--------|-------------|
| `evaluate_code_executor_handoff(d)` | `status=completed` 但 `metrics_summary` 空（⚠️ 依赖 Phase 0：字段统一后此检查才准确，否则对 FST 系 28/56 成功 handoff 误报）；`data_quality_warnings` 含 critical+blocks_downstream；`statistics` 有但 `per_subject` 空 | **依赖 Phase 0** |
| `evaluate_data_analyst_handoff(d)` | `completed` 但 `key_findings` 空；`outlier_findings` 中 counterfactual 为 null；`parameter_audit` 含 critical+blocks_downstream | 无 |
| `evaluate_chart_maker_handoff(d)` | 全部 chart 失败；`completed` 但 `chart_files` 和 `failed_charts` 都空 | 无 |
| `evaluate_report_writer_handoff(d)` | `completed` 但 `sections_written` 少于 6 段 | 无 |

**关键设计**：操作 raw dict（不依赖 Pydantic），防御性编程（任何异常返回空 list，绝不抛），可从中介软件直接调用无循环导入。注意 `evaluate_code_executor_handoff` 的 `metrics_summary` 检查依赖 Phase 0 字段统一——统一前它会复制 `_check_code_executor_content` 的误报。

#### 3.6 集成方式

**主方案：挂 executor（备选→主方案）**

在 `executor.py` 的 `_validate_handoff_emitted` 中，核心字段非空检查通过后，调 evaluator。返回的 issues 以 `logger.warning` 记录（不进 LLM，不阻断）。这匹配 3.4 的立意——确定性检查不需要 LLM。

```python
# executor.py _validate_handoff_emitted 中，核心字段非空检查通过后：
from deerflow.subagents.evaluator import (
    evaluate_code_executor_handoff,
    evaluate_data_analyst_handoff,
    evaluate_chart_maker_handoff,
    evaluate_report_writer_handoff,
)

_EVALUATORS = {
    "code-executor": evaluate_code_executor_handoff,
    "data-analyst": evaluate_data_analyst_handoff,
    "chart-maker": evaluate_chart_maker_handoff,
    "report-writer": evaluate_report_writer_handoff,
}

# 在核心字段非空通过后：
evaluator = _EVALUATORS.get(subagent_name)
if evaluator is not None:
    try:
        issues = evaluator(parsed_handoff)
        for issue in issues:
            logger.warning("[evaluator_gate] subagent=%s: %s", subagent_name, issue)
    except Exception:
        pass  # 防御性：evaluator 异常绝不影响主流程
```

**为什么不用 lead prompt 自检？**

把确定性逻辑判断塞回 LLM turn，与 3.4 的立意直接矛盾。且 lead 现在 9 条硬约束已经臃肿（规则 7 和 9 文本重复冗余），再加多步指令是往 deepseek prompt 加注意力负担。

#### 3.7 改动范围

| 文件 | 改动 |
|------|------|
| `subagents/evaluator.py` | **新建**，4 个确定性纯函数 |
| `subagents/executor.py` | `_validate_handoff_emitted` 中加 evaluator warning 日志（核心字段非空后） |
| `agents/lead_agent/prompt.py` | **不改**（不加第 10 条硬约束） |

---

### Phase 3：已砍

原方案（`write_progress` tool）已砍。理由：

- 每次 tool call = 一条 AIMessage → 计入 `max_turns=40` → 吃掉本就稀缺的预算
- code_executor 的 `<critical_rules>` 明令"任何额外探索都挤压写 handoff 余量"
- 30+ dogfood 故障中没有一个真实案例是"subagent 崩溃后 lead 不知进度"——真实故障全是 seal 卡死、参数审计、契约错乱
- 为假想收益给最脆弱的环节加负担，违反项目铁律

如果未来确实需要进度监控，应在 `_seal_handoff` 写 manifest 时顺带记录时间戳（零额外 tool call），而非让 LLM 主动调 tool。

---

## 四、向后兼容

- 所有新字段 `default=None` 或 `default_factory`，旧 handoff 文件照常解析
- `ConfigDict(extra="allow")` 保证未来加字段不破坏旧消费者
- subagent 不感知 task_context 的存在——`_seal_handoff` helper 内部自动处理确定性字段
- `model_cls(**payload)` 自动接新字段，seal tool 签名不变

## 五、测试策略

| Phase | 测试文件 | 覆盖点 |
|-------|---------|--------|
| 0 | `tests/test_handoff_content_validation.py`（已有） | `TestCodeExecutorFieldNameDivergence`：2 个 xfail(strict=True) 锚点——修复后自动转红 → 摘 xfail → 变绿 |
| 1 | `tests/test_task_context_schema.py`（新建） | TaskContext 默认值、部分填充、未知字段忽略；4 个 handoff model 接受 task_context |
| 1 | `tests/test_seal_handoff_tools.py`（扩展） | `_seal_handoff` 自动填充 file_changes 和 verify_commands（依赖 Phase 0 完成后字段可靠） |
| 2 | `tests/test_evaluator.py`（新建） | 4 个 evaluator 函数的 happy path + 边界（空 dict、缺失字段、错误类型、异常不抛）；code-executor evaluator 用统一后字段构造 fixture |

所有阶段完成后跑 `make test` 全量，确认零退化。

## 六、实施顺序

```
Phase 0（前置·根本，必须最先做）:
  统一 handoff metrics 字段格式 → schema/校验器对齐 → 摘 TestCodeExecutorFieldNameDivergence xfail
  （详见评审文档 §4.5 及正文 §三·Phase 0）

Phase 1a: handoff_schemas.py — 新增 TaskContext 模型（被动 schema，零风险）
Phase 1b: seal_handoff_tools.py — _seal_handoff helper 自动填充确定性字段（依赖 Phase 0）
Phase 1c: lead_agent/prompt.py — task() 派遣模板结构化声明 objective/constraints/next_steps
Phase 1d: 测试 — test_task_context_schema.py + 扩展 seal tool 测试

Phase 2a: subagents/evaluator.py — 4 个确定性纯函数（新建，依赖 Phase 0 干净地基）
Phase 2b: executor.py — _validate_handoff_emitted 加 evaluator warning 日志
Phase 2c: 测试 — test_evaluator.py
```

Phase 0 是真正的前置和核心——Phase 1b 和 Phase 2 的 `evaluate_code_executor_handoff` 都依赖它的字段统一。Phase 1 和 Phase 2 可在 Phase 0 完成后独立并行。Phase 3 已砍。
