# Handoff 系统优化设计评审

> 评审日期：2026-06-04
> 评审对象：[`2026-06-04-handoff-task-package-optimization-design.md`](2026-06-04-handoff-task-package-optimization-design.md)
> 评审结论：**三个 Phase 全部需重构——Phase 1 收窄、Phase 2 换路径、Phase 3 砍掉**

---

## 一、评审方法论

这份 spec 的核心主张是"handoff 应加上下游可直接消费的任务上下文字段（目标/约束/失败路径/验证命令/下一步）"，方向对。问题出在**把填这些字段的负担加在了最不该加的地方**——已经 turn 预算吃紧、刚因"让模型多起草"卡死过的 subagent 上。

评审依据：
- `feedback_subagent_seal_deadlock_is_prompt_not_budget.md`（2026-06-03，commit `4caa78b8`）——subagent 卡死的真根因是 prompt 让模型多起草结构化内容，烧光 turn 预算写不到 seal
- `feedback_skill_describing_tool_output_enables_hallucination.md`（2026-06-02）——deepseek 在长 prompt 多步指令下"写 = 做"、会脑补
- `code_executor.py:79-82` `<critical_rules>`——整段哀求模型省 turn、别多做、优先写 handoff
- `executor.py:888-911`——每条 AIMessage 不看有没有 tool_call 都计入 max_turns

---

## 二、致命盲点：Phase 1 与 seal deadlock 正面冲突

### 2.1 冲突分析

spec Phase 1 要求：

> 给 code-executor 的 workflow 再加一个 step，要求 LLM 在 seal 之前额外填 7 个字段的 task_context

而 `code_executor.py:79-82` 的 `<critical_rules>` 整段在哀求：

> turn 预算珍贵。完成主流程（read plan → bash metrics → bash stats → 聚合 handoff）通常需要 8-12 个 AI message。任何额外探索都会挤压"聚合 + 写 handoff + 输出 gate_signals"的余量。优先写 handoff 和输出 gate_signals

这两条指令直接冲突。spec 要加的那 7 个字段——`attempted_paths`（"read plan → bash x6 → …"）、`failed_paths`（"方法: 原因"）、`next_steps`（"建议 report-writer 在 Discussion 段提…"）——不是抄现成数据，是要模型回顾整个会话再归纳的。这正是 6-03 证明会制造"叙述黑洞"、把 turn 烧在起草而非 seal 上的那类内容。

spec 第 162 行的"扁平 vs 嵌套"设计决策做得很细，但它解决的是**填表负担**的次要矛盾，完全没意识到主要矛盾是**该不该让这个 subagent 多填东西**。`feedback_subagent_seal_deadlock` 这条 memory 在 spec 里一次都没被引用。

### 2.2 修正方向：责任倒置

TaskContext 的语义信息大部分是**派遣时已知的**，不需要 subagent 归纳：

| 字段 | 谁知道 | 怎么填 |
|------|--------|--------|
| `objective` | **Lead** | 派遣 prompt 里本来就有 |
| `constraints` | **Lead** | paradigm/n/分组来自 experiment-context.json |
| `attempted_paths` | **Seal tool（确定性）** | seal 时自动记录调了哪些脚本，不需要 LLM 归纳 |
| `failed_paths` | 混合 | subagent 已经在 `errors` 字段记录了；lead 重派时从 errors 提取 |
| `pending_items` | 混合 | `status=partial` 时 `errors` 已有信息 |
| `file_changes` | **Seal tool（确定性）** | seal 时自动从 `output_files` 提取，不需要 LLM 手写 |
| `verify_commands` | **Seal tool（确定性）** | 固定模板——JSON 语法校验 + 文件存在检查 |
| `next_steps` | **Lead** | 派遣时根据意图状态机就知道下一步是什么 |

> ⚠️ **前置依赖**：上表中 `attempted_paths` / `file_changes` 标注的「Seal tool 自动记录/从 output_files 提取」——这依赖 handoff 字段可靠。但 §4.5（Phase 0）实证：FST 系 handoff 的数据在 `metrics`/`metrics_results` 字段而非 `metrics_summary`/`output_files` 校验器认得的字段，**seal tool 的自动提取会在 FST 范式上读不到东西**。因此这几个「确定性自动填」必须排在 Phase 0（字段统一）之后，否则同样失效。

**修正后的 Phase 1 只做一件事：passive schema 扩展。**

```python
class TaskContext(BaseModel):
    """任务包元数据——被动数据结构，字段由 seal tool 自动填充或 lead 派遣时注入"""
    model_config = ConfigDict(extra="allow")
    objective: str = Field(default="")
    constraints: list[str] = Field(default_factory=list)
    file_changes: list[str] = Field(default_factory=list)
    verify_commands: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    # 以下字段保留 schema 定义但 v0.1 不强制填充：
    attempted_paths: list[str] = Field(default_factory=list)
    failed_paths: list[str] = Field(default_factory=list)
    pending_items: list[str] = Field(default_factory=list)
```

改动范围收窄为：

| 文件 | 改动 | 状态 |
|------|------|------|
| `handoff_schemas.py` | 新增 TaskContext 模型 | **保留** |
| `handoff_schemas.py` | 4 个 handoff model 各加 `task_context: TaskContext \| None = None` | **保留** |
| `seal_handoff_tools.py` | 4 个 seal tool **不加** task_context 参数 | **砍掉**（subagent 不填） |
| `seal_handoff_tools.py` | `_seal_handoff` helper 自动从已知信息填充 task_context 的确定性字段（`file_changes` 从 `output_files` 提取，`verify_commands` 用固定模板） | **新增** |
| 4 个 subagent prompts | **不改** | **砍掉**（不往 prompt 加任何内容） |
| `lead_agent/prompt.py` | lead 在 task() prompt 中结构化声明 objective/constraints/next_steps | **新增** |

---

## 三、盲点二：Phase 2 集成路径自相矛盾

### 3.1 矛盾分析

spec 3.4 立 flag：

> 确定性检查不需要 LLM……不应消耗 LLM turn

spec 3.6 推荐方案却是：

> lead agent prompt 自检——让 lead 用自然语言 prompt 去"读 handoff、检查 failed_paths、判断 pending_items"

把一个确定性逻辑判断塞回 LLM turn，和 3.4 直接打架。而且 lead 现在 9 条硬约束已经臃肿（规则 7 和 9 文本重复冗余），再加一条"读 handoff → 检查三个子字段 → 分别推理"的多步指令，是往 deepseek prompt 里加注意力负担。

### 3.2 修正方向：走 spec 自己的备选方案

evaluator 作为纯函数挂到 `executor.py` 的 `_validate_handoff_emitted` 后面，产 warning 日志，完全不进 LLM。这才匹配 3.4 的立意。

优先级翻转：备选方案变成主方案。不改 lead prompt。

---

## 四、盲点三：Phase 3 应砍

### 4.1 问题分析

`write_progress` 让 code-executor "每跑完 5 个脚本调一次"：

- 每次 tool call = 一条 AIMessage → 计入 `max_turns=40` → 吃掉本就稀缺的预算
- code_executor 的 `<critical_rules>` 明令"任何额外探索都挤压写 handoff 余量"
- spec 3.8 设想的收益（"subagent 中途崩溃 lead 知道算到哪了"）在 30+ dogfood 故障中没有真实案例支撑——真实故障全是 seal 卡死、参数审计、契约错乱

为假想收益给最脆弱的环节加负担，违反项目铁律。

### 4.2 修正方向

砍掉 Phase 3。如果未来确实需要进度监控，应该在 `_seal_handoff` 写 manifest 时顺带记录当前时间戳（零额外 tool call），而不是让 LLM 主动调 tool。

---

## 四点五、【前置·根本】Phase 0：handoff metrics 字段三分裂（已实证活 bug）

> 这是核实过程挖出的、比三个 Phase 都更根本的问题，**spec 与本 review 第二节的责任倒置方案都默认了「handoff 字段可靠、seal tool / 校验器读得到」——这个前提经 56 个真实样本实证为假。** 必须作为 Phase 0 先修。

### 4.5.1 实证：56 个真实落盘 handoff 统计

对 `.deer-flow/**/handoff_code_executor.json` 全量 56 个真实样本统计，顶层 metrics 字段名分布：

| 顶层字段名 | 样本数 | 是否符合 schema |
|-----------|-------|----------------|
| `metrics`（list） | **27** | ❌ schema 未声明，靠 `extra="allow"` 落盘 |
| `metrics_summary`（嵌套 dict） | 24 | ✅ 符合 |
| `metrics_results`（list） | 1 | ❌ 同 metrics |
| 无 metrics 字段 / 解析异常 | 4 | — |

即真实落盘存在**至少三种互不兼容的 metrics 字段名**。schema（`handoff_schemas.py:370`）只声明 `metrics_summary`。

### 4.5.2 后果：28/56 个完整成功的 handoff 被误标 FAILED

校验器 `_check_code_executor_content`（`executor.py:104`）只查 `metrics_summary` 非空。用真实校验函数跑全量 56 样本：**28 个 `status=completed`、数据完整的成功 handoff 被判「metrics_summary is empty」**（数据其实在 `metrics`/`metrics_results` 字段）。

抽查 thread `9f77adcc`（FST, completed）：`metrics_summary=None`，但 `metrics` 字段里是完整真数据——`immobility_time`: treatment 5.52 vs control 14.56，带 output_files。

后果路径（源码坐实 `executor.py:1002-1032`）：

```
校验返回非 None（误判残缺）
  → _attempt_seal_resume 补一轮（白烧一次 LLM + token）
  → 补轮不把数据搬进 metrics_summary
  → 第 1019 行同函数重校验仍非 None
  → 第 1032 行 try_set_terminal(FAILED)
  → 完整成功的分析被强制标 FAILED → 误触发 lead 重派整个 subagent
```

### 4.5.3 这是 seal deadlock 的一条独立根因

该后果表现为「terminated without emitting handoff」——与 `feedback_subagent_seal_deadlock`（6-03）故障表象**同形但根因独立**。6-03 修的是 prompt 矛盾层（诱导模型烧 turn 不发 seal）；**字段名分裂这层是即使 subagent 完美调了 seal、数据完整落盘，只要写的是 `metrics`/`metrics_results` 而非 `metrics_summary`，校验照样判 FAILED**。两层都要查。

### 4.5.4 这直接动摇本 review §2.2 的责任倒置方案

§2.2 提出「seal tool 自动从 `output_files` 提取 `file_changes`」——**这个方向漂亮，但同样依赖字段可靠**。FST 系的 `output_files` 和 metrics 数据都在校验器/消费者不认的字段里。**因此 §2.2 的 seal tool 自动填充必须排在 Phase 0 之后**，否则在 FST 范式上同样失效。

### 4.5.5 为什么这个 bug 一直绿着 + red 锚点

`test_handoff_content_validation.py::test_nonempty_metrics_summary_passes` 用合成 fixture `{"metrics_summary":{...}}`，真实的 `metrics` 字段数据从没进过测试——合成 fixture 测理想世界、真实落盘没覆盖（同 `feedback_pr_merge_must_run_full_suite_on_shared_logic` 盲区）。

已落盘 red 锚点：`tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence`（fixture 从真实 thread 9f77adcc / 7db437e7 裁剪），`14 passed, 2 xfailed`：

- `test_real_metrics_field_handoff_should_pass` / `test_real_metrics_results_handoff_should_pass` — `xfail(strict=True)`：当前 xfail（bug 存在）；修复后会「意外 pass」→ strict 转红 → 逼回来摘 xfail。
- `test_documents_current_buggy_behavior` — pass：固化当前错误行为；修复后应同步删除。

### 4.5.6 Phase 0 修复方向

1. **统一 handoff metrics 落盘格式**：FST 系（`metrics`/`metrics_results`）与 shoaling 系（`metrics_summary`）二选一，让所有范式 code-executor 脚本输出一致结构。
2. **schema 与现实对齐**：`handoff_schemas.py` 声明真实落盘字段，不再靠 `extra="allow"` 偷渡。
3. **修 `_check_code_executor_content`**：对齐统一后的字段，摘除 red 测试的 xfail。

详见 memory `feedback_handoff_metrics_field_divergence_mislabels_failed`。

---

## 五、肯定之处

以下部分经核实无误，值得保留：

- **问题陈述（spec 1.2）**：handoff 偏"产出报告"、下游确实在重新推理上游意图——这个观察有价值
- **Infra 盘点（spec 第二章）**：文件路径、`extra="allow"`、`_seal_handoff` helper、`_validate_handoff_emitted` 的双层检查——全部属实
- **Phase 1 的 passive schema 工程手法**：`TaskContext | None = None` + `extra="allow"` 向后兼容，`_seal_handoff` 的 `model_cls(**payload)` 自动接新字段——纯粹的被动数据结构改动零风险。危险的不是 schema，是"要求 LLM 填它"
- **Phase 2 的 evaluator 纯函数设计**：防御性编程、操作 raw dict、不依赖 Pydantic——这个设计本身是好的，问题只在集成路径选错了

---

## 六、修订后的实施方案

| | 原方案 | 修订后 |
|---|---|---|
| Phase 1 schema | 加 TaskContext 模型 | **保留**（零风险） |
| Phase 1 seal tools | subagent 填 task_context | **砍掉**。seal tool 自动填确定性字段；语义字段由 lead 在 task() prompt 中提供 |
| Phase 1 subagent prompts | 加 task_context 填写指引 | **砍掉**。不往 prompt 加任何新内容 |
| Phase 1 lead prompt | 加消耗逻辑 | 改为 lead 在 task() prompt 模板中结构化声明 objective/constraints/next_steps |
| Phase 2 evaluator | 推荐 lead prompt 自检 | 改为推荐 executor 挂载（纯函数 + warning 日志，不进 LLM） |
| Phase 3 progress | 加 write_progress tool | **砍掉** |

### 修订后的实施顺序

```
Phase 0（前置·根本，详见 §4.5）: 统一 handoff metrics 字段格式 + schema/校验器对齐
  └── 摘除 TestCodeExecutorFieldNameDivergence 的 xfail（red→green 锚点）
      ⚠️ Phase 1b 的 seal tool 自动提取依赖此 Phase 完成，否则 FST 范式读不到字段

Phase 1a: handoff_schemas.py — 新增 TaskContext 模型（被动 schema，零风险）
Phase 1b: seal_handoff_tools.py — _seal_handoff helper 自动填充确定性字段（依赖 Phase 0）
Phase 1c: lead_agent/prompt.py — task() 派遣模板结构化声明 objective/constraints/next_steps
Phase 1d: 测试 — test_task_context_schema.py

Phase 2a: subagents/evaluator.py — 4 个确定性纯函数（新建，依赖 Phase 0 干净地基）
Phase 2b: executor.py — _validate_handoff_emitted 加 evaluator warning 日志
Phase 2c: 测试 — test_evaluator.py
```

Phase 3 已砍。Phase 0 是真正的前置和核心。Phase 1 和 Phase 2 可在 Phase 0 完成后独立并行。
