# Spec：写入端 task_context — seal 工具内置确定性组装（阶段一）

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-04
> 范围：**仅写入端**。seal 工具在封存 handoff 时，自动确定性组装 task_context 的 4 个字段。**不含**读取端工具（阶段二）、objective/next_steps/constraints、evaluator。
> 关联：
> - 目标来源：handoff = 结构化「任务状态包」（用户读到的最佳实践文章 + [`2026-06-04-handoff-task-package-optimization-design.md`](2026-06-04-handoff-task-package-optimization-design.md) 的 TaskContext）
> - 工程修正：[`2026-06-04-handoff-contract-vision-design.md`](2026-06-04-handoff-contract-vision-design.md)
> - 铁律：`feedback_subagent_seal_deadlock_is_prompt_not_budget`、`feedback_single_source_of_truth`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`

---

## 〇、给实施 agent 的一句话

让 handoff 携带「任务状态包」信息（文件变更、验证命令、失败路径、未完成项），但**不让 LLM 手填**——由 seal 工具在封存时从 handoff 已有数据（output_files / errors / status）**确定性组装**。subagent 一个新参数都不填、一个 turn 都不烧。**核心约束：不破坏任何现有 handoff 读取方。**

---

## 一、设计原则（决定做哪些字段、不做哪些）

### 1.1 task_context 只装「产出方独有、消费方无法自行推导的执行事实」

这是本 spec 的判据，来自架构分析：

- EthoInsight 拓扑是 `subagent →(handoff)→ lead → 决定下一步`。handoff 是给 **lead** 看的，lead 在意图判断阶段已知目标和派遣链。
- 因此 **objective / next_steps 不进 handoff**：它们的唯一真相源是 lead 的意图判断（`feedback_single_source_of_truth`）。让 subagent 在 handoff 里回填 lead 已知的信息 = 知识双存。
- **只有 subagent 执行中产生、且 lead/下游无法自行推导的事实**才进 task_context：我创建了哪些文件、我哪些方法失败了、我哪些没算完。

### 1.2 本阶段做的 4 个字段（全部满足：产出方独有 + 有确定性源 + 留空有真实损害）

| 字段 | 为什么是产出方独有 | 确定性源（已核实） | 留空的真实损害 |
|------|------------------|------------------|--------------|
| `file_changes` | 只有 subagent 知道它创建了哪些产物 | handoff 的 `output_files`（已有 `metrics/statistics/charts` 等 key→path） | 下游/lead 不知道产物在哪 |
| `verify_commands` | 基于产物文件，产出方才能给 | 模板：对 output_files 中每个文件生成 JSON 校验 + 存在检查 | 下游无法快速验证 handoff 完整性 |
| `failed_paths` | 只有 subagent 知道它试过什么失败了 | handoff 的 `errors[]`（真有结构化错误，如 `"group 'control' metric 'distance_moved': n=2 — underpowered"`） | 下游可能**重试上游已失败的方法** |
| `pending_items` | 只有 subagent 知道它没算完什么 | `status=="partial"` + `errors[]` | 下游可能**漏掉没算完的指标** |

### 1.3 本阶段**不做**的字段及原因

| 字段 | 不做原因 |
|------|---------|
| `objective` | lead 意图判断已知，知识双存（§1.1） |
| `next_steps` | lead 派遣链已知，知识双存（§1.1） |
| `constraints` | 无干净确定性源（已核实：plan_metrics.json 的 n_per_group=None；能凑的 paradigm/ev19 偏 lead 侧知识，非产出方执行事实） |

schema 里**不为这三项预留字段**——加字段永远比删字段容易；将来若发现廉价的源再加。

---

## 二、实施改动（精确到文件:行）

### 2.1 改动 1：schema 新增 TaskContext（被动数据结构）

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

新增模型（只 4 字段，附设计原则注释）：

```python
class TaskContext(BaseModel):
    """任务状态包——只装「产出方独有、消费方无法自行推导」的执行事实。

    由 seal 工具在封存时确定性组装（不由 LLM 填）。
    刻意不含 objective/next_steps/constraints：前两者的真相源是 lead 的意图判断
    （知识双存禁忌），constraints 无干净确定性源。详见 spec 设计原则。
    """
    model_config = ConfigDict(extra="allow")

    file_changes: list[str] = Field(
        default_factory=list,
        description="本 subagent 创建/修改的产物文件虚拟路径（seal 从 output_files 自动提取）。",
    )
    verify_commands: list[str] = Field(
        default_factory=list,
        description="下游验证本 handoff 完整性的命令（seal 按模板自动生成）。",
    )
    failed_paths: list[str] = Field(
        default_factory=list,
        description="已尝试且失败、下游不应重试的方法（seal 从 errors 自动派生）。",
    )
    pending_items: list[str] = Field(
        default_factory=list,
        description="未完成项（status=partial 时，seal 从 errors 自动派生）。",
    )
```

给 4 个 handoff model 各加字段（保留各自现有 `extra="allow"`）：

```python
task_context: TaskContext | None = Field(
    default=None,
    description="任务状态包（seal 工具确定性组装，向后兼容：旧 handoff 为 None）。",
)
```

> ⚠️ **不要动任何 handoff model 的 `extra="allow"`**——它是兼容性命根子（§四）。

### 2.2 改动 2：seal 工具内置确定性组装（核心）

**文件**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`

新增一个纯函数 `_build_task_context(payload: dict) -> dict`，从 payload 已有字段确定性派生 4 项：

```python
def _build_task_context(payload: dict[str, Any]) -> dict[str, Any]:
    """从 handoff payload 已有字段确定性组装 task_context 的 4 个字段。
    纯函数、无 LLM、无副作用、任何异常返回部分结果（防御性）。
    """
    tc: dict[str, Any] = {
        "file_changes": [],
        "verify_commands": [],
        "failed_paths": [],
        "pending_items": [],
    }
    try:
        # file_changes: 从 output_files 的 value（路径）提取
        output_files = payload.get("output_files") or {}
        paths = []
        for v in output_files.values():
            if isinstance(v, str):
                paths.append(v)
            elif isinstance(v, list):
                paths.extend(p for p in v if isinstance(p, str))
        tc["file_changes"] = paths

        # verify_commands: 对每个产物文件生成存在性 + JSON 校验命令（模板）
        cmds = []
        for p in paths:
            if p.endswith(".json"):
                cmds.append(f"python -m json.tool {p} > /dev/null")
            else:
                cmds.append(f"ls {p}")
        tc["verify_commands"] = cmds

        # failed_paths: 从 errors 提取（errors 是产出方记录的失败事实）
        errors = payload.get("errors") or []
        tc["failed_paths"] = [e for e in errors if isinstance(e, str)]

        # pending_items: status=partial 时，未完成信息在 errors 里
        if payload.get("status") == "partial":
            tc["pending_items"] = [e for e in errors if isinstance(e, str)]
    except Exception:
        pass  # 防御性：组装失败不影响 seal 主流程
    return tc
```

在 `_seal_handoff` helper 中调用（紧挨现有的 analysis_config_id 注入，行 200 附近）：

```python
def _seal_handoff(model_cls, filename, payload, runtime):
    workspace = _resolve_workspace(runtime)
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

    # 自动组装 task_context（确定性，subagent 无感知）。
    # 仅当 payload 未显式提供 task_context 时注入（向前兼容）。
    payload.setdefault("task_context", _build_task_context(payload))

    try:
        handoff = model_cls(**payload)
    # ... 其余不变
```

> **关键**：4 个 seal tool 的**函数签名不加任何参数**——subagent 调 seal 的方式完全不变。task_context 在 helper 内部生成。

### 2.3 不改的地方（确认清单）

- ❌ seal 4 个 tool 的签名、docstring、参数——不动（subagent 无感知）。
- ❌ code_executor.py / data_analyst.py 等 subagent prompt——不加任何 task_context 指引（避开 `feedback_subagent_seal_deadlock`）。
- ❌ 任何 handoff model 的 `extra="allow"`——不动。
- ❌ 现有 handoff 字段（metrics_summary/errors/output_files…）——不动。

---

## 三、本任务**不做什么**（硬边界）

- ❌ **不做读取端**：下游继续 LLM read_file 读 handoff（阶段二再工具化）。
- ❌ **不做 objective/next_steps/constraints**（§1.3 架构判据）。
- ❌ **不做 evaluator**。
- ❌ **不改 subagent prompt**。
- ❌ **不收紧 `extra="allow"`**（契约 sprint 的事）。
- ❌ **不碰 lead 派遣链**。

---

## 四、兼容性硬验收（用户首要关切：改 handoff 不能让现有读取方崩）

handoff 是共享契约，按 `feedback_pr_merge_must_run_full_suite_on_shared_logic`，必须证明四个读取方（data-analyst / chart-maker / report-writer / lead）加 task_context 后仍正常。已核实兼容机制（见下），但必须用测试坐实：

1. **全量回归绿**：`cd packages/agent/backend && make test`。
   - ⚠️ 已知有 4 个全量测试污染（`deferred_tool_registry_promotion`×2 + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`），与本改动无关——见 `feedback_known_full_suite_test_pollution_4_tests`。失败集若**正好是这 4 个**，放行；多出别的才是回归。
2. **向后兼容测试（旧 handoff 无 task_context）**：构造一个无 `task_context` 字段的 handoff dict，`CodeExecutorHandoff.model_validate(d)` 不抛、`read_handoff()` 返回正常、`task_context` 为 None。
3. **向前兼容测试（新 handoff 有 task_context）**：seal 写出带 task_context 的 handoff，`model_validate` 通过；`extra="allow"` 确保未声明字段也不炸。
4. **读取方不崩**：四个读取方中走代码路径的是 lead（`read_handoff()` + Pydantic）——测 `read_handoff()` 读带 task_context 的 handoff 返回正常、不触发 FAIL_CLOSED。LLM 读的三个（data/chart/report）是 read_file 读文本，加字段=多几行文本，天然兼容（无需新测，但在 spec 记录此判断）。
5. **新增单测**：`_build_task_context` 的纯函数行为——
   - output_files 有路径 → file_changes 提取正确、verify_commands 生成正确（.json 用 json.tool，其余用 ls）
   - errors 非空 → failed_paths 提取
   - status=partial + errors → pending_items 提取；status=completed → pending_items 空
   - 空 payload / 异常输入 → 返回 4 个空 list，不抛
6. **seal 端测试**：调 seal_code_executor_handoff 后，落盘 handoff JSON 含 task_context.file_changes 等 4 项且值正确。
7. `make lint` 通过。
8. **grep 确认**：搜全仓库无「handoff 字段集合必须相等」的断言、无前端固定 schema 渲染（已初步核实无，实施 agent 复查一次）。

### 已核实的兼容机制（供实施 agent 理解，不需重新验证）
- 4 个 handoff model 全是 `ConfigDict(extra="allow")`（handoff_schemas.py），Pydantic 接受未声明字段。
- `read_handoff()` 默认 WARN 模式（schema 违规只记日志不抛），非 FAIL_CLOSED。
- 下游 prompt 无「handoff 必须恰好这些字段」断言；`task_context` 名字当前不存在，不撞。

---

## 五、提交

- 在 worktree 或从 `dev` 切分支（项目规范：先进 dev）。
- commit message 中文：「seal 内置 task_context 确定性组装（写入端阶段一）：file_changes/verify_commands/failed_paths/pending_items，subagent 零手填，向后兼容」。
- 不 push main、不自动建 PR，除非用户要求。

---

## 六、为什么这样做（设计依据速查）

- **不让 subagent 填** → 避开 `feedback_subagent_seal_deadlock`（让 subagent 起草自由文本会烧 turn 卡死 seal）。
- **不做 objective/next_steps** → `feedback_single_source_of_truth`（lead 意图判断是唯一真相源，不双存）。
- **只做有确定性源的 4 项** → 留空有真实损害（重试失败/漏指标）+ 数据已在 handoff 里，零 LLM 即可派生。
- **seal 内置而非独立工具** → 复用 seal 已有的「自动注入」模式（analysis_config_id 就是这么注入的），不增 tool_call、不增编排。
- **保留 extra="allow"** → v0.1 兼容性命根子；收紧留给契约 sprint。
