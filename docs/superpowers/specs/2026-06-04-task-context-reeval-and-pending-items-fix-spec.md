# Spec：task_context 价值重评估 + pending_items bug 修复 + 正名/TODO（收尾）

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-04
> 性质：收尾 + 纠偏。重评估 task_context 价值、修一个恒空 bug、给 task_context 正名并标 TODO。**取消原"读取端阶段二"。**
> 关联：
> - 写入端阶段一（已合 dev，b8d982b0）：[`2026-06-04-task-context-writeside-spec.md`](2026-06-04-task-context-writeside-spec.md)
> - 愿景背景：[`2026-06-04-handoff-contract-vision-design.md`](2026-06-04-handoff-contract-vision-design.md)
> - 铁律：`feedback_single_source_of_truth`、`feedback_subagent_seal_deadlock_is_prompt_not_budget`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`

---

## 〇、给实施 agent 的一句话

写入端 task_context 的 `pending_items` 有一个 bug：它从 `errors[]` 派生，但真实 partial 场景的 `errors[]` 恒空（partial 实为"统计检验因 n 太小被跳过"，非脚本失败），导致 pending_items 永远是空数组且误导。**删掉那段误导性派生逻辑，让 pending_items 诚实留空并注释说明，给 TaskContext 整体加价值说明 + TODO。不扩展、不删字段、不动其他三项。**

---

## 一、背景：为什么是这份 spec（三轮核实的结论）

写入端阶段一落地后，本应做"读取端阶段二（让下游/lead 消费 task_context）"。但三轮真实数据核实，连续推翻了 task_context 的预期价值：

1. **下游不需要 task_context**（核实：data-analyst/chart-maker/report-writer 各自消费的是 `data_quality_warnings`/`paradigm`/`key_findings` 等原始字段；task_context 四项对它们冗余）。
2. **lead 不读 handoff**（核实：lead 只通过 task ToolMessage 摘要 + `[gate_signals]` 块感知 subagent 结果；deerflow 无 "history message" 机制把 subagent trace 回流 lead）。
3. **真实 partial ≠ 未完成**（核实：5 个真实 partial 样本全部是"指标算完、但 n=1/n=2 统计检验被跳过"，`errors[]` 恒空；partial 的原因 `gate_signals.statistical_validity="skipped"` + `data_quality_warnings` 已充分覆盖 lead 决策需求）。

**结论**：task_context 是按"subagent 直接接力"的框架设计的，但 EthoInsight 是 **lead 中转拓扑 + 数据驱动 partial**，该框架的字段在此大多落空。详见 §二价值评估。

**处置原则（用户定）**：价值不高的部分先留 TODO，不强行做；但 bug 要修。

---

## 二、task_context 四字段价值评估（证据汇总，写进代码注释/文档）

| 字段 | 真实用户 | 数据源 | 价值判定 |
|------|---------|--------|---------|
| `file_changes` | 下游不用（output_files 更细）；lead 不读 handoff | output_files（确定性✅） | **低**——已有信息的副本 |
| `verify_commands` | 无消费者（三下游均无"验证上游"步骤） | 模板（✅） | **基本为零** |
| `failed_paths` | 下游用 data_quality_warnings 不用它 | errors（与 data_quality_warnings 重叠） | **低**——同一信息副本 |
| `pending_items` | lead（设想）；但真实 partial 原因 gate_signals 已覆盖 | **errors 派生，真实 partial 时 errors 恒空 → 恒空 bug** | **零 + bug** |

四字段无一满足"真实用户 + 不冗余 + 有内容"三者齐全。

---

## 三、本任务做什么（精确改动）

### 3.1 改动 1：删除 pending_items 的误导性派生逻辑（修 bug）

**文件**：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`
**位置**：`_build_task_context` 函数，第 222-224 行

**现状（bug）**：
```python
# pending_items: status=partial 时，未完成信息在 errors 里
if payload.get("status") == "partial":
    tc["pending_items"] = [e for e in errors if isinstance(e, str)]
```

**问题**：真实 partial 场景 `errors[]` 恒空（partial 实为统计跳过，非脚本失败），此逻辑恒产生空 list，且其存在制造"partial 时这里有未完成信息"的假象，误导未来维护者。

**改成**（删除派生、保留诚实空值 + 注释说明）：
```python
# pending_items: 暂留空。
# 真实 partial 语义是"指标已算完、但统计检验因样本量(n=1/n=2)被跳过"，
# 而非"指标未算完"——partial 的原因已由 gate_signals.statistical_validity="skipped"
# + data_quality_warnings 充分表达，lead 据此决策无需本字段。
# 当前无可靠的"未完成明细"数据源（errors 在此类 partial 时恒空）。
# TODO: 若未来出现"指标脚本失败导致 partial"的真实场景，再从
#   plan_metrics.json(计划) vs metrics_summary(实际) 的差集派生（见本 spec §五）。
# 不从 errors 派生（恒空且误导）。
```

> 即：`pending_items` 保持 `tc` 初始化时的默认 `[]`（第 196 行不动），只删 222-224 的派生块，替换为上述注释。

### 3.2 改动 2：给 TaskContext 模型正名 + 价值说明（防止重走老路）

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
**位置**：`TaskContext` 模型的 docstring

在现有 docstring 基础上补一段（不改字段定义）：

```python
class TaskContext(BaseModel):
    """任务状态包——只装「产出方独有、消费方无法自行推导」的执行事实。
    由 seal 工具确定性组装（不由 LLM 填）。

    ⚠️ 价值现状（2026-06-04 三轮真实数据核实）：当前 EthoInsight 拓扑是
    「subagent → lead 中转 → 下一个」，非「subagent 直接接力」，且 partial 多为
    「统计因样本量被跳过」而非「未完成」。经核实，本结构当前字段对下游 subagent
    冗余（下游消费原始字段），对 lead 多被 gate_signals 覆盖。故：
    - 下游 subagent prompt **不消费** task_context（保持现状，勿加"教消费"指引）。
    - 真实用户暂定为 audit/lineage（handoff 自描述）。
    - pending_items 当前恒空（无可靠未完成明细源，见 seal_handoff_tools._build_task_context 注释）。
    TODO: task_context 的整体价值待 v1.0「subagent 直接接力」拓扑或真实
    「脚本失败型 partial」场景出现后重评估。届时参考本类的字段设计。
    """
```

### 3.3 改动 3：调整测试以反映修复

**文件**：`packages/agent/backend/tests/test_seal_handoff_tools.py`

- 找到断言"status=partial 时 pending_items 从 errors 提取"的测试（阶段一新增的 _build_task_context 测试之一），改为断言"status=partial 时 pending_items 仍为空 list"（反映诚实空值的新行为）。
- 保留其余 file_changes/verify_commands/failed_paths 的测试不变（这三项行为未改）。
- 若有测试断言 pending_items 非空，会因本改动 fail——这是预期的，按新行为修正断言。

---

## 四、本任务**不做什么**（硬边界）

- ❌ **不做读取端阶段二**：取消"教下游/lead 消费 task_context"。下游 prompt 不碰（核实证明冗余）；gate_signals 不扩展（真实 partial 的 lead 需求已覆盖）。
- ❌ **不删 TaskContext 字段**：保留 schema 结构（向后兼容、不动已合 dev 的 b8d982b0 结构）。只删 seal helper 里的 pending_items 派生逻辑。
- ❌ **不改 file_changes/verify_commands/failed_paths 的派生**：它们价值虽低但无 bug、已合 dev、向后兼容，保持现状（评估结论写进注释即可，不动代码）。
- ❌ **不改 subagent prompt / lead prompt**。
- ❌ **不动 gate_signals**。
- ❌ **不收紧 extra="allow"**。

---

## 五、TODO（留作未来，不在本任务做）

记录在 TaskContext docstring + 本 spec，供未来 agent 参考：

1. **pending_items 的正确实现（若需要）**：未来若出现"指标脚本失败导致 partial"（而非统计跳过），pending_items 应从 `plan_metrics.json` 的 `metrics[].id` 集合 减去 `metrics_summary` 实际算出的 id 集合派生（核实确认两侧 id 命名一致，差集可行）。当前无此场景，不做。
2. **task_context 整体价值重评估**：v1.0 若转向"subagent 直接接力"拓扑，task_context 的设计（含 objective/next_steps，本次未做）值得重新评估。
3. **file_changes/failed_paths 的冗余清理**：若确认长期无消费者，考虑从 TaskContext 移除（当前保留，向后兼容）。

---

## 六、验收标准

1. `_build_task_context` 中 pending_items 的 errors 派生逻辑已删除，替换为说明注释；pending_items 对任何输入（含 status=partial）返回空 list。
2. 测试已调整：partial 时 pending_items 断言为空，其余三项测试不变且通过。
3. **全量回归绿**：`cd packages/agent/backend && make test`。
   - ⚠️ 已知 4 个全量测试污染（`deferred_tool_registry_promotion`×2 + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`），与本改动无关——见 `feedback_known_full_suite_test_pollution_4_tests`。失败集**正好这 4 个**则放行；多出别的才是回归。
4. `make lint` 通过。
5. TaskContext docstring 已加价值说明 + TODO。

---

## 七、提交

- 从 `dev` 切分支或在 worktree。
- commit message 中文：「修 task_context.pending_items 恒空 bug（删 errors 误导派生）+ 重评估正名 + TODO；取消读取端阶段二」。
- 不 push main、不自动建 PR，除非用户要求。

---

## 八、为什么这样收尾（依据速查）

- **修 bug 而非留着** → pending_items 从 errors 派生在真实 partial 时恒空且误导，删掉误导性死代码是诚实修复。
- **诚实留空而非硬造内容** → 用户原则"价值不高先留 todo，不强行做"；不给低价值字段硬塞 plan-diff 逻辑（那是未来真有"脚本失败型 partial"时才做）。
- **正名 + TODO 而非删字段** → 保留向后兼容（不动已合 dev 结构）；docstring 标清价值现状，防止下一个 agent 又来"教消费"重走本轮三次核实。
- **取消阶段二** → 三轮真实数据核实证明下游冗余、lead 需求已被 gate_signals 覆盖；做了是负债。
