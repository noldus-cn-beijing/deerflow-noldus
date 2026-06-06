# Handoff：handoff 系统优化评审 → task_context 三 commit 落地 + 框架认知收口

> 日期：2026-06-04
> 性质：一次长会话的交接。从「评审一份 handoff 优化 spec」演变为「修 2 个真 bug + 证伪 task_context 价值 + 沉淀一个架构认知」。
> 给下一个 Agent：本文档是给 AI 看的操作性交接。**最重要的一句：task_context 这条线已收口，别再来「让下游/lead 消费 task_context」——三轮真实数据核实已证伪，见下方「风险」。**

---

## 1. 当前任务目标（起点与终点）

**起点**：用户让评审一份 spec —— `docs/superpowers/specs/2026-06-04-handoff-task-package-optimization-design.md`（把 handoff 升级为「任务状态包」：目标/约束/失败路径/文件变更/验证命令/下一步七要素）。

**终点（已达成）**：评审推翻原 spec 的实现方式 → 挖出并修了 2 个真 bug → 用三轮真实数据核实证明 task_context 在 EthoInsight 拓扑下价值低 → 收口正名留 TODO。**3 个 commit 全部合 dev、review 通过、已 push。**

---

## 2. 当前进展（全部 ✅，无半成品）

| commit | 内容 | 状态 |
|--------|------|------|
| （Phase 0，早于本会话主线但本会话 review 过）| handoff metrics 字段三分裂 → 校验器加固（`_check_code_executor_content` 认 metrics_summary/metrics/metrics_results 三字段等价集 + 非规范格式记 warning）| ✅ 合 dev |
| `b8d982b0` | 写入端阶段一：seal 工具内置 `_build_task_context`，从 output_files/errors/status 确定性派生 4 字段（file_changes/verify_commands/failed_paths/pending_items），subagent 零手填、向后兼容 | ✅ 合 dev，已 push |
| `48ac5003` | 收尾：删 pending_items 恒空 bug（从 errors 派生在真实 partial 时恒空且误导）+ TaskContext docstring 价值评估正名 + 取消读取端阶段二 | ✅ 合 dev，已 push |

**全量回归**：`3717 passed, 4 failed`，4 failed 是已知 test 污染（见风险 §7），非本改动。

---

## 3. 关键上下文（文件/位置）

**产出文档**（都在 `docs/superpowers/specs/`）：
- `2026-06-04-handoff-task-package-optimization-design.md` — 原 spec（头部有评审横幅，正文改写为修订方案；**早期定性「Phase 0=活bug」已被后续核实降级，以下面 review/contract 文档为准**）
- `2026-06-04-handoff-task-package-optimization-design-review.md` — 评审意见（三盲点 + Phase 0 实证 §4.5）
- `2026-06-04-handoff-contract-vision-design.md` — 愿景文档（handoff = 节点间声明式契约；§五形态决策①Pydantic/②SSOT **未拍板，留着**）
- `2026-06-04-task-context-writeside-spec.md` — 写入端 spec（已实施）
- `2026-06-04-task-context-reeval-and-pending-items-fix-spec.md` — 收尾 spec（已实施）
- `2026-06-04-phase0-handoff-validator-hardening-spec.md` — Phase 0 spec（已实施）

**核心代码**（harness 在 `packages/agent/backend/packages/harness/deerflow/`）：
- `tools/builtins/seal_handoff_tools.py` — `_build_task_context`（187+）确定性组装；`_seal_handoff`（230+）helper，第 200 行注入 analysis_config_id、紧邻注入 task_context
- `subagents/handoff_schemas.py` — `TaskContext` 模型（含价值评估 docstring + TODO）；4 个 handoff model 各有 `task_context: TaskContext | None`；全是 `ConfigDict(extra="allow")`
- `subagents/executor.py:104` — `_check_code_executor_content`（Phase 0 加固后认三字段等价集）；`_validate_handoff_emitted`（146+）；`_HANDOFF_CONTENT_CHECKS`（129+）
- `subagents/builtins/code_executor.py:96-126` — gate_signals 输出格式（lead 的主要决策信息源）

**测试**：
- `tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence` — Phase 0 red 锚点（xfail 已摘，现 pass）
- `tests/test_seal_handoff_tools.py` — `_build_task_context` 测试（含 `test_partial_status_yields_empty_pending_items` 断言空）

---

## 4. 关键发现（最值钱的产出，已存 memory）

**核心认知**（memory: `feedback_task_context_framework_mismatches_ethoinsight_topology`）：
> 业界「handoff = 给下游直接接力的任务状态包」框架**只适配 EthoInsight 一半**。两个拓扑事实让它大半落空：
> 1. **lead 中转，非 subagent 直接接力**：`subagent → lead → 下一个`。接手的是 lead，下游 subagent 不直接接上游的包 → objective/next_steps 是 lead 意图判断已知的（双存禁忌），下游不需要执行状态包。
> 2. **数据驱动 partial，非失败 partial**：真实 partial 是「指标算完、但 n=1/n=2 统计检验被跳过」，errors 恒空 → failed_paths/pending_items 落空或与 data_quality_warnings 重复。

**lead 感知 subagent 结果的真实通道**（核实，无第三通道）：只有 (a) task ToolMessage（进度时间线标签 + 最后一条 AIMessage 全文）+ (b) subagent 输出的 `[gate_signals]` 块。**deerflow 无 "history message" 机制回流 subagent trace；lead 不读 handoff 文件。**

**两个被修的真 bug**：
- 字段三分裂 → 28/56 个 status=completed 完整 handoff 被误标 FAILED（memory: `feedback_handoff_metrics_field_divergence_mislabels_failed`）
- pending_items 从 errors 派生在真实 partial 时恒空且误导（本会话收尾修）

---

## 5. 未完成事项（按优先级）

**P2（留 TODO，无痛点不做）**：
- task_context 整体价值待 v1.0「subagent 直接接力」拓扑或真实「脚本失败型 partial」场景出现后重评估（已写进 TaskContext docstring + 收尾 spec §五）
- pending_items 正确实现（若未来真有脚本失败型 partial）：从 `plan_metrics.json` 的 metrics[].id 减 metrics_summary 实际 id 差集派生（核实确认 id 命名一致、可行）
- file_changes/failed_paths 长期无消费者则考虑移除（当前保留向后兼容）

**P3（未决方向，需痛点驱动）**：
- contract-vision §五形态决策（①Pydantic 强化 / ②独立 SSOT）—— 未拍板，等真有字段漂移痛点再说
- 「gate_signals 丰富度优化」—— **本会话末尾提出但当场否决为「无痛点证据的伪需求候选」**。除非观察到 lead 实际决策出问题（该重派没重派/反问不到点），否则别做

**已知 debt（独立于本线）**：
- 4 个全量测试污染（见风险 §7），值得单独立项修测试隔离，不阻塞业务

---

## 6. 建议接手路径

**如果继续 handoff 方向**：先读 memory `feedback_task_context_framework_mismatches_ethoinsight_topology`（理解为什么 task_context 价值低）。**不要**重启「教下游/lead 消费 task_context」或「gate_signals 补明细」——这两个本会话都已证伪/否决。EthoInsight handoff 真正可能值得优化的是**契约化防字段漂移**（contract-vision 文档），但那要等真有漂移痛点。

**如果开新方向**：本线已干净收口，无依赖，可直接开。

**第一步永远先做**：`cd packages/agent/backend && make test`，确认基线（应 3717 passed + 4 已知污染 failed）。

---

## 7. 风险与注意事项（容易混淆/别重走的坑）

1. **别再「让下游/lead 消费 task_context」**——三轮真实数据核实已证伪（下游冗余、lead 需求被 gate_signals 覆盖）。这是本会话最贵的教训，memory 已钉。
2. **4 个全量测试 failed 是已知污染，别归因到自己的改动**（memory: `feedback_known_full_suite_test_pollution_4_tests`）：`deferred_tool_registry_promotion`×2 + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`。单独跑全绿、全量跑必红 = test isolation 污染。**坐实法**：父 commit 建 detached worktree 全量跑同样红。失败集精确=这 4 个则放行。
3. **读到「handoff 最佳实践/任务状态包」类文章想照做时，先核实拓扑 + 真实数据**，别照搬框架。本会话用户连续四次「等等」逼出核实，每次都推翻了一个框架假设。
4. **改 handoff 共享逻辑必须跑全量**（memory: `feedback_pr_merge_must_run_full_suite_on_shared_logic`）+ grep 所有读取方。
5. **接 review/实施自述必现场核实**（memory: `feedback_grill_handoff_must_be_verified`）——本会话每份「实施完成」自述都现场跑测试 + 读 commit 坐实，且一次因用错 diff 基线虚惊（worktree HEAD vs dev），核实后澄清。
6. **保留 `extra="allow"`**（handoff_schemas.py）——是 v0.1 兼容性命根子，收紧留给契约 sprint。

---

## 8. 下一位 Agent 的第一步建议

1. 读本文档 + memory `feedback_task_context_framework_mismatches_ethoinsight_topology`（2 分钟，避免重走 task_context 老路）
2. `cd packages/agent/backend && make test` 确认基线（3717 passed + 4 已知污染）
3. 确认用户的新意图——本 handoff 线已收口，下一步取决于用户开什么方向；若用户提「优化 gate_signals」或「让 X 消费 task_context」，先回到风险 §1/§7-3 与用户对齐「有没有真实痛点」再动手

---

## milestone 建议

本会话让「handoff 语义优化」track 到达一个 checkpoint（task_context 写入端落地 + 价值证伪收口）。建议更新/创建 milestone：
- **标题**：handoff task_context — 写入端落地 + 框架适配性结论
- **关键摘要**：3 commit 合 dev（Phase0 字段加固 / 写入端 task_context / 收尾修 bug+正名）；核心结论=「任务状态包」框架只适配 EthoInsight 一半（lead 中转 + 数据驱动 partial），task_context 当前价值低已留 TODO；handoff 真正优化方向待痛点驱动（契约化）。memory 3 条已沉淀。
