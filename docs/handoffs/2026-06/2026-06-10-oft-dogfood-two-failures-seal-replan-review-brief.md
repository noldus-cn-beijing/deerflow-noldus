# Review Brief（骨架版）— 两处 harness 故障：subagent 漏调封存工具 + 协调器不持久化已解析状态

> 日期：2026-06-10 ｜ 写给 reviewer 的 AI Agent ｜ 性质：诊断 + 拟议方案，**不含已落地改动**。
> 本文是 multi-agent harness 的纯工程诊断，只列**故障结论 + 修复动作 + `file:line` 定位**。完整叙述见备份 `.bak-rewrite-v1`。
> 主仓库 `/home/wangqiuyang/noldus-insight`，HEAD `1d4a9d25`；harness 根 `packages/agent/backend/packages/harness/deerflow/`。

---

## 故障 A — worker 漏调封存工具，且兜底机制把它排除在外

**结论**：worker `code-executor` 跑完、产出全部结果文件后，**未调 `seal_code_executor_handoff` 就终止**（"terminated without emitting"）；被重派后下游 worker `data-analyst` 同样漏调 `seal_data_analyst_handoff`。

**根因（PROMPT 冲突，已逐字核实）**：合进同一条 SystemMessage（`subagents/executor.py:856-876`）的两份指令打架——
- system_prompt 权威要求走 seal 工具、禁 write_file：`subagents/builtins/code_executor.py` step 6。
- 过时 skill 仍教 `write_file handoff`、全文不提 seal：`skills/custom/ethoinsight-code/SKILL.md:30`；模板 `skills/custom/ethoinsight-code/templates/output-contract.md:22`。
- 模型把"写好叙述 + 输出 `[gate_signals]`"误当完成（同 `feedback_subagent_seal_deadlock_is_prompt_not_budget`，本轮触发源是过时 skill）。
- 加重项：write_file 路线会被 `guardrails/script_invocation_only_provider.py`（`_HANDOFF_WRITE_FILE_DENY` 约 :222-237）拒，退化成"完全没产物"。

**为何补救失效 = 决定性证据**：
- seal-resume 唯一一次补轮：`subagents/executor.py:891-985`，`_SEAL_RESUME_MAX_ATTEMPTS=1`（:171）；补轮措辞 `executor.py:917` 误用 `key_findings`（那是 DataAnalystHandoff 字段，给 code-executor 错 schema）。⚠️ 是否真触发需解码 checkpoints/gateway.log 坐实。
- 确定性兜底显式排除二者：`executor.py:282-285` `_AUTO_SEALABLE = {report-writer, chart-maker}`；:276-278 注释"不能从文件重建 → 永不 auto-seal"；:307 非白名单 early-return False。
- `data-analyst` 的 prompt 已加固到顶（`subagents/builtins/data_analyst.py:153/187`）仍漏调 → **证明纯 prompt 不可靠，必须 harness 兜底**。

**拟议方案 A**：
- **A1（PROMPT，治本）**：删 skill 的 write_file 旧流（`SKILL.md:30` + `output-contract.md:22`），改 defer 到 seal 工具；`code_executor.py` system_prompt 把 seal 重构为唯一且最后动作。SSOT 红线：权威只在 system_prompt/工具，skill 不重写（`feedback_single_source_of_truth`）。
- **A2（HARNESS，关键）**：扩 `_attempt_auto_seal_from_artifacts`（:288-399）接入 code-executor——从 `m_*.json` + `groups.json` + `plan_metrics.json` 机械重建必填字段（`handoff_schemas.py:386` `CodeExecutorHandoff` 仅 `status`+`summary` 必填，余 default 空；内容校验 `executor.py:122-135` 只需三者之一非空）。data-analyst 做降级 auto-seal（`status="partial"`，⚠️ 语义边界待 reviewer 定）。
  - ⚠️ **被推翻的假设**：注释"认知产物不能从文件重建"对"结果按命名约定落盘"的流水线是错的——值就在命名 JSON 里。
  - 红线：①不加 turn 预算；②不给 seal-resume 加 tool_choice（产空 args）；③动 `executor.py` 顶层 import 必须裸跑 `python -c "import app.gateway"` + `make_lead_agent`（conftest mock 藏循环导入，`feedback_conftest_mock_hides_circular_import`，前例 `3ffaf672`）。
- **A3**：`executor.py:917` 的 `key_findings` 改通用措辞。⚠️ `:913` 有 in-code 守卫锁定该段——只去专有名词、不重构。
- **A4（不做）**：不改 `validate_catalog`（`validate_catalog.py:346` 只遍历 `plan['metrics']`，结构上不可能对 plan 外项报错，是 red herring）。

---

## 故障 B — 协调器不持久化已解析状态，每轮从头重规划

**结论**：协调器 `lead agent` 在输入对齐阶段经历三轮独立 ask_clarification，中间反复重调 `inspect_uploaded_file`（≈9 次），触发 loop-detection 告警。

**根因（HARNESS，已核实）**：持久化机制存在但闭环没闭——
- 状态文件缺字段：`agents/middlewares/experiment_context.py:385-404` 不写 groups/输入映射（grep 命中 0）。
- 已解析答案落盘太晚：`groups.json` 由链路最后一个工具 `tools/builtins/prep_metric_plan_tool.py:229-261` 才写。
- 已落盘字段下轮也不回灌：`agents/lead_agent/agent.py:530-533` 只取 `paradigm`+`gate_completed`，其余读出即丢；system_prompt 无 `<experiment_context>` 注入（grep 命中 0）。
- `inspect_uploaded_file` 输出瞬态、从不持久化。
- prompt 缺复用规则：`lead_agent/prompt.py:461-482` 没说"历史已答=既定事实，别重读重问"。

**loop-detection 是症状探测器且有害**（`agents/middlewares/loop_detection_middleware.py:171`，`warn_threshold=3`，`config/loop_detection_config.py:31-34`）——逼正在合法收集信息的协调器提前收尾。**不要把削弱它当主修**。

**拟议方案 B**：
- **B1（HARNESS，最高杠杆）**：`agent.py:530` `read_context` 后扩消费面，渲染 `<experiment_context resolved="true">` 块回灌 system_prompt（文件 `lead_agent/agent.py`、`lead_agent/prompt.py`）。
- **B2（HARNESS/schema）**：让已解析答案即时落盘——`set_experiment_paradigm` 接收持久化（仿 :399-404）或加轻量 `set_groups` 工具。SSOT 红线：`groups.json` 权威，状态文件副本只派生（`feedback_single_source_of_truth`）；⚠️ 派生方向 + 全部消费方待 reviewer 确认。
- **B3（PROMPT，便宜）**：加正向复用规则——"历史已答的澄清是既定事实，先读再行动，别重复反问/重读已知文件；收到答案立即落库再处理下一项"。

---

## 同构本质

正确持久化机制 harness 里**已存在**，但 prompt 指向**错误终结动作**，harness 只**事后检测**、不在**终结边界确保**。

| | 正确机制 | prompt 指向的错误终结动作 | 只事后检测的层 |
|---|---|---|---|
| seal | seal 工具 + 原子写 | write_file / 输出 gate_signals 叙述 | `_validate_handoff_emitted` |
| planning | 状态文件 + `groups.json` | clarify-first / 从历史重推导 | loop-detection |

**统一解法**：prompt 指向结构化机制为唯一终结动作、删矛盾旁路，并在终结边界加薄 harness 落实（非更多事后检测、非 turn-budget/tool_choice 补丁）。

---

## 红线清单（实施前必读）

1. 不加 turn 预算解决 seal；不给 seal-resume 加 tool_choice（产空 args）。
2. 动 `executor.py` 顶层 import → 裸跑 `python -c "import app.gateway"` + `make_lead_agent`（conftest mock 藏循环导入）。
3. 改共享 guardrail/middleware（loop_detection / experiment_context）→ 跑全量测试，不能只跑新测试（PR #66 前例）。
4. 已知 4 个污染测试（deferred_tool_registry ×2 + inspect_gate/paradigm async）预存在、与本改动无关。
5. SKILL.md / output-contract.md / code_executor.py system_prompt 三处须一致改并守 SSOT（合进同一 SystemMessage，只改一处留矛盾）。
6. `executor.py:913` in-code 守卫锁定补轮措辞——A3 只去专有名词、不重构。
7. groups 字段不得与 `groups.json` 双存（SSOT）。
8. 两修复均 HARNESS 层，独立于上游领域阻塞，可立即推进。

---

## 待坐实 Open Questions（遵 `grill_handoff_must_be_verified`）

1. seal-resume 本轮是否真触发？gate `executor.py:1193-1208`；surfaced message 在补轮前/后（:1228-1233）需解码 checkpoints/gateway.log。
2. `plan_metrics.json` 是否真含报错涉及的结果项？读 trace 可定，不改根因。
3. B2 给状态文件加 groups 是否制造 SSOT 分裂？先确认 `groups.json` 全部消费方再定派生方向。
4. A2 harness auto-seal 与 seal-resume 是否双封存/多跑一轮？确认复用 `_SEAL_RESUME_MAX_ATTEMPTS=1`、不与 :1193 补轮重复。
5. A2 data-analyst 降级 auto-seal 语义边界：机械兜底是否可接受？是否仅"重派 N 次后"触发并标注"待人工复核"？

---

## 给 reviewer 的请求

审核：(1) 根因判断是否成立（尤其"data-analyst 也失败 ⇒ 必须 harness 兜底"）；(2) 分层方案是否右层、是否漏红线；(3) 哪些 open question 必须写 spec 前坐实。**本文档不含已落地改动，可作评审基线。**
