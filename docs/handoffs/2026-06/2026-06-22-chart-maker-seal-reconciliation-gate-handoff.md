# 交接：chart-maker 伪造失败原因 + 漏执行 plan 内 aggregate 图 —— 封存对账门实施

**日期**：2026-06-22
**分支**：`worktree-chart-maker-seal-gate`（基线 origin/dev b86f5f63）
**spec**：[docs/superpowers/specs/2026-06-22-chart-maker-fabricated-failure-and-execution-completeness-gate-spec.md](../../specs/2026-06-22-chart-maker-fabricated-failure-and-execution-completeness-gate-spec.md)

## 问题（spec 第 0 节，线上 thread `d58adb40` 实证）

chart-maker 用 28 只 EPM 数据跑分析，box/bar 图没出，`handoff_chart_maker.json` 却标 `status=completed`，`failed_charts` 写 `box_open_arm: "catalog.resolve skipped: missing columns in_zone_open_arms_*"`。

**这个 reason 是伪造的**：prod `plan_charts.json.skipped=[]`（resolver 什么都没跳过）、`box_open_arm` 的 args 完备可跑、outputs/ 只有 trajectory。本地对照组 `fb3ed752` 同数据跑通（outputs 有 box + bar）。纯粹是 subagent 行为差异——线上那次没把 box/bar 命令跑完，还抄了 6/16 旧列门 reason 当失败因。

## 根因（两条正交）

- **A**：`failed_charts[].reason` 是 LLM free-text，seal 工具原样透传，无任何对 `plan_charts.json.skipped[]` 的机读对账 → 可伪造。
- **B**：seal 只校验 `completed + chart_files 空`（trajectory 在，非空），看不见「plan 要画的 aggregate 图漏了」这个缺口 → 漏执行可标 completed。

## 实施（单一注入点）

改 `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`：

新增 `_reconcile_chart_maker_payload(payload, workspace)`，在 `_seal_handoff_to_workspace` 内 **仅当 `model_cls is ChartMakerHandoff`** 时调用——**一条注入点覆盖两条路径**（`seal_chart_maker_handoff` 工具 + `executor._attempt_auto_seal_from_artifacts` 兜底），spec 第 4 节风险边界要求的。

- **2.1（订正伪造 reason）**：读 `plan_charts.json.skipped[]` 建 `skipped_by_id`。对每条 `failed_charts`：
  - chart_id 在 skipped → 用 plan 的真实 detail 覆盖 LLM 自述；
  - chart_id 不在 skipped（本次 `box_open_arm` 正是此情形）→ 规整为机读形态 `"resolved in plan but not rendered before seal (likely max_turns/early-exit); chart-maker note: <LLM 原文>"`，LLM 原文仅作引用，权威 reason 是机读事实。打 warning。
- **2.2（堵漏执行）**：`status==completed` 时，`planned_aggregate`（plan 内 `output_mode=="aggregate"` 的图 basename 集）− `rendered`（outputs/ 实际 `*.png`）= missing。非空 → 抛 `ValueError`（消息含缺失 basename 列表），让 chart-maker 收到 error ToolMessage 重试补画。与既有 `_completed_requires_core_output` 同风格。
- **per_subject 豁免**：只对账 aggregate；被 `chart_budget` 截断的 per_subject 图不进此门（避免与 P5 预算规则打架）。
- **鲁棒性**：`plan_charts.json` 缺失/不可解析 → 跳过对账、原样封存 + warning（不 crash）。

**核实而非猜（spec 第 4 节纪律）**：dump 了真实 `plan_charts.json`（fb3ed752 控制组），确认图表条目字段是 `{id, output, output_mode, confidence, ...}`，`output_mode ∈ {aggregate, per_subject}`，`confidence` 在该样本恒为 `optional`。故 **aggregate 判据用 `output_mode=="aggregate"`**（spec 注释提到的 `confidence=="must_have"` 在实测样本里不成立）。

**auto-seal 交互（spec 第 4 节）**：兜底路径构造 `status=completed` 也过对账门；bug 场景下（plan 有 aggregate、outputs 缺）门抛 ValueError → 被 auto-seal 的 ROBUSTNESS try/except 捕获 → 返回 False（诚实 FAILED），而非兜底成"画了一半的 completed"。回归测试 `TestAutoSealGateInteraction` 钉死此行为。

配套：`subagents/builtins/chart_maker.py` 的 `<handoff_field_format>` 段加正面指令（deepseek 正面提示，memory 第 6 条）——「reason 只填当次执行脚本实际看到的 stderr 摘要；封存工具会用 plan_charts.json 的真实 skip 信息订正 reason」。

## 测试（红→绿，全绿）

`tests/test_chart_maker_seal_reconciliation.py`（10 个）：
1. `test_fabricated_missing_columns_reason_is_corrected` — 锁死本次 bug（reason 订正为机读形态）。
2. `test_real_skipped_reason_is_passed_through` — plan 真实 skip 的 reason 透传。
3. `test_completed_with_missing_aggregate_raises` — 2.2 响亮拒绝，消息含 `plot_box_open_arm.png`。
4. `test_partial_with_missing_aggregate_is_allowed` — partial 不被门拦。
5. `test_completed_passes` — fb3ed752 控制组形态（aggregate 全落盘）正常 completed。
6. `test_missing_per_subject_does_not_block` — per_subject 截断豁免。
7. `test_no_plan_charts_seals_anyway` / `test_no_plan_charts_keeps_failed_reason_as_is` — 无 plan 鲁棒。
8. `test_auto_seal_returns_false_when_aggregate_missing` / `test_auto_seal_succeeds_when_no_plan` — auto-seal × 门交互。

## 验证

- 新测试 10/10 绿；受影响既有套件（chart-maker schema/auto-seal/seal tools/handoff content validation/seal-resume/seal-gate middleware/subagent prompt）196/196 绿。
- 裸导入两生产入口：`import app.gateway` + `from deerflow.agents import make_lead_agent` 均 OK（无导入环，`test_gateway_import_no_cycle.py` 绿）。
- 全量 4205 passed；3 failed（`test_local_sandbox_provider_mounts.py`）经 git stash 对照确认为**基线既有污染**（sandbox mount 测试隔离问题，与本次改动无关），非新引入回归。
- ruff：改动文件 clean（chart_maker.py 的 4 个 E501 是基线既有，非本次引入）。

## 待用户

分支 `worktree-chart-maker-seal-gate` 已 push，待开 PR 到 dev。

## milestone 建议

本次终结「box/bar 反复出不来且每次错因都不一样」这个元问题（spec 第 5 节）——错因每次不同是因为信源（chart-maker 自述 reason）不可信；换成机读对账（plan_charts.json + outputs/）后错因唯一且真实。建议在 `docs/milestone/` 的 harness 鲁棒性 track 记一条（如无对应 track 可新建「seal 对账门」条目），关键摘要：**封存层引入机读真相对账，堵 chart-maker 伪造 reason + 漏执行 aggregate 图两类哑故障；单一注入点覆盖 seal 工具 + auto-seal 兜底**。
