---
name: ethoinsight-chart-maker
description: >
  chart-maker subagent 的执行工作流手册（how-to-execute）。
  包含：run_chart_plan 调用模板、fallback 决策树、handoff schema、用户意图模糊时的处理。
  画图执行经 run_chart_plan 工具（进程内并行、核磁盘落盘、确定性封存 handoff）——不在工具外手拼 bash python -m。
  图种知识（图种 → 适用场景，what-to-pick）见姐妹 skill `ethoinsight-charts`。
  Use when chart-maker receives a visualization task with handoff_code_executor.json.
version: 1.0.0
author: noldus-insight
---

# Chart-Maker 可视化决策手册

## 服务对象
`chart-maker` subagent。**lead 不读本 skill**。

## 前置阅读
执行前必读:`/mnt/skills/ethoinsight/references/execution-conventions.md`。

## 工作流
1. read `handoff_code_executor.json` → 拿 paradigm / n_per_group / n_groups / total_subjects
2. 调 `prep_chart_plan` 工具一步生成 plan_charts.json（取代 bash 拼 catalog.resolve）
   - **uploaded_files（必填）**: 原样取自 `plan_metrics.json.inputs.raw_files`（数组原样传入，**不要**从 handoff_code_executor.json 抄，**不要**用 `Path(...).resolve()` / `realpath`）。
   - **paradigm（必填）**: 原样取自 `handoff_code_executor.json.paradigm`。
   - **user_intent / total_subjects / n_per_group / n_groups（可选）**: 从 handoff + 派遣 prompt 原样传。
   - **chart_budget（P5，可选）**: 绘图预算总数，**默认省略 = 全画（不限资源，有多少画多少）**。aggregate 图（box/bar 组间对比）本就全画；per_subject 图（trajectory/heatmap 个体图）在全画模式下也逐个 subject 全部画。**只有 lead 在派遣 prompt 中明确给定了一个预算数字时才传它**（lead 只在用户主动表达「画几张就行/代表性/少画点/挑几个」时才给数字）；派遣 prompt 未给预算（默认情形）→ 省略 chart_budget = 全画。chart_budget 的值由 lead 决定，执行者只照搬，绝不自行揣测或塞默认数字。被预算截断的 per_subject 进 `charts_budget_remaining[]`。
   - **column_aliases / groups 不用你传**：工具内部自读 `experiment-context.json`（Gate 1 列语义对齐投影）+ `groups.json`（prep_metric_plan 落盘的分组）。这是取代「bash 手拼 `--column-aliases-file` / `--groups-json`」的确定性入口——session 级横切状态由工具自取，你无从遗漏（红线二正模式 1）。
   - 工具返回 `plan_summary`（chart_count / fallback_count / skipped_count / chart_ids / column_aliases_applied / groups_applied / budget_remaining_count / budget_remaining_ids），据此进入步骤 3 的决策树。**charts[] 顺序：传了 chart_budget 时按 output_mode 预算优先级筛过（aggregate 在前、per_subject 代表子集在后）；省略 chart_budget（默认全画）时为 catalog 声明顺序。两种情形都按数组顺序全部执行。**
3. read `plan_charts.json` → charts[] + charts_fallback_available[] + charts_budget_remaining[]
4. 决策(见 references/fallback-decision-tree.md)——决策只决定「该不该让这些图进 plan」，不决定「怎么画」（画图由 run_chart_plan 工具确定性执行）。
5. 调 `run_chart_plan` 一次画完全部图（产出+交付合一，对标 code-executor 的 run_metric_plan）。
   - 调 `run_chart_plan(plan_path="/mnt/user-data/workspace/plan_charts.json")`——工具内 ProcessPoolExecutor 进程内并行跑 charts[] 全部绘图脚本，逐个核 output png 真落盘（磁盘真相），自动封存 handoff_chart_maker.json（sealed_by="run_plan"）。**零 bash 画图、零 args 重拼、零 LLM 自报产物**。
   - 重跑子集（如用户追加某张图）传 `only_chart_ids=["box_open_arm"]`；遇失败想快速停传 `on_error="abort"`。默认全画 + continue。
   - **画图全走 run_chart_plan**——args 由工具透传自 plan_charts.json 的 entry.args（resolve 阶段已按 catalog 拼好完整 argv，含 `--parameters-json`），彻底消除 dogfood thread 339512dd 那种手拼漏 `--parameters-json` 致 bar 图失败、靠重试才救活的脆弱。
6. run_chart_plan 内部已 seal handoff（**不要再调 seal_chart_maker_handoff**）。run_chart_plan 自动透传 plan.charts_budget_remaining 进 handoff.remaining_charts（P5 降级指纹，红线一）。
7. 图表已由 run_chart_plan 自动登记并呈现给用户（前端画廊直接可见）。再调一次 `present_files(<run_chart_plan 落盘的 png 列表>)` 把同批图登记进消息通道——IM 渠道（飞书/Slack）据此把图作为附件推送；Web 端已由 run_chart_plan 呈现，此步为幂等补充（reducer 按 path 去重不会重复）。
8. 输出 `OK: charts written\n[gate_signals]\n...`（charts_generated / failed_charts / chart_files 直接引用 run_chart_plan 返回的 gate_signals）

## 失败硬规范(2026-06-24 改：执行确定性化，画图全走 run_chart_plan)

任何退出路径都由 run_chart_plan 工具落盘 handoff_chart_maker.json（status/failed_charts/remaining_charts 由工具据磁盘+脚本 stderr 机读构造）。

| 触发条件 | 处理 |
|---|---|
| run_chart_plan 返回 status=partial | 工具已落盘 handoff（含 failed_charts）；据返回的 failures 把 chart_id + reason（机读真相）据实汇报给 lead，present_files 已成功的 png，输出 `OK` |
| run_chart_plan 返回 status=failed | 工具已落盘 handoff（chart_files=[]）；把全部失败 chart 汇报给 lead，输出 `OK:` + [gate_signals] |
| run_chart_plan 返回 error_code（如 plan_missing / empty_plan） | 工具已落盘 failed handoff；按 message 处理（缺 plan 时先补 prep_chart_plan），向 lead 报错说明原因 |
| chart_budget 截断 per_subject 图 | run_chart_plan 自动从 plan.charts_budget_remaining 透传进 handoff.remaining_charts[]（红线一：预算挤掉产出要留痕），无需你手填 |

## 用户语义解析(细节见 references/user-intent-parsing.md)

| 用户原话特征 | 解析 |
|---|---|
| "再画几个图" / "再来几张" | 复数 + 模糊 → 选 fallback 全部 |
| "画 trajectory" / "轨迹图" | 单选明确 → 只跑 trajectory |
| "箱线图比较" | 组间比较 + 要 box_* (catalog charts) |
| 无明确语义 | 跑 charts[] 全部;无 charts 则跑 fallback 全部 |

## Fallback 决策树(细节见 references/fallback-decision-tree.md)

- charts != [] → 跑 charts[] (catalog 命中)
- charts == [] AND fallback != [] + user_intent 明确 → 选匹配子集
- charts == [] AND fallback != [] + user_intent 模糊 → 跑 fallback 全部
- charts == [] AND fallback == [] → 写 handoff status=failed

## 不允许的操作
- ❌ 不要 read 别的 subagent 的 handoff
- ❌ 不要尝试 catalog 之外的图 (C7:新 chart 必须通过 catalog YAML + 同 PR 加脚本)
- ❌ 不要把图保存到 `/mnt/user-data/workspace/` 之外的路径
