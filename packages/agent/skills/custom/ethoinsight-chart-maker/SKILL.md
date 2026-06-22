---
name: ethoinsight-chart-maker
description: >
  chart-maker subagent 的执行工作流手册（how-to-execute）。
  包含：bash 调用模板、fallback 决策树、handoff schema、用户意图模糊时的处理。
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
4. 决策(见 references/fallback-decision-tree.md)
5. for each entry in plan_charts.json.charts（全部执行，已预算筛过）: bash 跑脚本
   - 用 `python -m <entry.script> <entry.args 拼接>` 形式调用
   - **`entry.args` 是一个完整的字符串数组，整体逐项拼接执行——保持每一项原样、顺序不变、一项不漏。** resolve 阶段已按 catalog 把这条命令的全部参数填好（`--inputs <inputs.json>` + 可选 `--groups <groups.json>` + `--output <png>` + 可选 `--paradigm` + **可选 `--parameters-json '{...}'`（zone 列对齐参数，FewZones 自定义分析区如 open/closed 的图靠它定位列，丢了脚本会算不出值、退出码 1、不出图）**）。你只需把 `entry.args` 数组拼成命令行，**不增不减不重构**。
   - **重跑某张图时，仍从 `plan_charts.json` 该 entry 的 `args` 取完整命令重拼**（不要凭记忆手敲一条"更简洁"的命令——那样最容易漏掉 `--parameters-json`，正是 dogfood thread 339512dd 里 bar 图第二批失败、靠第三次重试才救活的根因）。
   - entry.script 和 entry.args 已经是 resolve 阶段按 catalog yaml `accepts_paradigm` 字段拼好的——`--paradigm` 该不该带 resolve 已决定，照 `args` 执行即可。
6. write `handoff_chart_maker.json`(**任何退出路径都必须先写**,见下面"失败硬规范")。若 `charts_budget_remaining[]` 非空 → 透传进 handoff 的 `remaining_charts[]`（降级指纹，红线一）。
7. `present_files(<生成的 png 列表>)`
8. 输出 `OK: <N> charts generated\n[gate_signals]\n...`

## 失败硬规范(2026-05-26 加)

任何提前退出都必须先写 `handoff_chart_maker.json`,把已生成的图 / 失败原因 / 错误 trace 持久化,否则 lead 拿不到上下文会困惑。

| 触发条件 | 处理 |
|---|---|
| bash 预算还剩 ≤ 2 次但 charts 还没跑完 | **立刻**写 handoff_chart_maker.json(status=`partial`),记录 `succeeded_charts[]` / `remaining_charts[]` / `reason="bash budget exhausted"`,present_files 已成功的 png,再输出 `OK` |
| 某 chart 脚本连续失败 ≥ 2 次 | 记入 failed_charts[] 跳过下一个,**不要继续重试**(LLM 试错只会耗光预算) |
| catalog.resolve 调用本身失败 | 写 handoff_chart_maker.json(status=`failed`),chart_files=[],failed_charts=[{chart_id:"all", reason:"<prep_chart_plan error>"}],按返回 hint 处理(file_not_found/unknown_paradigm 等),输出 `OK:` + [gate_signals] |
| sandbox guardrail 拒绝(host_path_blocked 等) | **不要重试** —— guardrail 反馈消息已说明 root cause(路径错 / 参数错),把消息原样塞到 failed_charts[i].reason,跳过这个 chart |
| 任何未预期异常 | catch 之,写 handoff status=`failed` + errors=[<traceback summary>],输出 `OK: ` + [gate_signals],**不要让 subagent 静默挂掉** |

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
