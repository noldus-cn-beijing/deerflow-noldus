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
2. bash `python -m ethoinsight.catalog.resolve --mode charts --paradigm <p> --user-intent "<原话>" ... --output plan_charts.json`
   - **`--raw-files-json` 必须指向一个 JSON 数组,里面是虚拟路径**(`/mnt/user-data/uploads/xxx.txt`),**不可写宿主机绝对路径**。
   - raw_files 路径的**单一真源是 `plan_metrics.json.inputs.raw_files`**：read 这个文件,把数组**原样**写进 raw_files.json。**不要**从 handoff_code_executor.json 抄(其 inputs 字段历史上有过宿主路径污染),**不要**用 `Path(...).resolve()` / `realpath` 转换。
   - resolve.py 会把这些路径原样写到 `plan_charts.json` 的 `input` 字段,后续脚本被 sandbox guardrail 拦宿主机路径。
3. read `plan_charts.json` → charts[] + charts_fallback_available[]
4. 决策(见 references/fallback-decision-tree.md)
5. for each entry in plan_charts.json.charts: bash 跑脚本
   - 用 `python -m <entry.script> <entry.args 拼接>` 形式调用
   - **不要自己拼 paradigm 前缀或追加 --paradigm 参数**——entry.script 和 entry.args 已经是 resolve 阶段按 catalog yaml `accepts_paradigm` 字段拼好的（PR-1 落地)
6. write `handoff_chart_maker.json`
7. `present_files(<生成的 png 列表>)`
8. 输出 `OK: <N> charts generated\n[gate_signals]\n...`

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
