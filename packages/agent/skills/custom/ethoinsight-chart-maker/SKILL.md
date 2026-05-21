---
name: ethoinsight-chart-maker
description: >
  chart-maker subagent 的可视化决策手册。基于 catalog plan_charts.json
  和用户语义,决策跑哪些图脚本。
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
   - raw_files 路径直接从上游 `handoff_code_executor.json` 的 `inputs.raw_files` 字段(或 lead 派遣 prompt 给的虚拟路径)抄过来,**不要用 `Path(...).resolve()` 或 `realpath` 把它转成宿主机绝对路径**。
   - resolve.py 会把这些路径原样写到 `plan_charts.json` 的 `input` 字段,后续脚本被 sandbox guardrail 拦宿主机路径。
3. read `plan_charts.json` → charts[] + charts_fallback_available[]
4. 决策(见 references/fallback-decision-tree.md)
5. for each selected chart: bash 跑脚本
   - 通用图(`_common.plot_timeseries` / `_common.plot_trajectory`)必须追加 `--paradigm <p>`,让脚本按范式选默认 y_col。漏传会让 timeseries 退到 `distance_moved` 列;FST/LDB 等范式数据里没有这列,会出空图。
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
