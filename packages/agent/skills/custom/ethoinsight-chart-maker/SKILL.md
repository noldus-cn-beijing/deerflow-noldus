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
   - **前置:catalog.resolve 的 `--columns-file` 是必填参数**,先 bash `python -m ethoinsight.parse.dump_headers --input "<raw_files[0]>" --output /mnt/user-data/workspace/columns.json` 产出 columns.json,再把 `--columns-file /mnt/user-data/workspace/columns.json` 传给 resolve。这是 catalog.resolve 自带的标准入口,直接调即可(sandbox 已放行)。
   - **`--raw-files-json` 必须指向一个 JSON 数组,里面是虚拟路径**(`/mnt/user-data/uploads/xxx.txt`),**不可写宿主机绝对路径**。
   - raw_files 路径的**单一真源是 `plan_metrics.json.inputs.raw_files`**: read 这个文件,数组**原样**写进 raw_files.json,**不要**从 handoff_code_executor.json 抄(其 inputs 字段历史上有过宿主路径污染),**不要**用 `Path(...).resolve()` / `realpath`。
   - **若 handoff_code_executor.json.inputs.groups 存在,把它整个 dict 写到 `groups.json` 并加 `--groups-json /mnt/user-data/workspace/groups.json` 参数**;catalog 中 `needs_groups: true` 的 aggregate chart(box/bar 等)依赖该文件做组间对比。
   - resolve.py 会把这些路径原样写到 `plan_charts.json` 的 `input` 字段,后续脚本被 sandbox guardrail 拦宿主机路径。
3. read `plan_charts.json` → charts[] + charts_fallback_available[]
4. 决策(见 references/fallback-decision-tree.md)
5. for each entry in plan_charts.json.charts: bash 跑脚本
   - 用 `python -m <entry.script> <entry.args 拼接>` 形式调用
   - **entry.args 永远是 `--inputs <inputs.json>` + 可能 `--groups <groups.json>` + `--output <png>` + 可能 `--paradigm`** —— resolve 已物化对应 JSON 文件到 workspace,**不要自己拼 `--input` 单文件形式**(脚本不再接受单文件 `--input` 之外的 fallback)
   - **不要自己拼 paradigm 前缀或追加 --paradigm 参数**——entry.script 和 entry.args 已经是 resolve 阶段按 catalog yaml `accepts_paradigm` 字段拼好的
6. write `handoff_chart_maker.json`(**任何退出路径都必须先写**,见下面"失败硬规范")
7. `present_files(<生成的 png 列表>)`
8. 输出 `OK: <N> charts generated\n[gate_signals]\n...`

## 失败硬规范(2026-05-26 加)

任何提前退出都必须先写 `handoff_chart_maker.json`,把已生成的图 / 失败原因 / 错误 trace 持久化,否则 lead 拿不到上下文会困惑。

| 触发条件 | 处理 |
|---|---|
| bash 预算还剩 ≤ 2 次但 charts 还没跑完 | **立刻**写 handoff_chart_maker.json(status=`partial`),记录 `succeeded_charts[]` / `remaining_charts[]` / `reason="bash budget exhausted"`,present_files 已成功的 png,再输出 `OK` |
| 某 chart 脚本连续失败 ≥ 2 次 | 记入 failed_charts[] 跳过下一个,**不要继续重试**(LLM 试错只会耗光预算) |
| catalog.resolve 调用本身失败 | 写 handoff_chart_maker.json(status=`failed`),chart_files=[],failed_charts=[{chart_id:"all", reason:"<resolve error>"}],输出 `OK:` + [gate_signals] |
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
