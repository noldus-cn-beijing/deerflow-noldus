---
name: ethovision-paradigm-knowledge
description: >
  EthoVision XT 19 模板知识库（20 大类 / 62 变体）+ 学术实验范式领域知识。
  用于：(1) 识别用户的实验范式 + EV19 模板变体（如 PlusMaze-AllZones），
  通过对话识别 + ask_clarification 反问，把识别结果写入 experiment-context.json；
  (2) 在范式分析、统计解读、报告写作各阶段，按需 read 对应范式的领域知识
  （必算指标、判读哲学、报告解读语言）。
version: 0.1.0
author: noldus-insight
---

# EthoVision Paradigm Knowledge — EV19 模板识别 + 学术范式领域知识

## 何时使用此 skill

**必须使用**：用户提到任何 EthoVision 实验数据分析需求时（含上传 raw txt 文件 + 请求分析/统计/可视化/报告）。

**可跳过**：纯知识问答（无数据上传 + 概念性问题）；追问已有分析结果；闲聊。

## 核心原则

1. **EV19 模板 = 用户语言**（agent 与用户对话时使用），**学术范式 = 内部分析路径**（agent 调 set_experiment_paradigm 时填这个）。
2. **模板识别走 tool，不自己读文件**——lead agent 在调 `set_experiment_paradigm` 之前，**必须先调 `identify_ev19_template` 工具**。该工具内部完成所有文件读取和候选交叉排除，返回结构化结果。lead 不要自己 `read_file` 读取 `_facts.md` / `identification-decision-tree.md` / `by-template/*.md`，否则会消耗 LoopDetectionMiddleware 配额导致 FORCED STOP。
3. **不要硬猜**——如果 tool 返回 `status="ambiguous"`，用返回的 `clarification_question` 直接调 `ask_clarification`；不要自己编反问。
4. **反问最多 1 次**——LoopDetectionMiddleware 会在重复反问时强制中断。

## 工作场景

| 工作场景 | 做法 |
|---|---|
| 识别用户的实验 / 模板 / 变体 | **调 `identify_ev19_template` 工具**（1 次 tool call 完成）。不要自己 `read_file` 读取引用文件。 |
| 分析 / 解读 / 写报告 | `read_file` `references/by-experiment/<范式>.md`（领域知识供 data-analyst / report-writer 使用） |
| 回答"EV19 如何计算 X" | `read_file` `references/ev19-dependent-variables.md`（EV19 因变量权威公式，供 knowledge-assistant 引用） |

**路径占位符**：

- `<范式>` = `experiment-context.json` 的 `paradigm` 字段（snake_case 英文，如 `epm`、`open_field`）。
- `by-template/*.md` 和 `_facts.md` **不再需要 lead 手动 read**——`identify_ev19_template` 工具内部使用它们。

## 反问哲学:Gate before guess (spec §13.5)

**不允许**:在范式不明时默认选最像的范式继续。

**必须**:范式无法唯一确定时,`ask_clarification` 带证据反问。

反问模板见 `ethoinsight-lead-interaction` skill 的
`references/clarification-templates.md` 的 "范式多候选反问" /
"范式不明 + 用户也不知道" 两段。
