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
2. **不要硬猜**——如果信息不足，**用 ask_clarification 给结构化选项**让用户选；不要瞎填导致下游分析跑错路径。
3. **反问最多 1 次**——LoopDetectionMiddleware 会在重复反问时强制中断；如果范式无法唯一确定，必须用 ask_clarification 带证据反问，不允许默认猜测（见"Gate before guess"节）。

## 工作场景

每个 agent 按自己**当下在做什么**对照下表，决定是否 read 对应文件。SKILL.md 不规定具体哪个 agent 必须 read 什么——这交给 agent 自己根据任务上下文判断。

| 工作场景 | read 哪个文件 |
|---|---|
| 识别用户的实验 / 模板 / 变体 | `references/_facts.md` + `references/identification-decision-tree.md` + `references/by-template/<大类>.md` |
| 分析 / 解读 / 写报告 | `references/by-experiment/<范式>.md` |

**路径占位符**：

- `<范式>` = `experiment-context.json` 的 `paradigm` 字段（snake_case 英文，如 `epm`、`open_field`）。识别阶段未写入时，按用户消息中文意推断（如"高架十字迷宫" → `epm`）后 read `by-experiment/epm.md` 验证候选。
- `<大类>` = EV19 模板大类（`PlusMaze`、`OpenFieldRectangle` 等）。识别阶段从用户消息或文件名推断；如已确定 `ev19_template`（如 `PlusMaze-AllZones`），取 `-` 前的部分（`PlusMaze`）。
- 文件不存在时（同事尚未填写的范式 / 大类）：read 报错可接受，回到 `references/identification-decision-tree.md` 用 `_facts.md` 兜底，范式无法唯一确定时 ask_clarification 反问。

## 反问哲学:Gate before guess (spec §13.5)

**不允许**:在范式不明时默认选最像的范式继续。

**必须**:范式无法唯一确定时,`ask_clarification` 带证据反问。

反问模板见 `ethoinsight-lead-interaction` skill 的
`references/clarification-templates.md` 的 "范式多候选反问" /
"范式不明 + 用户也不知道" 两段。
