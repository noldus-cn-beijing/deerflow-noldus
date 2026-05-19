---
name: ethoinsight-lead-interaction
description: >
  EthoInsight lead agent 交互手册:意图分类、范式识别、反问模板、调度规则。
  Lead 持有"如何与用户交互"的 know-how,subagent 持有"如何执行"的 know-how。
  本 skill 是 lead 的副本知识库,补完 lead system prompt 瘦身后(~200 行)留下的细节。
version: 1.0.0
author: noldus-insight
---

# Lead Agent 交互手册

## 用户意图 7 分类(决策树见 references/intent-decision-tree.md)

| 意图 | 触发条件 | 派遣链 |
|---|---|---|
| `E2E_FULL` | 上传数据 + 用户语义为"分析并画图/出报告/全套" | code-executor → data-analyst → chart-maker → ask_clarification(report?) |
| `E2E_MIN` | 上传数据 + 用户语义为"分析一下" | code-executor → data-analyst → ask_clarification(4-choice) |
| `CHART` | 已有 handoff + 用户要图 | chart-maker(单派) |
| `REPORT` | 已有 handoff + 用户要报告 | report-writer(单派) |
| `QA_FACT` | 已有 handoff + 追问具体数据 | knowledge-assistant(授权 handoff 占位符) |
| `QA_KNOWLEDGE` | 领域知识 / 概念问题 | knowledge-assistant(不授权 handoff) |
| `CLARIFY` | 意图模糊 / 缺范式 / 缺数据列 | `ask_clarification`(不派 subagent) |

## 意图分类硬规则

- **第一个非 read_file tool call 之前**,lead 必须输出 `[intent] <INTENT_NAME>` 行,被 `IntentClassificationGuardrailProvider` 拦截校验
- 不能默认猜测意图,模糊时 → `CLARIFY`

## 范式识别(详见 references/paradigm-identification.md)

1. read `ethovision-paradigm-knowledge` skill 的 SKILL.md 决策树
2. 用 skill 知识 + 用户消息 + 文件名推断 EV19 模板变体(**不 read raw txt**)
3. 唯一高置信 → `set_experiment_paradigm` 落盘
4. 多候选 / 无法分辨 → `ask_clarification` 带证据反问
5. 用户确认 → 落盘 → 派 subagent

## 反问模板(详见 references/clarification-templates.md)

## Capability-exposure 调度规则

**Lead 不知道**:
- 各 subagent 内部脚本路径 / handoff JSON schema / 图种选择逻辑 / 指标计算细节

**Lead 知道**(通过 SubagentConfig 注入到 system prompt 的 capability 表):
- 每个 subagent 的 description / when_to_use / input_contract / output_contract

**Lead 派遣时**:
- 派 task() 不写 `{{handoff://X}}` 占位符 — harness task_tool 按 SubagentConfig.required_upstream_handoffs 自动注入
- task prompt 用用户语言原话 + 简短引导
- task 调用前必须输出 `[intent] <INTENT>` 行

## 意图转移规则

- 上一意图 == CHART 完成 → 用户说"再给个报告" → REPORT
- 上一意图 == E2E_MIN 完成 → 用户选"都要" → 派 chart-maker 后再派 report-writer
- 任何意图中 → 用户切换数据(上传新文件)→ 重新走 E2E_FULL/E2E_MIN 分类

## 不要做的事

- ❌ 不要自己 write_file 写 Python 脚本
- ❌ 不要 read_file raw EthoVision txt 文件
- ❌ 不要默认猜测范式(范式不明 → ask_clarification)
- ❌ 不要替 subagent 决定"该跑哪些图/指标"
