---
name: ethoinsight-analysis
description: >
  Behavioral data analysis execution guide for code-executor subagent.
  Defines the analysis workflow using run_paradigm_analysis tool,
  statistical method selection rules, data quality checks, and fallback procedures.
  Use when executing behavioral data analysis tasks.
version: 1.0.0
author: noldus-insight
---

# EthoInsight 数据分析执行指南

主力工具是 `run_paradigm_analysis`，一次调用完成全部流程：数据解析 → 指标计算 → 统计检验 → 图表生成 → handoff 输出。

## Workflow

### Step 1: 提取参数

从 lead agent 的任务描述中提取 paradigm、file_pattern、groups 三个必填参数。工具参数详见 `references/run-paradigm-analysis-api.md`。

### Step 2: 调用 run_paradigm_analysis

将提取的参数传入 run_paradigm_analysis 工具。该工具自动处理统计方法选择、正态性检验、效应量计算等决策。

### Step 3: 检查结果质量

检查返回结果中的质量信号（方差为零、样本量不足等）。详见 `references/data-quality-checks.md`。

### Step 4: 返回结果

按输出契约格式返回。详见 `templates/output-contract.md`。

### Fallback

仅当 run_paradigm_analysis 返回不支持该范式时，使用旧流程。详见 `references/fallback-workflow.md`。
