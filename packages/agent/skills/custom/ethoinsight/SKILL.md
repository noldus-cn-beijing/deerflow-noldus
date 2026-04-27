---
name: ethoinsight
description: >
  Behavioral neuroscience data analysis guide for EthoVision XT exported data.
  Covers paradigm interpretation, confound checking, effect size assessment,
  and APA-format scientific reporting. Use when analyzing animal behavior data
  or writing analysis reports.
version: 1.0.0
author: noldus-insight
---

# EthoInsight — 行为学数据分析指南

## 核心方法论

行为学数据分析的核心是**组间对比**，不是绝对阈值。

1. 始终以对照组为基线，计算实验组的相对变化
2. 统计显著性（p < 0.05）必须配合效应量（Cohen's d > 0.5）才有实际意义
3. 多个范式的一致性方向比单一范式的强信号更可靠

## Workflow

### Step 1: 混杂因素排查

在解读行为指标前，先排除混杂因素。详见 `references/confound-checklist.md`。

### Step 2: 范式解读

根据实验范式（EPM、OFT、Shoaling 等）查阅指标含义和组间对比判读原则。详见 `references/paradigm-interpretation.md`。

### Step 3: 统计方法选择

根据数据类型、组数、设计类型选择合适的统计检验方法。详见 `references/statistics-decision-tree.md`。

### Step 4: 效应量评估

统计显著性必须配合效应量判断实际意义。详见 `references/effect-size-guide.md`。

### Step 5: 报告撰写

按报告结构骨架撰写对 Result 的洞察。详见 `references/report.md`。

## 知识库查询

使用 noldus-kb MCP tools 获取更多领域信息：
- `search_knowledge`: 搜索论文、手册、应用笔记
- `get_paradigm`: 查询范式详情（setup、指标、相关产品）
- `get_terminology`: 查询行为学术语定义和计算方法
- `list_products` / `list_paradigms`: 浏览可用范式和产品
