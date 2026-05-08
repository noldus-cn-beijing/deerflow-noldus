---
name: ethovision-paradigm-knowledge
description: >
  Progressive disclosure of EthoVision XT 19 template knowledge base covering
  20 behavioral paradigm categories with 62 variants. References are organized
  by template and experimental paradigm. Use when agents need detailed template
  information or paradigm specifications for behavioral analysis tasks.
version: 1.0.0
author: noldus-insight
license: MIT
---

# EthoVision XT 19 范式知识包

EthoVision XT 19 模板知识库，包含 20 个行为学范式分类及 62 个变体。参考资料按模板和实验范式组织。

## 文件组织

```
references/
├── by-template/       # 按模板分类（20个md文件）
│   └── [各模板详细说明]
└── by-experiment/     # 按实验范式分类（20个md文件）
    └── [各范式详细说明]
```

## 何时使用

当需要以下信息时使用此知识包：

- **模板选择** — 查询特定行为测量的适用模板
- **范式规格** — 了解实验范式的详细参数和配置
- **协议对应** — 将行为学协议与 EV19 模板对应
- **学科知识** — 理解特定行为范式的理论基础

## 访问模式

代理应按需读取 `references/` 目录下的 markdown 文件：

- 模板细节：从 `by-template/` 中选择
- 范式信息：从 `by-experiment/` 中选择
- 组织查询：使用文件名进行导航

## 补充说明

这是行为学领域专家创建的知识参考资料，经过同行评审。用于支持 EthoInsight 智能代理在行为学数据分析中的决策和参数选择。
