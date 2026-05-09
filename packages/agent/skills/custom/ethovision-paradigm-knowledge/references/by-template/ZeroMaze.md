# EV19 大类：ZeroMaze

**中文名**（待补充）：零迷宫（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **2** 个变体。

## 🟡 这个大类用来做什么？

- 主要研究对象：啮齿类动物（大鼠、小鼠）
- 典型实验类型：O迷宫（Zero Maze），评估焦虑样行为
- 学术范式名：O迷宫 / Zero Maze (Elevated Zero Maze)

## 🟡 何时不该选这个大类？

- 如果测试对象是鱼类或昆虫，不应使用
- 焦虑评估可选 EPM 替代

## 🟡 关键参考文献

- Shepherd JK, et al. (1994). "The elevated zero-maze: an ethological analysis of the effects of diazepam in the rat." *Psychopharmacology*.
- Kulkarni SK, et al. (2007). "Elevated zero maze: a paradigm to evaluate antianxiety effects of drugs." *Methods and Findings in Experimental and Clinical Pharmacology*.

---

## 变体：ZeroMaze-AllZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`ZeroMaze-AllZones`
- **arena_template**：`O-maze`
- **zone_template**：`Closed and open arms`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZones`

### 🟡 这个变体相对其他变体的核心差异

包含 Closed and Open arms zone，可直接计算开放区相关指标

### 🟡 推荐的实验场景

标准 O迷宫实验

### 🟡 不该用这个变体的场景

无需 zone 分析时应选 NoZones

### 🟡 对应学术范式

- O迷宫 / Zero Maze

---

## 变体：ZeroMaze-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`ZeroMaze-NoZones`
- **arena_template**：`O-maze`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZones`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，需实验员自行划定区域

### 🟡 推荐的实验场景

需自定义 zone 的 O迷宫实验

### 🟡 不该用这个变体的场景

标准实验应优先选 AllZones

### 🟡 对应学术范式

- O迷宫 / Zero Maze

---
