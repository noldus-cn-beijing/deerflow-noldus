# EV19 大类：PlusMaze

**中文名**（待补充）：高架十字迷宫（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **3** 个变体。

## 🟡 这个大类用来做什么？

- 主要研究对象：啮齿类动物（大鼠、小鼠）
- 典型实验类型：高架十字迷宫（EPM），评估焦虑样行为
- 学术范式名：高架十字迷宫 / Elevated Plus Maze (EPM)

## 🟡 何时不该选这个大类？

- 如果测试对象是鱼类或昆虫，不应选 PlusMaze，应选对应物种的模板
- 如果实验目的不是焦虑评估（如学习记忆、运动功能），不应使用此大类

## 🟡 关键参考文献

- Rodgers RJ, et al. (1997). "Behavioural profiles in the murine elevated plus maze." *Psychopharmacology*.
- Walf AA, Frye CA (2007). "The use of the elevated plus maze as an assay of anxiety-related behavior in rodents." *Nature Protocols*.

---

## 变体：PlusMaze-AllZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PlusMaze-AllZones`
- **arena_template**：`Elevated plus maze`
- **zone_template**：`Closed-, open arms, head dip zone`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：True
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZones`

### 🟡 这个变体相对其他变体的核心差异

额外包含 head dip zone，可分析探头行为

### 🟡 推荐的实验场景

需要分析探头行为（head dip）的 EPM 实验

### 🟡 不该用这个变体的场景

标准 EPM 实验无需探头行为分析时，应选 FewZones

### 🟡 对应学术范式

- epm（高架十字迷宫）

---

## 变体：PlusMaze-FewZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PlusMaze-FewZones`
- **arena_template**：`Elevated plus maze`
- **zone_template**：`Closed and open arms`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：FewZones
- **推测阵列规模**：Single
- **目录名尾缀**：`FewZones`

### 🟡 这个变体相对其他变体的核心差异

包含 Closed arms + Open arms zone，是最通用的 EPM 配置

### 🟡 推荐的实验场景

标准 EPM 实验，需计算开臂时间百分比、开臂进入百分比、总进臂次数

### 🟡 不该用这个变体的场景

需要分析探头行为（head dip）时应选 AllZones

### 🟡 对应学术范式

- epm（高架十字迷宫）

---

## 变体：PlusMaze-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PlusMaze-NoZones`
- **arena_template**：`Elevated plus maze`
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

需要自定义 zone 划分方案的特殊 EPM 实验

### 🟡 不该用这个变体的场景

标准 EPM 实验应优先选 FewZones 或 AllZones

### 🟡 对应学术范式

- epm（高架十字迷宫）

---
