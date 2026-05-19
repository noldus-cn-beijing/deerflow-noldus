# EV19 大类：PorsoltCylinder

**中文名**（待补充）：强迫游泳圆筒（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **2** 个变体。

## 🟡 这个大类用来做什么？

- 主要研究对象：啮齿类动物（大鼠、小鼠）
- 典型实验类型：强迫游泳实验（FST）和悬尾实验（TST），评估抑郁样行为中的绝望表型
- 学术范式名：强迫游泳 / Forced Swim Test (FST)；悬尾实验 / Tail Suspension Test (TST)

## 🟡 何时不该选这个大类？

- 如果实验目的是焦虑评估，不应使用

## 🟡 关键参考文献

- Porsolt RD, et al. (1977). "Behavioural despair in rats: a new model sensitive to antidepressant treatments." *Nature*.
- Slattery DA, Cryan JF (2012). "Using the rat forced swim test to assess antidepressant-like activity in rodents." *Nature Protocols*.

---

## 变体：PorsoltCylinder-AllZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PorsoltCylinder-AllZones`
- **arena_template**：`Porsolt cylinder`
- **zone_template**：`Diving zone`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：True
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZones`

### 🟡 这个变体相对其他变体的核心差异

包含 Diving zone，用于识别动物是否力竭溺水以自动停止采集

### 🟡 推荐的实验场景

需要力竭溺水安全检测的 FST 实验

### 🟡 不该用这个变体的场景

标准 FST 仅需不动行为时应选 NoZones

### 🟡 对应学术范式

- 强迫游泳 / Forced Swim Test (FST)

---

## 变体：PorsoltCylinder-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PorsoltCylinder-NoZones`
- **arena_template**：`Porsolt cylinder`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：True
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZones`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，适合标准 FST 实验

### 🟡 推荐的实验场景

标准 FST 实验，仅需记录不动行为

### 🟡 不该用这个变体的场景

需分析潜水行为时应选 AllZones

### 🟡 对应学术范式

- 强迫游泳 / Forced Swim Test (FST)
- 悬尾实验 / Tail Suspension Test (TST)

---
