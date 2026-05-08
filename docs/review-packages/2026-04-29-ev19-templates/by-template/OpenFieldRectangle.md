# EV19 大类：OpenFieldRectangle

**中文名**（待补充）：矩形旷场（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **7** 个变体。

## 🟡 这个大类用来做什么？

- 主要研究对象：啮齿类动物（大鼠、小鼠）
- 典型实验类型：旷场实验（OFT），评估焦虑样行为和自主活动性；也支持新物体识别等
- 学术范式名：旷场实验 / Open Field Test (OFT)；新物体识别 / Novel Object Recognition (NOR)

## 🟡 何时不该选这个大类？

- 如果测试对象是鱼类，不应选 OpenFieldRectangle，应选 OpenFieldCircle 或 AquariumTrack3D
- 如果实验需要圆形竞技场消除角落效应，应选 OpenFieldCircle

## 🟡 关键参考文献

- Prut L, Belzung C (2003). "The open field as a paradigm to measure the effects of drugs on anxiety-like behaviors: a review." *European Journal of Pharmacology*.
- Gould TD, et al. (2009). "Open field test." *Encyclopedia of Behavioral Neuroscience*.

---

## 变体：OpenFieldRectangle-AllZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-AllZones`
- **arena_template**：`Open field, square`
- **zone_template**：`Center, border, corners`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZones`

### 🟡 这个变体相对其他变体的核心差异

包含 Center + Border + Corners zone，可直接计算中心区域相关指标

### 🟡 推荐的实验场景

标准啮齿动物 OFT 实验，需量化中心区域回避

### 🟡 不该用这个变体的场景

鱼类或昆虫应选对应物种的模板

### 🟡 对应学术范式

- 旷场实验 / Open Field Test (OFT)

---

## 变体：OpenFieldRectangle-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-NoZones`
- **arena_template**：`Open field, square`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
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

需要自定义 zone 划分方案的特殊 OFT 实验

### 🟡 不该用这个变体的场景

标准 OFT 实验应优先选 AllZones

### 🟡 对应学术范式

- 旷场实验 / Open Field Test (OFT)

---

## 变体：OpenFieldRectangle-NoZonesFishInsects

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-NoZonesFishInsects`
- **arena_template**：`Open field, square`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：fish, insect
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZonesFishInsects`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，指定适用于鱼类和昆虫

### 🟡 推荐的实验场景

鱼类或昆虫的矩形旷场实验

### 🟡 不该用这个变体的场景

啮齿动物应选 OpenFieldRectangle-AllZones 或 NoZones

### 🟡 对应学术范式

- 取决于实验设计（如 aquatic open field）

---

## 变体：OpenFieldRectangle-NovObjZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-NovObjZones`
- **arena_template**：`Open field, square`
- **zone_template**：`Novel object zones`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：True
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：NovObjZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NovObjZones`

### 🟡 这个变体相对其他变体的核心差异

包含 Novel object zones，预定义了新物体放置区域

### 🟡 推荐的实验场景

新物体识别实验（NOR）

### 🟡 不该用这个变体的场景

标准 OFT 无需新物体区域，应选 AllZones

### 🟡 对应学术范式

- 新物体识别 / Novel Object Recognition (NOR)

---

## 变体：OpenFieldRectangle-Subdivided2x2

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-Subdivided2x2`
- **arena_template**：`Open field, square`
- **zone_template**：`Subdivided arena, 2x2`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：Subdivided2x2
- **推测阵列规模**：Single
- **目录名尾缀**：`Subdivided2x2`

### 🟡 这个变体相对其他变体的核心差异

将观察区 2×2 等分（4 个子区域），适合分区实验

### 🟡 推荐的实验场景

需要将观察区分为多个区域的实验设计

### 🟡 不该用这个变体的场景

标准 OFT 应选 AllZones

### 🟡 对应学术范式

- 待确认

---

## 变体：OpenFieldRectangle-Subdivided3x3

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-Subdivided3x3`
- **arena_template**：`Open field, square`
- **zone_template**：`Subdivided arena, 3x3`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：Subdivided3x3
- **推测阵列规模**：Single
- **目录名尾缀**：`Subdivided3x3`

### 🟡 这个变体相对其他变体的核心差异

将观察区 3×3 等分（9 个子区域）

### 🟡 推荐的实验场景

需要更精细分区分析的实验

### 🟡 不该用这个变体的场景

标准 OFT 应选 AllZones

### 🟡 对应学术范式

- 待确认

---

## 变体：OpenFieldRectangle-Subdivided4x4

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldRectangle-Subdivided4x4`
- **arena_template**：`Open field, square`
- **zone_template**：`Subdivided arena, 4x4`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：Subdivided4x4
- **推测阵列规模**：Single
- **目录名尾缀**：`Subdivided4x4`

### 🟡 这个变体相对其他变体的核心差异

将观察区 4×4 等分（16 个子区域）

### 🟡 推荐的实验场景

需要更精细分区分析的实验

### 🟡 不该用这个变体的场景

标准 OFT 应选 AllZones

### 🟡 对应学术范式

- 待确认

---
