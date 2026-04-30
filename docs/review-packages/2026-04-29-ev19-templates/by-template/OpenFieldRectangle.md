# EV19 大类：OpenFieldRectangle

**中文名**（待补充）：矩形旷场（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **7** 个变体。

## 🟡 这个大类用来做什么？（待补充）

<!-- 例：「测试啮齿动物在新环境中的探索-焦虑行为」「测试鱼群同伴聚集程度」 -->

- 主要研究对象：
- 典型实验类型：
- 学术范式名（中英）：

## 🟡 何时不该选这个大类？（待补充）

<!-- 例：「如果测试对象是鱼，不应选 OpenFieldRectangle，应选 OpenFieldCircle 或 AquariumTrack3D」 -->

## 🟡 关键参考文献（待补充）

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

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

### 🟡 这个变体相对其他变体的核心差异（待补充）

<!-- 例：「比 -NoZones 多了 Center/Border/Corners 三个 zone 列，能直接计算中心区停留时间」 -->

### 🟡 推荐的实验场景（待补充）

<!-- 例：「啮齿动物焦虑测试，需要量化中心区回避指数」 -->

### 🟡 不该用这个变体的场景（待补充）

<!-- 例：「鱼类不应使用此变体（subject 类型不匹配），应选 OpenFieldCircle-NoZones-Fish」 -->

### 🟡 对应学术范式（待补充）

<!-- 一对一：填一个；一对多（ambiguous）：列多个并说明区分依据 -->

- 

---
