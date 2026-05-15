# EV19 大类：PhenoTyper

**中文名**（待补充）：PhenoTyper 居家观察舱（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **15** 个变体。

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

## 变体：PhenoTyper-16x-AllZonesMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-16x-AllZonesMice`
- **arena_template**：`PhenoTyper, 16x`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：AllZones
- **推测阵列规模**：16x
- **目录名尾缀**：`16x-AllZonesMice`

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

## 变体：PhenoTyper-16x-AllZonesRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-16x-AllZonesRatOther`
- **arena_template**：`PhenoTyper, 16x`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：AllZones
- **推测阵列规模**：16x
- **目录名尾缀**：`16x-AllZonesRatOther`

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

## 变体：PhenoTyper-16x-FeedingShelterMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-16x-FeedingShelterMice`
- **arena_template**：`PhenoTyper, 16x`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：16x
- **目录名尾缀**：`16x-FeedingShelterMice`

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

## 变体：PhenoTyper-16x-FeedingShelterRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-16x-FeedingShelterRatOther`
- **arena_template**：`PhenoTyper, 16x`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：16x
- **目录名尾缀**：`16x-FeedingShelterRatOther`

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

## 变体：PhenoTyper-16x-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-16x-NoZones`
- **arena_template**：`PhenoTyper, 16x`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：NoZones
- **推测阵列规模**：16x
- **目录名尾缀**：`16x-NoZones`

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

## 变体：PhenoTyper-AllZonesMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-AllZonesMice`
- **arena_template**：`PhenoTyper`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZonesMice`

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

## 变体：PhenoTyper-AllZonesRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-AllZonesRatOther`
- **arena_template**：`PhenoTyper`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Single
- **目录名尾缀**：`AllZonesRatOther`

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

## 变体：PhenoTyper-FeedingShelterMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-FeedingShelterMice`
- **arena_template**：`PhenoTyper`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：Single
- **目录名尾缀**：`FeedingShelterMice`

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

## 变体：PhenoTyper-FeedingShelterRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-FeedingShelterRatOther`
- **arena_template**：`PhenoTyper`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：Single
- **目录名尾缀**：`FeedingShelterRatOther`

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

## 变体：PhenoTyper-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-NoZones`
- **arena_template**：`PhenoTyper`
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

## 变体：PhenoTyper-Quad-AllZonesMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-Quad-AllZonesMice`
- **arena_template**：`PhenoTyper, 4x`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Quad
- **目录名尾缀**：`Quad-AllZonesMice`

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

## 变体：PhenoTyper-Quad-AllZonesRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-Quad-AllZonesRatOther`
- **arena_template**：`PhenoTyper, 4x`
- **zone_template**：`Feeding, spot light and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：AllZones
- **推测阵列规模**：Quad
- **目录名尾缀**：`Quad-AllZonesRatOther`

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

## 变体：PhenoTyper-Quad-FeedingShelterMice

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-Quad-FeedingShelterMice`
- **arena_template**：`PhenoTyper, 4x`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：mice
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：Quad
- **目录名尾缀**：`Quad-FeedingShelterMice`

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

## 变体：PhenoTyper-Quad-FeedingShelterRatOther

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-Quad-FeedingShelterRatOther`
- **arena_template**：`PhenoTyper, 4x`
- **zone_template**：`Feeding and shelter zones`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rat_or_other
- **推测 zone 配置**：FeedingShelter
- **推测阵列规模**：Quad
- **目录名尾缀**：`Quad-FeedingShelterRatOther`

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

## 变体：PhenoTyper-Quad-NoZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`PhenoTyper-Quad-NoZones`
- **arena_template**：`PhenoTyper, 4x`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：True
- **bypass_subject_count_and_roles**：True
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：not specified by name
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Quad
- **目录名尾缀**：`Quad-NoZones`

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
