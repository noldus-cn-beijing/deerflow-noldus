# EV19 大类：OpenFieldCircle

**中文名**（待补充）：圆形旷场（待行为学同事确认中文）

> 本文档：行为学同事补充领域知识；机器解析的 arena/zone/subject 字段已经填好。
> 🟢 表示自动从 XML 抽取，**不要修改**。🟡 表示需要行为学同事补充。

该大类下共 **5** 个变体。

## 🟡 这个大类用来做什么？

- 主要研究对象：鱼类、昆虫、啮齿类动物均可
- 典型实验类型：圆形旷场实验；鱼类群体行为（shoaling）；水生动物开放场实验
- 学术范式名：旷场实验 / Open Field Test (OFT)；鱼群行为 / Shoaling

## 🟡 何时不该选这个大类？

- 标准啮齿动物 OFT 通常选 OpenFieldRectangle（方形是标准配置）

## 🟡 关键参考文献

- Prut L, Belzung C (2003). "The open field as a paradigm to measure the effects of drugs on anxiety-like behaviors: a review." *European Journal of Pharmacology*.
- Gould TD, et al. (2009). "Open field test." *Encyclopedia of Behavioral Neuroscience*.

---

## 变体：OpenFieldCircle-AllZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldCircle-AllZones`
- **arena_template**：`Open field, round`
- **zone_template**：`Border, center, quadrants`
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

包含 Border + Center + Quadrants zone，可直接计算中心区域相关指标

### 🟡 推荐的实验场景

需要圆形观察区的啮齿动物 OFT 实验

### 🟡 不该用这个变体的场景

鱼类应选 NoZones-Fish

### 🟡 对应学术范式

- 旷场实验 / Open Field Test (OFT)

---

## 变体：OpenFieldCircle-NoZones-Fish

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldCircle-NoZones-Fish`
- **arena_template**：`Open field, round`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：fish
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZones-Fish`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，指定适用于鱼类

### 🟡 推荐的实验场景

鱼类群体行为（shoaling）或水生动物开放场实验

### 🟡 不该用这个变体的场景

啮齿动物应选 AllZones

### 🟡 对应学术范式

- 鱼群行为 / Shoaling
- 水生动物开放场 / Aquatic Open Field

---

## 变体：OpenFieldCircle-NoZones-Insects

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldCircle-NoZones-Insects`
- **arena_template**：`Open field, round`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：False

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：insect
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZones-Insects`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，指定适用于昆虫

### 🟡 推荐的实验场景

昆虫的圆形旷场实验

### 🟡 不该用这个变体的场景

啮齿动物应选 AllZones

### 🟡 对应学术范式

- 取决于实验设计

---

## 变体：OpenFieldCircle-NoZones-Rodents-Other

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldCircle-NoZones-Rodents-Other`
- **arena_template**：`Open field, round`
- **zone_template**：`No zone template`
- **bypass_arena_grid**：False
- **bypass_subject_count_and_roles**：False
- **bypass_subject_features**：False
- **m_bOtherTypes**：True

### 🟢 从目录名推断（仅作提示，行为学同事可在下方说"以名称提示为准"或"修正"）

- **推测适用 subject**：rodent
- **推测 zone 配置**：NoZones
- **推测阵列规模**：Single
- **目录名尾缀**：`NoZones-Rodents-Other`

### 🟡 这个变体相对其他变体的核心差异

无预定义 zone，指定适用于啮齿类

### 🟡 推荐的实验场景

需要自定义 zone 的圆形旷场啮齿动物实验

### 🟡 不该用这个变体的场景

标准 OFT 应优先选 AllZones

### 🟡 对应学术范式

- 旷场实验 / Open Field Test (OFT)

---

## 变体：OpenFieldCircle-NovObjZones

### 🟢 EV19 机器字段（自动抽取，请勿修改）

- **目录名**：`OpenFieldCircle-NovObjZones`
- **arena_template**：`Open field, round`
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

圆形观察区的新物体识别实验（NOR）

### 🟡 不该用这个变体的场景

标准 OFT 无需新物体区域

### 🟡 对应学术范式

- 新物体识别 / Novel Object Recognition (NOR)

---
