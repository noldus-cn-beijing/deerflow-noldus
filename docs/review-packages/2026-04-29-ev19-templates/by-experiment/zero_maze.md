# 实验：零迷宫 (Zero Maze)

**slug**：`zero_maze` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

## 🟡 一句话定义

利用啮齿类动物对开放高处的天然回避倾向，通过环形迷宫中开放区与封闭区的探索行为来评估焦虑样行为水平。

## 🟡 适用模板（按推荐顺序 + 取舍说明）

- `ZeroMaze-AllZones` — **推荐首选**。包含 Closed and Open arms zone，可直接计算开放区相关指标
- `ZeroMaze-NoZones` — 无预定义 zone，需实验员自行划定区域；不推荐常规使用

## 🟡 必须计算的指标

- 开放区时间百分比
- 开放区滞留时长
- 开放区移动距离
- 犹豫次数：从封闭区探头后缩回的次数，与焦虑倾向正相关
- 总移动距离：反映总体活动水平，用于运动混杂检查

## 🟡 常见脱险点 / 数据质量风险

- 总移动距离过低时，开放区指标的下降可能为运动抑制而非焦虑增加，需标注警告
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 犹豫次数的判定依赖实验员的主观标准，不同实验室之间可能不一致

## 🟡 报告解读语言

- 使用"开放区滞留时间百分比"和"开放区滞留时长"作为标准表述
- 解读方向：开放区指标降低提示焦虑样行为增加；需同时报告总移动距离以排除运动抑制
- 区分"数据支持的发现"（统计显著）和"趋势性变化"（未达显著但方向一致）

## 🟡 关键参考文献

- Shepherd JK, et al. (1994). "The elevated zero-maze: an ethological analysis of the effects of diazepam in the rat." *Psychopharmacology*.
- Kulkarni SK, et al. (2007). "Elevated zero maze: a paradigm to evaluate antianxiety effects of drugs." *Methods and Findings in Experimental and Clinical Pharmacology*.

## 🟡 与其他实验的区分

- 与 EPM 的区别：Zero Maze 为环形无中心区，动物持续处于开放或封闭区，消除了 EPM 中心区决策延迟的干扰；两者原理相同但结构不同
- 与 OFT 的区别：Zero Maze 依赖高度差异产生回避，OFT 依赖空间开阔度产生回避
