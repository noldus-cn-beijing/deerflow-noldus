# 实验：高架十字迷宫 (Elevated Plus Maze (EPM))

**slug**：`epm` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

<!-- 起草备注（同事可删除）：焦虑金标范式之一 -->

## 🟡 一句话定义

利用啮齿类动物对开放高处的天然回避倾向，通过量化开放臂探索行为来评估焦虑样行为水平。

## 🟡 适用模板（按推荐顺序 + 取舍说明）

- `PlusMaze-FewZones` — **推荐首选**。包含 Closed arms + Open arms zone，覆盖开臂时间百分比、开臂进入百分比、总进臂次数等核心指标，是 EPM 最通用的配置
- `PlusMaze-AllZones` — 额外包含 head dip zone，仅当需要分析探头行为时选用
- `PlusMaze-NoZones` — 无预定义 zone，需实验员自行划定区域；不推荐常规使用

## 🟡 必须计算的指标

- 开臂时间百分比
- 开臂进入百分比
- 开臂进臂次数
- 开臂进臂时间
- 总进臂次数：反映总体活动水平，用于运动混杂检查

## 🟡 常见脱险点 / 数据质量风险

- 总进臂次数过低（< 8）时，开臂指标的下降可能为运动抑制而非焦虑增加，需标注警告
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 实验动物掉下迷宫的情况需排查，异常轨迹应排除

## 🟡 报告解读语言

- 使用"开臂滞留时间百分比"和"开臂进入百分比"作为标准表述，避免简称"开臂时间""开臂次数"
- 解读方向：开臂指标降低提示焦虑样行为增加；需同时报告总进臂次数以排除运动抑制
- 区分"数据支持的发现"（统计显著）和"趋势性变化"（未达显著但方向一致）

## 🟡 关键参考文献

- Rodgers RJ, et al. (1997). "Behavioural profiles in the murine elevated plus maze." *Psychopharmacology*.
- Walf AA, Frye CA (2007). "The use of the elevated plus maze as an assay of anxiety-related behavior in rodents." *Nature Protocols*.

## 🟡 与其他实验的区分

- 与 Zero Maze（O迷宫）的区别：EPM 为十字形，有中心区，动物需主动选择进入开放臂；Zero Maze 为环形无中心区，动物持续处于开放或封闭区。两者原理相同但结构不同，Zero Maze 消除了中心区的决策延迟问题
- EPM 不用于评估抑郁样行为（不动时间），如需评估绝望行为应选 FST 或 TST
