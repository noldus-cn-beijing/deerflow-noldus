# 实验：旷场实验 (Open Field Test (OFT))

**slug**：`open_field` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

<!-- 起草备注（同事可删除）：经典啮齿焦虑/探索范式，方形 vs 圆形要看物种习惯 -->

## 🟡 一句话定义

利用啮齿类动物对开阔环境的天然回避倾向，通过量化中心区与周边区的探索行为来评估焦虑样行为水平和自主活动性。

## 🟡 EV19 模板识别

调 `identify_ev19_template` 工具获取候选模板，**不要自己列举候选**。工具会根据数据结构返回唯一或 ambiguous 结果。

## 🟡 必须计算的指标

- 中心区域时间百分比
- 中心区域距离百分比
- 进入中心区域次数
- 总运动距离：反映总体活动水平，用于运动混杂检查

## 🟡 常见脱险点 / 数据质量风险

- 总运动距离过低（< 1000 cm）时，中心区域指标的下降可能为运动抑制而非焦虑增加，需标注警告
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 中心区域定义不一致会导致结果不可比，需确认 zone 划分方案

## 🟡 报告解读语言

- 使用"中心区域时间百分比"和"中心区域距离百分比"作为标准表述，避免简称"中心时间""中心距离"
- 解读方向：中心区域指标降低提示焦虑样行为增加；需同时报告总运动距离以排除运动抑制
- 区分"数据支持的发现"（统计显著）和"趋势性变化"（未达显著但方向一致）

## 🟡 关键参考文献

- Prut L, Belzung C (2003). "The open field as a paradigm to measure the effects of drugs on anxiety-like behaviors: a review." *European Journal of Pharmacology*.
- Gould TD, et al. (2009). "Open field test." *Encyclopedia of Behavioral Neuroscience*.

## 🟡 与其他实验的区分

- 与 EPM、Zero Maze 的区别：OFT 不依赖高度差异，焦虑通过中心区回避体现；EPM/Zero Maze 通过开放高处回避体现
- OFT 的总运动距离同时反映自主活动性，可用于运动功能评估；EPM 的总进臂次数功能类似但范围更窄
- 圆形 vs 矩形旷场：圆形消除角落聚集效应，适合鱼类和昆虫；矩形是啮齿动物 OFT 的标准配置
