# 实验：强迫游泳 (Forced Swim Test (FST))

**slug**：`forced_swim` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

<!-- 起草备注（同事可删除）：抑郁/绝望经典范式 -->

## 🟡 一句话定义

利用啮齿类动物在不可逃避的强迫游泳环境中产生的不动行为，评估抑郁样行为中的绝望表型。

## 🟡 适用模板（按推荐顺序 + 取舍说明）

- `PorsoltCylinder-NoZones` — **推荐首选**。无预定义 zone，适合标准 FST 仅需记录不动行为的场景
- `PorsoltCylinder-AllZones` — 包含 Diving zone，用于识别动物是否力竭溺水以自动停止采集

## 🟡 必须计算的指标

- 累计不动时间：整个测试期间不动时间总和
- 不动潜伏期：首次出现不动行为的时间
- 不动次数

## 🟡 常见脱险点 / 数据质量风险

- 累计不动时间增加需排除游泳能力差异，动物可能因体力而非绝望导致不动
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 水温过高或过低会影响不动时间，需确保实验条件标准化

## 🟡 报告解读语言

- 使用"累计不动时间"和"不动潜伏期"作为标准表述
- 解读方向：累计不动时间增加提示抑郁样行为增加；不动潜伏期缩短提示快速放弃行为
- FST 与 TST 的解读各自独立，不可交叉引用

## 🟡 关键参考文献

- Porsolt RD, et al. (1977). "Behavioural despair in rats: a new model sensitive to antidepressant treatments." *Nature*.
- Slattery DA, Cryan JF (2012). "Using the rat forced swim test to assess antidepressant-like activity in rodents." *Nature Protocols*.

## 🟡 与其他实验的区分

- 与 TST（悬尾实验）的区别：FST 将动物置于水中，TST 将动物悬吊于空中；两者均测量不动行为评估绝望表型，但物理刺激方式不同，结果可相互验证但不可混用
- 与焦虑类范式（EPM/OFT/Zero Maze/LDB）的区别：FST 评估抑郁样行为（绝望），焦虑类范式评估焦虑样行为（回避）
