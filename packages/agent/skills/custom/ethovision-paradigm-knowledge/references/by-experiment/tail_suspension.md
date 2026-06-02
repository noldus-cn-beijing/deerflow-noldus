# 实验：悬尾实验 (Tail Suspension Test (TST))

**slug**：`tail_suspension` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

## 🟡 一句话定义

利用啮齿类动物在不可逃避的悬吊环境中产生的不动行为，评估抑郁样行为中的绝望表型。

## 🟡 适用模板（按推荐顺序 + 取舍说明）

- `PorsoltCylinder-NoZones` — **推荐**。无预定义 zone，用于记录不动行为

> **⚠️ 待同事裁决（2026-06-01 标记）**：本文件模板写 `PorsoltCylinder-NoZones`，但 `ev19_facts.py` 映射为 `NoTemplate`。真实数据（tstHelperDemoVideo）经 `parse_header` 解析为单观察区单对象、无 zone，结构上更接近 NoTemplate。两者各有依据，属于领域归类问题，待行为学同事确认 TST 在 EthoVision XT 里标准配 arena 的方式后统一。当前工程层先按 `NoTemplate` 跑通。

## 🟡 必须计算的指标

- 累计不动时间
- 不动潜伏期
- 不动次数

## 🟡 常见脱险点 / 数据质量风险

- 累计不动时间增加需排除运动能力差异
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 尾部固定方式不当可能引起疼痛反应，干扰不动行为判定
- 悬尾实验一般采用 Activity / Activity State 作为测量参数，而强迫游泳一般采用 Mobility / Mobility State，两者不可混用

## 🟡 报告解读语言

- 使用"累计不动时间"和"不动潜伏期"作为标准表述
- 解读方向：累计不动时间增加提示抑郁样行为增加；不动潜伏期缩短提示快速放弃行为
- TST 与 FST 的解读各自独立，不可交叉引用

## 🟡 关键参考文献

- Steru L, et al. (1985). "The tail suspension test: a new method for screening antidepressants in mice." *Psychopharmacology*.
- Can A, et al. (2012). "The tail suspension test." *Journal of Visualized Experiments*.

## 🟡 与其他实验的区分

- 与 FST（强迫游泳）的区别：TST 将动物悬吊于空中，FST 将动物置于水中；两者均测量不动行为评估绝望表型，但物理刺激方式不同
- 与焦虑类范式的区别：TST 评估抑郁样行为（绝望），焦虑类范式评估焦虑样行为（回避）

## Pendulum 参数判据

TST 不动行为判定使用 pendulum 算法区分钟摆摆动与真实挣扎。完整参数表和算法流程见 [tail_suspension-pendulum-params.md](tail_suspension-pendulum-params.md)（来源：`docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md`，同事 tstYoyo 撰写）。
