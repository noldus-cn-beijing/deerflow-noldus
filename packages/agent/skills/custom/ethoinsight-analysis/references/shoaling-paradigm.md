# Shoaling 范式分析规则
本文档整合行为学专家规则，指导 agent 在斑马鱼群集行为（shoaling）范式的分析中应遵循的原则和判据。

## 离群检测判据

### 优先级顺序

1. **主判据：mean_nnd（Mean Nearest Neighbor Distance）**
   - 远高于群体均值 → 个体离开群体、距最近邻鱼更远
   - 数据来源：EthoVision XT JS Continuous 自定义变量导出，或 metrics.py 从原始 X/Y 坐标计算

2. **主判据：象限分布与停留模式**
   - 个体长期停留在单一象限，且该象限与群体其他成员的象限分离
   - 结合 polarity：若 polarity小于0.3 而群体均值大于0.6，表示该个体方向行为与群体不同步

3. **辅助判据：mean_iid（Mean Inter-Individual Distance）**
   - 用于描述群体内整体紧密度
   - 与 NND 联合评估：NND 高 + IID 升高 vs NND 高但 IID 不升高

4. **禁用判据：total distance_moved、velocity_mean**
   - 运动量低不等于离群（鱼可能静止但位置仍在群体中）

## IID/NND 的数据来源

来源 A：EthoVision XT JS Continuous 自定义变量
- 在 EthoVision 项目中通过 JS 脚本定义和计算
- 导出 raw data 时包含在数据表中

来源 B：metrics.py 从原始坐标计算
- 当 raw data 中有多个 subject 的同步 X/Y 坐标列时自行计算
- 计算结果可能与 EthoVision JS 存在浮点差异

## 离群个体的处理原则

不能主动建议排除。理由：离群与否需要生物学解释。

研究员决策流程：
1. 数值特征描述（Subject 3 的 mean_nnd=70mm，高于群体均值）
2. counterfactual 分析（若排除 Subject 3，组间 p 值变化）
3. 是否符合预设排除条件

**完全由研究员决定是否排除。论文中必须报告排除理由和数量。**

## 统计学判读哲学

- 不做常模/baseline 对比
- 以统计检验 + 效应量为判据
- Result 中陈述事实，Discussion 中解读含义

## 数据质量警告

- n_subjects 小于 3：统计检验力不足
- 同一 raw data 多次分析，指标完全一致：不是 artifact（输入相同是正常的）
