# 实验：明暗箱 (Light-Dark Box)

**slug**：`light_dark_box` （这是 paradigm key，agent 内部用）

> 行为学同事补充。机器侧 ev19_template 字段在 by-template/ 里维护。

<!-- 起草备注（同事可删除）：EV19 没有专门的 LightDark 模板？同事确认是不是用 Subdivided 或自定义 -->

## 🟡 一句话定义

利用啮齿类动物对明亮环境的天然回避倾向，通过量化明暗两区之间的探索行为来评估焦虑样行为水平。

## 🟡 适用模板（按推荐顺序 + 取舍说明）

- `NoTemplate` — **推荐**。需实验员手动配置竞技场，画出观察区、明区和暗区；若暗区带有遮光顶板，还需绘制隐藏分析区。LDB 无专用 EV19 模板。

## 🟡 必须计算的指标

- 明箱时间百分比
- 穿梭次数：反映探索与回避平衡
- 潜伏期：首次进入明箱的时间

## 🟡 常见脱险点 / 数据质量风险

- 穿梭次数过低（< 4）时，明箱时间百分比的下降可能为探索动机不足而非焦虑增加，需标注警告
- 每组样本量少于 5 只时统计功效不足，结论需谨慎
- 明暗区光照强度差异不足时，动物可能无法区分两区，导致数据失效

## 🟡 报告解读语言

- 使用"明箱滞留时间百分比"作为标准表述，避免简称"明箱时间"
- 解读方向：明箱滞留时间百分比降低提示焦虑样行为增加；需同时报告穿梭次数以排除探索动机不足
- 区分"数据支持的发现"（统计显著）和"趋势性变化"（未达显著但方向一致）

## 🟡 关键参考文献

- Hascoët M, Bourin M (1998). "A new approach to the light/dark test procedure in mice." *Pharmacology Biochemistry and Behavior*.
- Bourin M, Hascoët M (2003). "The mouse light/dark box test." *European Journal of Pharmacology*.

## 🟡 与其他实验的区分

- 与 EPM、Zero Maze 的区别：LDB 不依赖高度差异，通过光照强度差异产生回避；EPM/Zero Maze 通过开放高处回避
- 与 OFT 的区别：LDB 有明确的物理隔断（明暗两区），OFT 为单一开放空间
