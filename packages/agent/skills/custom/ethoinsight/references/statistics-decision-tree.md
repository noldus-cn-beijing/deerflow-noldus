# 统计方法选择指南

## 决策流程

1. **数据类型?** 连续型 → 继续; 分类/等级 → 卡方/Fisher 检验
2. **几组比较?** 两组 → Step 3; 三组+ → Step 4
3. **两组比较:**
   - 配对 + 正态 → 配对 t 检验
   - 配对 + 非正态 → Wilcoxon 符号秩
   - 独立 + 正态 + 方差齐(Levene) → 独立 t 检验
   - 独立 + 正态 + 方差不齐 → Welch t 检验
   - 独立 + 非正态 → Mann-Whitney U
4. **三组+比较:**
   - 独立组 + 正态 + 方差齐 → One-way ANOVA → 事后比较
   - 独立组 + 不满足 → Kruskal-Wallis → Dunn's 事后
   - 重复测量 + 正态 + 球形 → RM-ANOVA
   - 重复测量 + 不满足 → Friedman 检验
5. **多因素设计** (处理 x 时间 x 基因型) → 多因素 ANOVA / 混合设计 ANOVA

## 事后比较选择

| 条件 | 方法 |
|------|------|
| 方差齐 + 样本量相等 | Tukey HSD |
| 方差不齐或样本量不等 | Games-Howell |
| 仅与对照组比较 | Dunnett's |
| 非参数 (KW 后) | Dunn's + Bonferroni 校正 |

## 范式 x 统计方法速查

| 范式 | 典型因变量 | 推荐方法 |
|------|-----------|---------|
| OFT/EPM/明暗箱 | 区域时间、移动距离 | 独立 t / ANOVA |
| MWM 训练曲线 | 逃避潜伏期（多天） | RM-ANOVA（同一动物多天测量） |
| MWM 探针测试 | 目标象限时间 | 独立 t / ANOVA |
| NOR 辨别指数 | DI | 单样本 t (vs 0) + 组间 ANOVA |
| 社交偏好 | 社交接触时间 | 配对 t（同一动物两 stimulus） |
| FST/TST | 不动时间 | 独立 t / ANOVA |
| CFC 冻结时间 | 冻结比例（多阶段） | RM-ANOVA |
| Shoaling | IID/NND/极性 | 独立 t / ANOVA |
| 转棒实验 | 落下潜伏期 | Cox 回归 / RM-ANOVA |
