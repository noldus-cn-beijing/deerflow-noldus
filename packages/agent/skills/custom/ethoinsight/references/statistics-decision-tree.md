# 统计方法选择指南

## 决策流程

1. **数据类型?** 连续型 → 继续; 分类/等级 → 卡方/Fisher 检验
2. **几组比较?** 两组 → Step 3; 三组+ → Step 4
3. **两组比较:**
   - 配对 + 正态 → 配对 t 检验
   - 配对 + 非正态 → Wilcoxon 符号秩
   - 独立 + 正态 + 方差齐(Levene) → 独立 t 检验
   - 独立 + 正态 + 方差不齐 → Welch t 检验
   - 独立 + 非正态 → 尝试 log/sqrt 变换；若恢复正态 → 回到参数路径；仍非正态 → Mann-Whitney U
4. **三组+比较:**
   - 独立组 + 正态 + 方差齐 → One-way ANOVA → 事后比较
   - 独立组 + 不满足 → Kruskal-Wallis → Dunn's 事后
   - 重复测量 + 正态 + 球形 → RM-ANOVA
   - 重复测量 + 不满足 → Friedman 检验
5. **多因素设计** (处理 x 时间 x 基因型) → 多因素 ANOVA / 混合设计 ANOVA
6. **协变量存在**（基线差异需校正）→ ANCOVA

## 数据变换指南

当 Shapiro-Wilk 检验不通过时，在放弃参数方法前先尝试变换：

| 变换 | 适用场景 | 示例 |
|------|---------|------|
| log(x) | 右偏分布（移动距离、反应时间） | `distance_moved` 常呈 log-normal |
| √x | 计数数据（进入次数、不动次数） | `center_entry_count` |
| logit(p) | 比例数据（0-1 之间） | `open_arm_time_ratio` |

变换后重新做 Shapiro-Wilk：若 p > 0.05，走参数路径并标注"经 log/sqrt 变换后满足正态假设"。

## 事后比较选择

| 条件 | 方法 |
|------|------|
| 方差齐 + 样本量相等 | Tukey HSD |
| 方差不齐或样本量不等 | Games-Howell |
| 仅与对照组比较 | Dunnett's |
| 非参数 (KW 后) | Dunn's + Bonferroni 校正 |

## 多重比较校正选择

同时检验多个指标时必须校正 α，但 Bonferroni 过于保守：

| 校正方法 | 适用场景 | 特点 |
|----------|---------|------|
| Bonferroni | 验证性分析、注册假设 | 最保守，控制 family-wise error rate |
| Holm-Bonferroni | 验证性分析 | 比 Bonferroni 更强但不那么保守 |
| BH-FDR (Benjamini-Hochberg) | 探索性分析、多指标筛选 | 控制假发现率，行为学近年标准做法 |

**选择规则**：用户预先指定假设 → Holm；同时报告 5+ 指标做筛选 → BH-FDR。

## 统计功效评估

报告统计结果时，必须评估结果的可信度：

- **显著结果**：报告效应量大小（Cohen's d / η²），帮助用户判断生物学意义
- **不显著结果**：报告 post-hoc power 或置信区间宽度，帮助用户判断是"确实没效应"还是"样本量不够"
- **建议**：当 power < 0.5 且 p > 0.05 时，在报告中注明"阴性结果可能因统计功效不足，不能排除存在效应"

## 范式 x 统计方法速查

| 范式 | 典型因变量 | 推荐方法 |
|------|-----------|---------|
| OFT/EPM/明暗箱 | 区域时间、移动距离 | 独立 t / ANOVA |
| OFT/EPM (多因素) | 基因型 × 处理 × 时间 | Two-way ANOVA / Mixed ANOVA |
| MWM 训练曲线 | 逃避潜伏期（多天） | RM-ANOVA（同一动物多天测量） |
| MWM 探针测试 | 目标象限时间 | 独立 t / ANOVA |
| NOR 辨别指数 | DI | 单样本 t (vs 0) + 组间 ANOVA |
| 社交偏好 | 社交接触时间 | 配对 t（同一动物两 stimulus） |
| FST/TST | 不动时间 | 独立 t / ANOVA |
| FST (时间分段) | 前 2 min vs 后 4 min 不动时间 | 配对 t / RM-ANOVA |
| CFC 冻结时间 | 冻结比例（多阶段） | RM-ANOVA |
| Shoaling | IID/NND/极性 | 独立 t / ANOVA |
| 转棒实验 | 落下潜伏期 | Cox 回归 / RM-ANOVA |
