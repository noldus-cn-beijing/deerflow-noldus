# 报告结构指南

## 报告定位

report-writer 输出的**不是**期刊投稿论文，不套 APA 句式。

读者：研究员的导师 / 教授 / 学术监督者。
场景：导师打开报告，用 5-10 分钟判断实验做了什么、结论是什么、下一步怎么走。
属性：严肃、结构化、可信、可追溯。

## 6 段骨架

### 开头：一句话摘要
以 blockquote 呈现，示例：
> 本次分析了 X 条 [物种] 的 [范式] 行为，比较 [A 组] vs [B 组]，主要发现是 [核心结论]。样本量限制下结论为描述性。

### 1. 实验概况
- 范式、受试个体、分组、处理描述、数据来源
- **处理描述**字段：从 planning skill 追问到的 group_semantics 读取；用户未提供则诚实写"用户未提供具体处理描述"，不编造

### 2. 分析方法
- 计算指标清单
- 统计方法（t-test / Mann-Whitney U / ANOVA 等）
- 方法选择依据（method_warnings）
- 多重比较校正方式

### 3. 结果（仅事实，不含解读）
3.1 描述性统计（M ± SD 表格）
3.2 组间比较（U/t/F、p、Cohen's d 列表）
3.3 个体层面观察（只陈述数值偏离事实，不做行为学判断）
3.4 图表（引用 handoff.chart_paths，markdown 图片语法）

### 4. 观察与洞察（行为学解读）
整合 data-analyst 的 key_findings，自然段落陈述。必须覆盖：
- 核心发现
- 统计功效评估
- 关于离群个体：只建议"单独标注并检查生物学依据"，不主动建议"排除"
- 群体指标数据来源说明（若为兜底计算）

### 5. 数据质量与局限
单独章节，非脚注。列出所有 data_quality_warnings。

### 6. 下一步建议
措辞克制，用"可考虑的方向"而非"应该做"。

### 尾注
追溯信息（生成日期、session 标识）

## 统计量呈现：禁止 APA 句式

❌ APA 句式："The treatment group showed significantly higher IID (M = 45.2, SD = 12.3) compared to controls (M = 32.1, SD = 15.7), t(10) = 2.34, p = .031, d = 0.85."

✅ 直接列数值：
- Treatment mean_iid: 45.2 ± 12.3 mm
- Control mean_iid: 32.1 ± 15.7 mm
- Mann-Whitney U = X, p = X.XX, Cohen's d = 0.85

理由：导师看的是事实，不是论文腔。统计符号国际通用不需翻译，但句式不套论文模板。

## Result / Discussion 严格分离

§3 结果段**只含数值和统计量**。任何解读、原因推测、行为学意义陈述都必须放 §4。

- §3 示例 ✅：`Subject 3 的 mean_nnd (70.02 mm) 高于同组其他个体 (36.09-39.86 mm)`
- §3 示例 ❌：`Subject 3 的 mean_nnd 异常偏高，可能反映焦虑样行为`（这是解读，放 §4）

## 文献引用

当前阶段不强制引用。noldus-kb 接入后，§4 可在自然段落中插入"参考 Author et al., Year"格式。不要编造引用。

## 未来扩展

- PDF 输出（plugin，未来）
- APA 格式生成（plugin，配合 noldus-kb 文献数据库，未来）
