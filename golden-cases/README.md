# Golden Cases — 行为学专家标注指南

## 这是什么

Golden Case 是行为学专家为 EthoInsight agent 定义的"标准答案"。一份数据 + 专家认为正确的分析结论 = agent 的 benchmark。

## 您需要做什么

对于每个范式，我们已经为您创建了 case 目录，里面包含：

| 文件 | 您需要填写 | 难度 |
|------|----------|------|
| `raw-data/` | 放入 EthoVision XT 导出的原始数据文件（.txt / .xlsx / .csv） | 简单 |
| `metadata.yaml` | 填写范式名、物种、分组、实验条件等基本信息 | 简单（5 分钟） |
| `expected-analysis.yaml` | **核心**：对每个指标给出您认为合理的数值范围，以及对 agent 报告应该包含什么结论的断言 | 中等（20-30 分钟） |
| `notes.md` | 用自然语言写下您的推理过程——您看到数据后是怎么判断的 | 自由（15-20 分钟） |

**每个 case 总计约 40-60 分钟。**

## 6 个目标范式（v0.1）

| Case 编号 | 范式 | 目录 |
|-----------|------|------|
| case-001 | **EPM**（高架十字迷宫） | `case-001-epm-baseline/` |
| case-002 | **OFT**（旷场） | `case-002-oft-baseline/` |
| case-003 | **LDB**（明暗箱） | `case-003-ldb-baseline/` |
| case-004 | **FST**（强迫游泳） | `case-004-fst-baseline/` |
| case-005 | **Zero Maze**（零迷宫） | `case-005-zero-maze-baseline/` |
| case-006 | **TST**（悬尾实验） | `case-006-tst-baseline/` |

## 填写顺序建议

1. **先把数据放进 `raw-data/`**——从您手头的 EthoVision 导出文件里选一份典型数据
2. **填 `metadata.yaml`**——描述这份数据的基本信息（物种、分组、实验条件等）
3. **填 `expected-analysis.yaml`**——这是最核心的文件，下面有详细说明
4. **最后填 `notes.md`**——自由写下您的推理过程

## expected-analysis.yaml 填写说明

这个文件告诉 agent "对于这份数据，正确的分析应该是什么样的"。分三部分：

### 第一部分：expected_metrics（指标数值范围）

每个指标只需要填 `expected_range`（您认为合理的数值范围）。指标名已经帮您填好了。

```yaml
expected_metrics:
  - metric: open_arm_time_ratio
    expected_range: [0.05, 0.15]   # ← 您只需要填这个区间
    # treatment 组的开放臂时间比例应该在 5%-15% 之间
```

**不需要算精确值**，给一个合理的范围就行。比如"正常 C57 小鼠在 EPM 的开放臂时间比例大约 10-20%，给 diazepam 后可能升到 20-35%"。范围的宽度取决于您的信心——范围越窄，对 agent 的要求越严格。

### 第二部分：expected_findings（应该发现的结论）

这是专家判断的核心。分为 5 种类型：

| 类型 | 含义 | 示例 |
|------|------|------|
| `statistical_conclusion` | **必填至少 1 条**。组间比较的统计结论 | "treatment 组开放臂时间显著高于 control（p < 0.05）" |
| `phenotype_indication` | 表型判断 | "diazepam 表现出抗焦虑效应" |
| `confound_note` | 需要注意的混杂因素 | "需检查运动能力是否影响开放臂进入" |
| `data_quality_warning` | 数据质量问题 | "control 组 n=2 样本量不足" |
| `outlier_detection` | 离群个体 | "Subject 3 在所有指标上偏离群体 2 SD" |

每条 finding 有三个关键字段：
- `required_keywords`：agent 报告**必须包含**的关键词（至少 1 个词命中即得分）
- `forbidden_claims`：agent 报告**绝对不能出现**的错误说法
- `severity`：如果 agent 漏了这个发现，严重程度是 low / moderate / high / critical

### 第三部分：should_not_contain（禁区）

agent 输出中绝对不能出现的字符串。比如：
- 不存在的受试者编号
- 不可能的 p 值或效应量
- 与范式无关的结论

## 数据要求

- **至少 2 组**：control + treatment（或 2 个不同处理条件）
- **每组至少 3 只**动物（n ≥ 3 才能做统计检验）
- **典型数据**：选"正常"的分析场景，不要选太特殊的边界 case
- **格式**：EthoVision XT 导出的标准格式（.txt / .xlsx / .csv 均可）

## 完成标准

一个 case 合格的标志：
- [ ] raw-data/ 下有完整的 EthoVision 导出文件
- [ ] metadata.yaml 所有必填字段已填
- [ ] expected-analysis.yaml 中 at least 1 条 statistical_conclusion
- [ ] expected-analysis.yaml 中 at least 1 条 phenotype_indication
- [ ] notes.md 有 300 字以上的推理过程
- [ ] `python3 scripts/validate_golden_case.py <case-dir>` 通过

## 问题？

遇到不确定的地方，直接在 `notes.md` 里写下您的疑问，工程师会根据您的反馈调整模板。
