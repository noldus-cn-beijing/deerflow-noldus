# Golden-Case Schema 字段字典

> 本文档定义 EthoInsight golden-case 的结构规范。
> 行为学同事按此模板标注，工程师据此编写自动化校验和回归断言。
> 最后更新：2026-04-22

---

## 目录结构

每个 golden-case 是一个目录：

```
case-XXX-<paradigm>-<description>/
├── raw-data/                  # EthoVision XT 导出的原始轨迹文件
│   └── *.txt                  # 至少 1 份
├── expected-analysis.yaml     # ★ 核心：专家期望的分析结论（机器可读）
├── metadata.yaml              # case 背景（物种、范式、实验条件）
└── notes.md                   # 专家思维过程记录（半结构化）
```

---

## 1. metadata.yaml

case 的身份信息，用于过滤和分组。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `case_id` | string | 是 | 格式 `case-XXX-<paradigm>-<tag>`，如 `case-001-shoaling-baseline` |
| `paradigm` | enum | 是 | 见下方范式枚举表 |
| `species` | string | 是 | 如 `zebrafish`、`C57BL/6 mouse`、`Wistar rat` |
| `strain` | string | 否 | 品系，如 `AB strain`、`Sprague-Dawley` |
| `n_subjects` | int | 是 | 受试者数量 |
| `n_groups` | int | 是 | 分组数量 |
| `groups` | dict | 是 | 组名 → 受试者列表，如 `{control: [Subject 1, Subject 2]}` |
| `experimental_condition` | string | 是 | 实验条件简述（药物名称/剂量/手术/基因型等），若无实际处理填 `"演示数据，无实际实验处理"` |
| `source` | string | 是 | 数据来源，如 `"同事A的实验"` 或 `"demo-data/DemoData/斑马鱼鱼群行为"` |
| `raw_data_files` | int | 是 | raw-data/ 目录下文件数量 |
| `created` | date | 是 | YYYY-MM-DD |
| `annotator` | string | 是 | 标注人姓名 |
| `reviewer` | string | 否 | review 人姓名（如有） |
| `difficulty` | enum | 否 | `easy` / `moderate` / `hard`，标注 case 复杂度 |
| `tags` | list[string] | 否 | 自由标签，如 `[small_sample, outlier_present, confound_possible]` |

### 范式枚举

| 值 | 中文名 |
|---|---|
| `shoaling` | 斑马鱼鱼群行为 |
| `epm` | 高架十字迷宫 |
| `open_field` | 旷场实验 |
| `fst` | 强迫游泳实验 |
| `mwm` | 莫里斯水迷宫 |
| `y_maze` | Y 迷宫 |
| `o_maze` | O 迷宫 |
| `barnes` | 巴恩斯迷宫 |
| `nor` | 新物体识别 |
| `three_chamber` | 三箱社交测试 |
| `social_interaction` | 社会互动测试 |
| `light_dark` | 明暗箱 |
| `novel_suppressed_feeding` | 新奇抑制摄食 |
| `footprint` | 足迹分析 |
| `fine_behavior` | 动物精细行为识别 |
| `phenotyper` | PhenoTyper 硬件系统 |

---

## 2. expected-analysis.yaml

**核心文件** — 专家期望 agent 得出的分析结论。自动化测试会断言 agent 输出是否匹配。

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `paradigm` | enum | 是 | 与 metadata.yaml 一致 |
| `species` | string | 是 | 与 metadata.yaml 一致 |
| `n_subjects` | int | 是 | 与 metadata.yaml 一致 |
| `groups` | dict | 是 | 与 metadata.yaml 一致 |
| `expected_metrics` | list[MetricExpectation] | 否 | 期望的数值指标（允许区间） |
| `expected_findings` | list[FindingExpectation] | 是 | 期望的洞察结论（★ 重点） |
| `should_not_contain` | list[string] | 否 | agent 不应输出的内容（反例断言） |

### MetricExpectation（数值期望）

用于断言 Layer A 的计算结果是否正确。

| 字段 | 类型 | 必填 | 说明 | 示例 |
|---|---|---|---|---|
| `subject` | string | 是 | 受试者名 | `"Subject 3"` |
| `metric` | string | 是 | 指标名（与 ethoinsight metrics.py 对齐） | `"mean_nnd"` |
| `expected_range` | [float, float] | 是 | 允许区间 [min, max] | `[65, 75]` |
| `unit` | string | 是 | 单位 | `"mm"` |

### FindingExpectation（洞察期望）

用于断言 Layer B 的判断质量。**这是 golden-case 最重要的部分**。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | enum | 是 | 发现类型（见下方枚举） |
| `subject` | string | 条件必填 | 当 type 为 `outlier_detection` / `counterfactual_analysis` 时必填 |
| `metrics` | list[string] | 条件必填 | 当 type 为 `outlier_detection` 时必填，涉及哪些指标 |
| `claim` | string | 条件必填 | 当 type 为 `statistical_conclusion` / `counterfactual_analysis` / `confound_note` 时必填 |
| `reasoning` | string | 是 | 专家的判断理由（1-3 句） |
| `severity` | enum | 条件必填 | 当 type 为 `outlier_detection` 时必填：`low` / `moderate` / `high` / `critical` |
| `required_keywords` | list[string] | 否 | agent 输出中**必须包含**的关键词（字符串子串匹配） |
| `forbidden_claims` | list[string] | 否 | agent 输出中**不得包含**的表述（字符串子串匹配） |

#### FindingExpectation.type 枚举

| 值 | 含义 | 何时用 |
|---|---|---|
| `outlier_detection` | 识别到离群个体 | 某受试者指标显著偏离群体 |
| `counterfactual_analysis` | 排除某个体后的反事实统计 | 验证"组间差异是否由个体驱动" |
| `confound_note` | 混杂因素提示 | 样本量不均、体重差异、时辰差异等 |
| `statistical_conclusion` | 统计结论的措辞要求 | "差异不显著""效应量小"等核心表述 |
| `data_quality_warning` | 数据质量警告 | 跟踪丢失、样本量不足、缺失值 |
| `phenotype_indication` | 行为表型推断 | "探索型表型""焦虑样行为" |

### severity 枚举

| 值 | 含义 | 判断标准 |
|---|---|---|
| `low` | 轻微偏离 | 偏离 1-1.5 SD，对结论影响小 |
| `moderate` | 中度异常 | 偏离 1.5-2.5 SD 或拉偏组均值 |
| `high` | 显著异常 | 偏离 > 2.5 SD 或改变统计显著性 |
| `critical` | 致命问题 | 数据无效（跟踪丢失）或颠覆整个结论 |

### should_not_contain 列表

字符串列表。agent 最终输出中如果**包含任一条**的子串，该 case 判定失败。

用于捕获幻觉和过度推断：
- 不存在的受试者名（`"Subject 6"`）
- 不可能的 p 值（`"p < 0.001"`，当样本量只有 5 时）
- 不应出现的因果断言（`"药物显著改变"`，当数据不支持时）

---

## 3. notes.md

专家思维过程的自由记录。半结构化：固定大纲 + 自由段落。

### 必需大纲

```markdown
# Case-XXX 分析笔记

## 1. 数据背景
（实验条件、物种、动物数、分组含义、实验目的）

## 2. 初看数据的第一印象
（扫一眼指标表立刻注意到了什么？直觉判断）

## 3. 逐个指标的判断过程
### 3.1 <指标名>
（正常范围是什么？这份数据落在哪？为什么？）
### 3.2 <指标名>
...

## 4. 异常识别与辨别
（异常模式 A/B/C/D 里属于哪种？怎么排除其他解释？）
  - A: 个体表型变异（群体内自然行为型差异）
  - B: 混杂因素（非实验变量影响指标）
  - C: 统计离群（数据分布上的极端值）
  - D: 设备/采集故障（硬件或软件问题）

## 5. 最终结论
（一段话总结，对应 expected-analysis.yaml 的 statistical_conclusion）

## 6. 参考文献（可选）
```

### notes.md 的双重用途

1. **给人看**：行为学同事之间对齐分析思路
2. **给机器用**：每个段落可以自动切出来，作为 SFT 训练数据中 `(reasoning_input, expert_cot_output)` 对的来源。所以每个段落的**推理过程**比最终结论更重要。

---

## 4. raw-data/ 目录

- 存放 EthoVision XT 导出的 `.txt` 轨迹文件（或 `.xlsx` 统计数据）
- 文件名保留原始导出名，不做重命名
- 如果数据来自真实实验，脱敏处理（去掉实验员姓名等）

---

## 5. 标注流程

```
Step 1  工程师：建目录 + 拷贝 raw data + 填 metadata.yaml + 填 expected-analysis.yaml 的数值字段
Step 2  工程师：跑 validate_golden_case.py 确认结构合法
Step 3  行为学同事：review expected-analysis.yaml，补齐 reasoning/severity 字段
Step 4  行为学同事：写 notes.md（按大纲）
Step 5  工程师：跑校验 → 合入仓库 → 写自动化回归测试
```

---

## 6. 自动化断言规则

`scripts/validate_golden_case.py` 执行两层检查：

**Schema 校验（必须通过）**：
- metadata.yaml 和 expected-analysis.yaml 所有必填字段存在
- 枚举值合法
- YAML 语法正确

**内容校验（警告级别）**：
- `expected_metrics` 中 subject 在 groups 里有定义
- `required_keywords` 不为空列表（空列表 = 没有断言 = case 无意义）
- `notes.md` 包含所有 6 个大纲标题

---

## 7. 常见问题

**Q: expected_range 的区间应该多宽？**
A: 取决于指标的计算稳定性。纯算术指标（mean_nnd、distance_moved）±5% 即可。涉及随机采样的指标（如某些 bootstrap 结果）放宽到 ±10%。

**Q: required_keywords 用中文还是英文？**
A: 和用户语言一致。当前系统锁定中文输出，所以填中文关键词。

**Q: 一个 case 应该有多少条 expected_findings？**
A: 至少 1 条 `statistical_conclusion`。异常 case 通常 3-5 条。不需要面面俱到 — 测的是"关键结论 agent 有没有抓到"，不是"agent 说了多少话"。

**Q: notes.md 写多长？**
A: 300-800 字为宜。太短（< 100 字）推理过程丢失，太长（> 1500 字）增加标注负担。关键不是篇幅，是**判断的推理链**。
