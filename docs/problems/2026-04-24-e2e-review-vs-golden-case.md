# E2E 测试结果 vs Golden Case 交叉分析

**日期**: 2026-04-24
**作者**: Claude (Opus 4.7, 1M context)
**分析对象**:
- E2E 测试结果: `docs/e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md`
- Golden Case: `golden-cases/case-001-shoaling-baseline/` (notes.md + expected-analysis.yaml + metadata.yaml)
- 交接文档: `docs/handoffs/2026-04-24-p3-test-fix-handoff.md`

**目的**: 把 agent 端到端分析的实际输出，和行为学同事在 golden case 里标注的 ground truth 做对齐检查，找出系统性差距并提出修复建议。请其他模型交叉评审本文档的结论与建议。

---

## 1. 背景与数据

### 1.1 E2E 测试场景

研究员（模拟用户）上传 5 条斑马鱼的 EthoVision XT 导出数据，告知 agent "沿用相同的实验分组"（Control = Subject 1, 2；Treatment = Subject 3, 4, 5），请 agent 做 shoaling 群体行为分析。

Agent 完整跑完 `planning → code-executor → data-analyst` 流水线，产出：

- 数值结果：Control (n=2) vs Treatment (n=3)，distance_moved 和 mean_nnd 均不显著（p=1.000, 0.800），Cohen's d = 0.54–0.65
- 专家解读：Subject 3 驱动组间差异；n=2 时 MWU 最小双尾 p=0.2 数学上不可能显著；IID/Polarity 与上轮完全一致疑似 artifact
- 数据质量警告 3 条（control n=2 / Subject 3 离群 / IID 值复现）

### 1.2 Golden Case 标注要点（行为学同事给出的 ground truth）

来自 `case-001-shoaling-baseline/notes.md` 中带 `ANSWER:` 的段落：

1. **分组无生物学意义**：case-001 的 Control/Treatment 是演示分组，"5 条斑马鱼并未进行实际分组"。
2. **真实实验里，分组含义必须问用户**："一般来说 control 和 treatment 可能是病理/毒理/转基因造模，年龄，剂量等不同，需要在分析前由用户提供信息。没有提供的话**需要对话核实**"。
3. **离群阈值不能用总运动距离**："离群阈值不应该用总运动距离判断"——shoaling 范式里 IID、NND、象限分布等才是离群判据，distance_moved 不是。
4. **不主动建议排除个体**："一般只有造模失败/检测任务学习失败的时候，才会在进入下一个试验阶段时排除掉个体，且**排除数量和原因需要在论文中进行报告**"。
5. **不使用常模/baseline**："由于行为学实验受动物类型、品系、造模类型及造模程度、实验时间等诸多因素影响，**一般不给定范式 baseline 或常模**。直接以同项目实验组间对比"。
6. **Result / Discussion 分离**："文章的 Result 和 Discussion 是分开的。Result 仅汇报结果和统计检验信息，以及相关补充。解读是放在 Discussion 中的"。
7. **IID/NND 在 EthoVision 里是用户配置的 JS Continuous 自定义变量**，notes.md 附了完整 JavaScript 源码。导出的 raw data 文件**不一定包含**这些群体计算的结果列——需要研究员在 EthoVision 项目里启用对应脚本。

---

## 2. 差距分析（agent 实际行为 vs golden case ground truth）

### 2.1 🔴 分组语义：agent 盲目接受 Control/Treatment 裸 label

**E2E 实际**：agent 收到"沿用相同的实验分组"后直接进入分析，没有追问"Treatment 对应什么处理"。

**Ground truth**：行为学同事明确要求"没有提供的话需要对话核实"。

**后果**：
- 所有后续解读失去抓手——agent 不知道 Treatment 是药物、年龄、品系还是什么，Discussion 段只能写通用套话
- 在 case-001 这种"演示分组、无真实处理"的场景下，agent 输出的 "Treatment 组 mean_nnd 偏高主要由 Subject 3 驱动" 虽然数值描述正确，但**把它当成组间比较本身就是误导**——没有处理就没有"组间效应"可谈
- golden case 的 `expected-analysis.yaml` 第 15 行 `experimental_condition` 字段其实就是给这个信息留的位置，但 agent 不知道有这一层

**这个问题的严重程度**：高。它不是一个 prompt 措辞问题，是 planning 流程的必经环节缺失。

### 2.2 🔴 离群值处理：agent 主动建议"排除 Subject 3"

**E2E 实际**：data-analyst 输出"**建议检查该鱼健康状态，并在后续分析中将其作为离群值单独报告或排除**"。

**Ground truth**："如果是真实实验，**需要保留**。一般只有造模失败/检测任务学习失败的时候，才会在进入下一个试验阶段时排除掉个体"。

**后果**：
- 如果研究员听从 agent 建议排除 Subject 3，treatment 组 n 从 3 降到 2，等于放弃该组
- 更深层：离群值处理是**生物学判断**（是否造模失败 / 任务学习失败），不是**统计判断**（数值偏离多少 σ）。agent 越过了自己的能力边界
- golden case 的 `forbidden_claims` 目前没列这一条，所以回归测试也 catch 不到

**这个问题的严重程度**：高。直接影响研究员的数据处理决策。

### 2.3 🔴 离群判据：agent 用 distance_moved 当离群证据

**E2E 实际**：agent 在 "Subject 3 是所有组间差异的唯一来源" 的论述里，把 distance_moved (12,518 mm ≈ 其他一半) 作为离群的主要证据。

**Ground truth**："离群阈值**不应该用总运动距离判断**"。shoaling 范式的离群判据是 IID、NND、象限分布。

**后果**：
- 在 shoaling 范式里，运动距离低**本身不是离群**——可能只是这条鱼探索策略不同，群体聚集得更紧
- 真正的 shoaling 离群是 NND 偏高（Subject 3 的 70 mm vs 其他 36-40 mm 这个才是合法证据）
- agent 的论述里**把两个证据混在一起**，逻辑上降低了 NND 偏高这条硬证据的可信度

**这个问题的严重程度**：中。agent 最终结论方向正确（Subject 3 确实偏离），但论证链不纯正。

### 2.4 🟠 IID/Polarity artifact：方向对，根因错

**E2E 实际**：agent 识别出 mean_iid (86.38 mm) 和 mean_polarity (0.528) **跨实验完全一致**，建议研究员"**检查 EthoVision 的多鱼同时追踪模式是否开启**"。

**Ground truth**（notes.md 第 55-320 行展开）：
- IID 在 EthoVision 里是 **JS Continuous 自定义变量**，研究员要在项目里手动加脚本
- NND 同样是自定义 JS 变量
- 导出 raw data 里**可能不包含这些列**——需要研究员配置

**后果**：
- agent 的建议方向对（确实是配置问题），但把责任归到"多鱼追踪模式"——EthoVision 里没有这个开关，研究员会找不到
- 真正的根因是 **`packages/ethoinsight/ethoinsight/metrics.py` 在没有多鱼同步坐标时，算出了几何 placeholder 数值当 IID 输出**
- 正确做法应该是 metrics 层**检测到输入不满足多鱼同步条件时返回 NaN**，并在 data quality warning 里说明"IID/Polarity 需要在 EthoVision 项目里启用对应的 JS Continuous 自定义变量并导出"

**这个问题的严重程度**：中。agent 表面现象识别对了，但没给出可执行的修复路径，研究员收到后会困惑。

### 2.5 🟠 Result / Discussion 未分离

**E2E 实际**：agent 的最终呈现把"Subject 3 驱动差异（数值事实）"和"建议检查健康状态（解读+建议）"混在一段里。

**Ground truth**：APA 论文 Result 和 Discussion 必须分开。

**后果**：
- 研究员复制 agent 输出到论文时需要自己拆分
- `report-writer` 如果未来生成 APA 格式报告，当前的信息组织方式需要重新解构
- 这是 `data-analyst` prompt 里的输出结构问题

**这个问题的严重程度**：低（不影响正确性，影响可用性）。

### 2.6 ✅ agent 做对的部分（值得保持）

- **n=2 时 MWU 最小双尾 p=0.2** 这一观察非常扎实——把"不显著≠无差异"用数学 backup 而不是套话
- **3 个 critical 级别的数据质量警告都触发了**——质量门控工作正常
- **组间比较，不用绝对阈值** 这条 CLAUDE.md 铁律在 e2e 里没被违反
- **planning → code-executor → data-analyst 的信息流**经 handoff_code_executor.json 传递，即使 /mnt/shared 写失败也没中断（fallback 工作）

---

## 3. 修复建议（按优先级排序）

### 🔴 P1 — Planning skill 加强制追问：分组裸 label 必须核实

**修改位置**: `packages/agent/backend/skills/custom/ethoinsight-planning/SKILL.md`（或对应 reference 文件）

**加一条硬规则**：

> 收到分组信息后，若 group name 为 `control/treatment/对照/实验/ctrl/trt` 等通用 label 且用户未同时提供处理描述（药物、剂量、造模类型、基因型、年龄等），必须调用 `ask_clarification` 追问一次：
>
> "请问 [Treatment/实验组] 对应的具体处理是什么？例如药物剂量、造模类型、基因型、年龄差异等。这会影响 Discussion 段的解读方向。"
>
> 用户回答后，把回答内容记录到后续 code-executor 和 data-analyst 的 handoff context 里。

**验证**：在 case-001 的 expected findings 里加一条 `type: clarification_required`，要求 agent 至少问过一次处理描述。

### 🔴 P1 — data-analyst prompt 加两条硬约束

**修改位置**: `packages/agent/backend/src/deerflow/agents/data_analyst/prompt.py`（或对应 skill reference）

**加约束 A — 离群值处理**：

> 发现可能的离群个体时，输出措辞必须是 "**建议单独标注该个体，并检查是否存在造模失败 / 任务学习失败 / 设备故障等排除依据，最终是否剔除由研究员判断**"。
>
> 禁止的措辞：
> - "建议排除"（不加条件）
> - "将其作为离群值排除"
> - "剔除后重新分析"
>
> 理由：是否排除个体是生物学判断，超出 agent 能力边界。论文中排除个体必须报告原因和数量。

**加约束 B — Result / Discussion 结构分离**：

> 最终呈现必须分两段：
>
> ### Result
> 仅包含：数值、统计检验、p 值、效应量、样本量、数据质量警告。不含解读。
>
> ### Discussion
> 包含：组间差异的可能原因（需用户提供处理描述后才展开）、离群个体的行为学诠释、对后续实验设计的建议。

### 🔴 P1 — shoaling 范式离群判据固化到 skill

**修改位置**: `packages/agent/backend/skills/custom/ethoinsight-analysis/`（shoaling 范式部分）

**加一条 reference**：

> shoaling 范式的离群判据优先级：
> 1. **主判据**：mean_nnd（远高于群体均值 → 远离群体）
> 2. **主判据**：象限分布（长期停留单一象限且远离其他 subject）
> 3. **辅助判据**：mean_iid（需确认 EthoVision JS 变量已启用）
> 4. **禁用判据**：total distance_moved、velocity_mean——运动量低不等于离群，可能只是探索策略差异
>
> 论述离群时，只引用主判据作为"定位 + 偏离方向"的证据，辅助判据可补充但不独立成立离群结论。

### 🟠 P2 — metrics.py 修复 IID/Polarity 的根因

**修改位置**: `packages/ethoinsight/ethoinsight/metrics.py`

**改动**：

1. 在计算 IID/Polarity 前，检测输入数据是否满足"多 subject 同时间戳同步坐标"条件
2. 如果不满足：对应列输出 `NaN`，**不要输出看起来合理但实际是几何 placeholder 的数值**
3. 在 `data_quality_warnings` 里 append `critical` 级别警告：
   > IID/Polarity 需要 EthoVision 项目启用多鱼同步坐标导出（通过 JS Continuous 自定义变量，见 golden-cases/case-001-shoaling-baseline/notes.md 第 55-320 行的脚本示例）。当前数据不满足此条件，群体层面指标不可用于生物学结论。

**验证**：用 case-001 的 raw data 跑 metrics，确认 IID/Polarity 列是 NaN 而不是 86.38。

### 🟠 P2 — golden case 补充 forbidden_claims

**修改位置**: `golden-cases/case-001-shoaling-baseline/expected-analysis.yaml`

当前第 97-103 行的 `should_not_contain` 加：

```yaml
should_not_contain:
  # 既有
  - "Subject 6"
  - "Subject 7"
  - "p < 0.01"
  - "显著差异"
  - "药物显著"
  - "处理效应显著"
  # 新增（对应本次 e2e 发现的问题）
  - "建议排除 Subject"          # 不主动建议排除个体
  - "排除后重新分析"             # 同上
  - "基于 IID"                  # 多鱼追踪未确认前不引 IID 结论
  - "基于 Polarity"             # 同上
  - "正常范围"                  # 不用常模/baseline
  - "典型值"                    # 同上
```

并在 `expected_findings` 里加一条：

```yaml
  - type: clarification_required
    claim: "agent 应在接受 Control/Treatment 分组后，追问一次具体处理描述（药物/造模/年龄/基因型）"
    reasoning: "行为学同事明确要求：裸 control/treatment label 没有解读价值，必须核实"
    required_keywords: ["处理", "药物", "造模"]  # 至少匹配一个
```

### 🟢 P3 — 把 notes.md 的 ANSWER 段提炼为通用行为学规则

**修改位置**: 新建 `packages/agent/backend/skills/custom/ethoinsight-analysis/references/interpretation-rules.md`

**内容**：从 case-001 notes.md 提取跨 case 通用的行为学判读规则：

1. 只做组间对比，不用常模/baseline
2. 离群个体默认保留，排除需生物学依据且论文报告
3. shoaling 范式离群判据优先级（如 P1 第 3 条）
4. Result / Discussion 必须分离
5. 分组的生物学含义必须来自用户，不能假设

**为什么放这里**：这些规则跨 case-001/002/003... 全部适用，每个 golden case 的 notes.md 单独重复是浪费；同时 data-analyst 每次启动 load skill 就能看到。

---

## 4. 我可能判错的地方 / 需要交叉验证的点

以下几点我自己不确定，希望交叉评审时重点质疑：

1. **P1 的"追问处理描述"会不会太噪**？每次都问一遍用户会不会烦？是否应该做成"session 级 memory——用户回答过一次后，本 session 内不再问"？
2. **distance_moved 作为离群判据**这件事，我是完全相信了行为学同事的话。但万一他的意思是"**单独用** distance_moved 不够"，而不是"**完全不能用**"？notes.md 第 55 行原话是"**不应该**用总运动距离判断"，我倾向于后者解读，但值得确认。
3. **IID/Polarity 如果让 metrics.py 输出 NaN**，会不会破坏现有已经在跑的 case？需要先 grep 一下有哪些地方消费这两列。
4. **Result / Discussion 分离**我判定为 P2，但如果未来 `report-writer` 生成 APA 报告，它可能依赖 data-analyst 的输出结构——如果是这样应该提到 P1。
5. **expected-analysis.yaml 加 `clarification_required` 这种新 type**，当前的 golden case 回归测试 runner 可能不支持——需要确认 SCHEMA.md 里是否已定义此 type，或者需要同步扩展 schema。

---

## 5. 一句话总结

E2E 测试的**工程流水线**工作正常（数据流、质量门控、统计严谨性都在线），但**与行为学同事的 ground truth 对齐度**有三个系统性缺口：(a) 盲目接受 Control/Treatment 裸 label 不追问处理描述；(b) 主动建议"排除"离群个体越过了 agent 的能力边界；(c) IID/Polarity 用虚假数值当"疑似 artifact" 而没修到 metrics 计算的根因。三条都是 prompt + skill reference + metrics.py 的局部改动可以修掉，不需要重构流水线。

---

## 6. 交叉评审请关注

- 第 2 节的 5 个差距，哪些被我夸大了？哪些其实是行为学同事的 ANSWER 被我过度解读？
- 第 3 节的优先级排序（P1/P2/P3）是否合理？有没有哪条应该升级或降级？
- 第 4 节我列出的 5 个不确定点，是否有其他维度的 risk 我没意识到？
- 整份文档是否遗漏了 e2e 测试结果里其他值得 flag 的问题？
