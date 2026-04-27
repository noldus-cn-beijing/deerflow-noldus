# Golden Case 对齐 Skill 与 Report-Writer 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把行为学同事在 `golden-cases/case-001-shoaling-baseline/` 里确认的 6 条判读规则，落实到 skill 层（4 个 custom skill）和 report-writer subagent 的 system_prompt 里，让下一次 e2e 测试时 agent 的输出与 golden case ground truth 对齐。

**Architecture:** 纯 skill + subagent prompt 改动。不碰 DeerFlow 上游受保护文件（lead_agent/prompt.py、mcp/tools.py、sandbox/tools.py、subagents/builtins/__init__.py），不碰 ethoinsight 库代码（metrics.py / assess.py / statistics.py 已验证正确）。`report_writer.py` 虽在 `subagents/builtins/` 下，但用户确认是 Noldus 自建的工具型 subagent（非 DeerFlow 原生），可修改。

**Tech Stack:** Markdown（skill reference）、Python（report_writer.py system_prompt 字符串）、YAML（golden case expected-analysis）

**Ground Truth 来源：** `golden-cases/case-001-shoaling-baseline/notes.md` 中行为学同事以 `ANSWER:` 标注的 6 条规则：
1. 只做组间对比，不用常模/baseline
2. 离群判据用 NND/象限分布，不用 total distance_moved
3. 不主动建议排除个体（排除需生物学依据且论文报告）
4. Control/Treatment 裸 label 必须追问具体处理描述
5. Result 和 Discussion 必须分离
6. IID/NND 在 EthoVision 中是 JS Continuous 自定义变量，raw data 不一定包含

---

## File Structure

本次修改涉及 10 个文件（8 处修改、1 处重命名、1 处新建并验证回归）：

### Skill 层（`packages/agent/skills/custom/`）

| 文件 | 动作 | 责任 |
|---|---|---|
| `ethoinsight/SKILL.md` | 改第 30 行 | 删除 "正常范围" 引导词 |
| `ethoinsight/references/paradigm-interpretation.md` | 整份重写 | 去所有范式的绝对阈值表格；shoaling 段补 JS Continuous 说明 |
| `ethoinsight/references/confound-checklist.md` | 改第 6 行 | distance_moved 改为"混杂因素候选，不作离群依据" |
| `ethoinsight/references/effect-size-guide.md` | 改第 27-33 行 | 删除"三层验证"第 2 条 activity control |
| `ethoinsight/references/apa-reporting-format.md` | **重命名为 `report.md`** + 整份重写 | 6 段报告骨架指南（取代 APA 模板） |
| `ethoinsight-planning/SKILL.md` | 改 Step 2（第 52-63 行） | 加处理描述检查 |
| `ethoinsight-planning/references/quality-gates.md` | 删第 30 行两条 | 删除 distance_moved 触发的质量警告 |

### Subagent 层（`packages/agent/backend/packages/harness/deerflow/subagents/builtins/`）

| 文件 | 动作 | 责任 |
|---|---|---|
| `report_writer.py` | 重写 system_prompt（保留 language/contract/json_writing/failure/chunking 基础设施段，重写定位/workflow/formatting） | 从"APA 论文章节撰写者"转为"结构化严肃报告撰写者（读者：导师/教授）" |

### Golden Case 层（`golden-cases/case-001-shoaling-baseline/`）

| 文件 | 动作 | 责任 |
|---|---|---|
| `expected-analysis.yaml` | `should_not_contain` 补 4 条 + outlier_findings metrics 从 `["mean_nnd","distance_moved"]` 收窄到 `["mean_nnd"]` | 新增违规行为的回归锚点 |

### 验证层

| 文件 | 动作 | 责任 |
|---|---|---|
| `docs/e2e/` 下新建一份 e2e 结果 | 新建 | 用 case-001 raw data 重跑完整流水线，对比改动前后的 agent 输出 |

---

## Task 1: 改 ethoinsight/SKILL.md 第 30 行

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight/SKILL.md:30`

**背景：** 第 30 行当前指向 `paradigm-interpretation.md` 时说"查阅**正常范围**和**异常判断标准**"——这与第 16-20 行的方法论声明（"核心是组间对比，不是绝对阈值"）自相矛盾。

- [ ] **Step 1: 读当前文件**

```bash
# 读 packages/agent/skills/custom/ethoinsight/SKILL.md 第 28-34 行
```

Expected: 看到第 30 行 `根据实验范式（EPM、OFT、Shoaling 等）查阅正常范围和异常判断标准。详见 references/paradigm-interpretation.md。`

- [ ] **Step 2: 改第 30 行**

将：
```
根据实验范式（EPM、OFT、Shoaling 等）查阅正常范围和异常判断标准。详见 `references/paradigm-interpretation.md`。
```

改为：
```
根据实验范式（EPM、OFT、Shoaling 等）查阅指标含义和组间对比判读原则。详见 `references/paradigm-interpretation.md`。
```

- [ ] **Step 3: 验证改动**

```bash
grep -n "组间对比判读原则" packages/agent/skills/custom/ethoinsight/SKILL.md
```

Expected: 输出第 30 行包含新措辞

- [ ] **Step 4: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/SKILL.md
git commit -m "fix(skill): ethoinsight SKILL 方法论与 paradigm-interpretation 引用对齐

删除 '正常范围和异常判断标准' 引导词，改为 '指标含义和组间对比判读原则'，
与 SKILL 开头第 16-20 行声明的'组间对比，不用阈值'方法论一致。对应
golden-case notes.md ANSWER 规则 1。"
```

---

## Task 2: 重写 paradigm-interpretation.md

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md`（整份重写）

**背景：** 当前文件为 6 个范式列出"正常范围 / 高焦虑 / 低焦虑"三栏绝对阈值表格，违反 golden case ANSWER 规则 1 和 CLAUDE.md 第 9 条铁律。

- [ ] **Step 1: 整份重写**

用下面内容**完全替换**文件：

```markdown
# 范式解读指南

## 判读原则（所有范式通用）

- **组间对比为唯一判据**：以统计检验（Mann-Whitney / t-test 视分布）+ 效应量（Cohen's d）为结论基础
- **不使用绝对阈值或常模范围**——行为学数据受品系/日龄/温度/光照等多因素影响，无跨实验通用 baseline
- **Result 段仅含统计量**；行为学解读放 Discussion 段
- **离群判据依范式专属指标**，不统一使用 total distance_moved

## 高架十字迷宫 (EPM)

### 核心指标
- **开臂时间比**: 焦虑水平的经典读出；treatment 组显著低于 control → 焦虑样行为
- **开臂进入比**: 开臂进入次数 / 总进入次数，补充开臂时间比
- **总臂进入次数**: 总活动量，用于排除运动量混杂（不作为焦虑读出）

## 旷场实验 (Open Field)

### 核心指标
- **中心区时间比**: 焦虑水平的读出；treatment 组显著低于 control → 焦虑样行为
- **总移动距离**: 总活动量，用于排除运动量混杂（不作为焦虑读出）
- **平均速度**: 补充活动量指标

## O 迷宫 (Zero Maze)

### 核心指标
- **开放区时间比**: 焦虑水平的读出
- **方向偏好指数**: 行为侧化指标，组间比较看是否有差异

## 明暗箱 (Light-Dark Box)

### 核心指标
- **明箱时间比**: 焦虑水平的读出
- **明暗穿梭次数**: 活动量/探索性指标
- **首次进入明箱潜伏期**: 探索动机指标

## 新物体识别 (Novel Object)

### 核心指标
- **辨别指数** = (新物体探索时间 - 熟悉物体探索时间) / 总探索时间
- 组间比较：treatment vs control 的 DI 差异；单样本 t 检验 vs 0 判断是否有记忆

## Y 迷宫 (Y-Maze)

### 核心指标
- **自发交替率** = 实际交替 / (总进臂次数 - 2) × 100%
- 与随机水平 22.2% 比较（单样本 t 检验）
- 组间比较：treatment vs control 的自发交替率差异

## 斑马鱼群体行为 (Shoaling)

### 核心指标
- **IID (Inter-Individual Distance)**: 所有鱼两两距离的平均值，反映群体紧密度
- **NND (Nearest Neighbor Distance)**: 每条鱼到最近同伴的距离，对离群个体敏感
- **群体极性 (Polarity)**: R ∈ [0,1]，越高越同向
- **象限分布**: 四象限内的鱼数分布及聚集时间

### 离群判据（shoaling 专属）
按优先级：
1. **主判据**：mean_nnd 远高于群体均值 → 远离群体
2. **主判据**：象限分布（长期停留单一象限且远离其他 subject）
3. **辅助判据**：mean_iid（需确认数据来源，见下文）
4. **禁用判据**：单独使用 total distance_moved 或 velocity_mean 判定离群
   - 运动量低可能只是探索策略差异，不等于离群
   - 若 distance_moved 与同组差异大，可在 Discussion 段作为"混杂因素候选"提及，但不独立成立离群结论

### 数据来源说明
- IID / NND / 象限分布在 EthoVision XT 中通常是 **JS Continuous / State 自定义变量**，需研究员在项目里启用对应脚本（完整 JS 脚本示例见 `golden-cases/case-001-shoaling-baseline/notes.md` 第 55-320 行）
- 导出 raw data **不一定包含这些列**——ethoinsight.metrics 会从 X/Y 坐标自行计算作为兜底
- 两种来源的数值可能不同（脚本版可含研究员的领域参数，坐标计算版是通用几何）
- handoff 应标注数据来源；Discussion 段解读时注明来源以避免误解
- metrics.py 在只有 1 个 subject 时 IID/Polarity 返回 `{"applicable": false, "reason": "..."}`，不编造假数值——agent 识别到 applicable=false 时应如实告知用户群体指标不适用

### 离群个体处理原则
- **不主动建议"排除"个体**。排除个体是生物学判断（造模失败 / 任务学习失败 / 设备故障），不是统计判断（数值偏离多少 σ）
- 若发现可能离群个体，措辞为"建议单独标注该个体，并检查是否有生物学排除依据，最终是否剔除由研究员判断"
- 论文中排除个体必须报告原因和数量
```

- [ ] **Step 2: 验证文件**

```bash
wc -l packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md
grep -c "正常范围" packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md
grep -c "组间对比" packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md
```

Expected:
- 行数约 60-80 行
- `正常范围` 计数为 0
- `组间对比` 计数 ≥ 2

- [ ] **Step 3: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md
git commit -m "refactor(skill): paradigm-interpretation 删除所有绝对阈值，改为组间对比原则

- 删除 EPM/OFT/Zero Maze/Light-Dark/NOR/Y-maze/Shoaling 所有范式的
  '正常范围 / 高焦虑 / 低焦虑' 三栏阈值表格（违反 CLAUDE.md 第 9 条）
- 每个范式改为'核心指标 + 判读原则'说明
- Shoaling 段扩展：补充 JS Continuous 数据来源说明、离群判据优先级、
  离群个体处理原则（不主动建议排除）
- 对应 golden-case notes.md ANSWER 规则 1, 2, 3, 6"
```

---

## Task 3: 改 confound-checklist.md 第 6 行

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight/references/confound-checklist.md:6`

**背景：** 第 6 行 "总距离/速度异常 → 焦虑/抑郁指标不可信——可能是运动障碍" 把 distance_moved 当成焦虑指标的污染源，违反 golden case ANSWER 规则 2。

- [ ] **Step 1: 定位当前行**

```bash
grep -n "运动量异常" packages/agent/skills/custom/ethoinsight/references/confound-checklist.md
```

Expected: 第 5 行输出 `- **运动量异常**: 如果总距离/速度异常（< 对照组 50% 或 > 200%），焦虑/抑郁指标不可信——可能是运动障碍`

- [ ] **Step 2: 替换该行**

将：
```
- **运动量异常**: 如果总距离/速度异常（< 对照组 50% 或 > 200%），焦虑/抑郁指标不可信——可能是运动障碍
```

替换为：
```
- **运动量差异的处理方式**: 若某 subject 的 distance_moved 或 velocity 与同组其他个体差异较大：
    - 不作为离群值排除依据（运动量低可能是探索策略差异，不是病理异常）
    - 不作为"焦虑指标不可信"的理由
    - 可作为 Discussion 段的混杂因素候选，提示研究员检查是否有独立的健康/造模依据
```

- [ ] **Step 3: 验证**

```bash
grep -c "可能是运动障碍" packages/agent/skills/custom/ethoinsight/references/confound-checklist.md
grep -c "探索策略差异" packages/agent/skills/custom/ethoinsight/references/confound-checklist.md
```

Expected:
- `可能是运动障碍` 计数 0
- `探索策略差异` 计数 1

- [ ] **Step 4: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/references/confound-checklist.md
git commit -m "fix(skill): confound-checklist 运动量差异不再作为污染源

distance_moved 差异改为 Discussion 段的混杂因素候选而非排除依据。对应
golden-case notes.md ANSWER 规则 2：离群判据不应使用总运动距离。"
```

---

## Task 4: 改 effect-size-guide.md "三层验证"第 2 条

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight/references/effect-size-guide.md:27-33`

**背景：** "表型确认三层验证"第 2 条 "活动距离正常排除运动障碍" 是 distance_moved 污染源论的又一复刻。

- [ ] **Step 1: 定位**

```bash
grep -n "表型确认\|活动控制" packages/agent/skills/custom/ethoinsight/references/effect-size-guide.md
```

Expected: 看到第 27 行"表型确认三层验证"标题及第 30 行"活动控制"

- [ ] **Step 2: 替换"三层验证"段**

将：
```
## 表型确认三层验证

确认焦虑表型需要：
1. **方向一致性**: 多个范式信号方向一致（EPM 开臂减少 + 旷场中心回避）
2. **活动控制**: 总活动距离正常（排除运动障碍）
3. **效应量**: 统计显著（p < 0.05）且 Cohen's d > 0.5
```

替换为：
```
## 表型确认原则

确认行为表型需要综合考虑：
1. **方向一致性**: 多个范式或多个范式内指标的信号方向一致（EPM 开臂时间减少 + 旷场中心回避）
2. **效应量**: 统计显著（p < 0.05）配合 Cohen's d > 0.5；仅 p 值不足以下结论
3. **可重复性**: 同一实验室多批次结果一致（若有多批数据）

注意：total distance_moved 或 velocity 差异不作为"表型确认"或"排除运动障碍"的判据——
活动量差异可能反映探索策略不同、造模程度不同等多种原因，不能独立判定病理状态。
```

- [ ] **Step 3: 验证**

```bash
grep -c "排除运动障碍" packages/agent/skills/custom/ethoinsight/references/effect-size-guide.md
grep -c "表型确认原则" packages/agent/skills/custom/ethoinsight/references/effect-size-guide.md
```

Expected:
- `排除运动障碍` 计数 0
- `表型确认原则` 计数 1

- [ ] **Step 4: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/references/effect-size-guide.md
git commit -m "fix(skill): effect-size-guide 表型确认删除 activity control

三层验证第 2 条 '活动距离正常排除运动障碍' 与 golden-case ANSWER 规则 2
冲突。改为'表型确认原则'，明示 distance_moved 差异不作独立判据。"
```

---

## Task 5: 重命名并重写 report.md（原 apa-reporting-format.md）

**Files:**
- Rename: `packages/agent/skills/custom/ethoinsight/references/apa-reporting-format.md` → `packages/agent/skills/custom/ethoinsight/references/report.md`
- Modify: `packages/agent/skills/custom/ethoinsight/SKILL.md:42`（同步改引用路径）

**背景：** 原文件是 4 行 APA 句式模板。按用户决策：当前阶段不输出 APA 格式，而是"对 Result 的洞察"。APA 格式留待 noldus-kb 接入后作为未来 plugin。

- [ ] **Step 1: git mv 重命名**

```bash
cd /home/qiuyangwang/noldus-insight
git mv packages/agent/skills/custom/ethoinsight/references/apa-reporting-format.md \
       packages/agent/skills/custom/ethoinsight/references/report.md
```

- [ ] **Step 2: 整份重写 report.md**

用下面内容**完全替换**文件：

```markdown
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
```

- [ ] **Step 3: 同步 SKILL.md 引用路径**

改 `packages/agent/skills/custom/ethoinsight/SKILL.md` 第 42 行：

将：
```
使用 APA 格式模板撰写统计结果描述。详见 `references/apa-reporting-format.md`。
```

替换为：
```
按报告结构骨架撰写对 Result 的洞察。详见 `references/report.md`。
```

- [ ] **Step 4: 验证**

```bash
ls packages/agent/skills/custom/ethoinsight/references/ | grep -E "apa|report"
grep -n "report.md\|apa-reporting-format.md" packages/agent/skills/custom/ethoinsight/SKILL.md
```

Expected:
- 只看到 `report.md`，没有 `apa-reporting-format.md`
- SKILL.md 第 42 行引用 `report.md`，无旧文件引用

- [ ] **Step 5: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/
git commit -m "refactor(skill): apa-reporting-format → report 重命名 + 内容重写

- 当前阶段报告目标是'给导师看的严肃结构化报告'，不是 APA 投稿章节
- 重写为 6 段骨架 + 统计量呈现规则 + Result/Discussion 分离原则
- APA 格式和 PDF 输出留作未来 plugin（noldus-kb 接入后）
- SKILL.md 第 42 行引用路径同步更新"
```

---

## Task 6: 改 ethoinsight-planning/SKILL.md Step 2

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-planning/SKILL.md:52-83`

**背景：** 当前 Step 2 只检查"是否有分组"，不检查分组 label 的语义。用户说 "Control vs Treatment" 就直接通过，不追问具体处理描述。违反 golden case ANSWER 规则 4。

- [ ] **Step 1: 定位 Step 2 段**

```bash
grep -n "Step 2\|需求完整性检查\|必问项" packages/agent/skills/custom/ethoinsight-planning/SKILL.md
```

Expected: 看到第 52 行左右的 Step 2 标题

- [ ] **Step 2: 替换 Step 2 表格**

将第 52-63 行的：

```markdown
### Step 2: 需求完整性检查（仅 2 个必问项）

检查以下信息：

| 信息 | 推断来源 | 缺失时行动 |
|------|---------|----------|
| **范式** | 文件名关键词（如 EPM, OFT, Shoaling） / 用户明示 | **推断失败 → `ask_clarification`** |
| **分组** | 文件名前缀（如 control_*, treatment_*） / 用户明示 | **无法推断 → `ask_clarification`** |
| 实验设计 | 关键词表（重复测量/配对/独立组） | 推断失败 → 走"自动判断" |
| 特殊需求 | 用户额外说明 | 缺失 → 走默认 |

**关键规则**：范式或分组缺失 → 立即 `ask_clarification`，不要进入后续步骤。
```

替换为：

```markdown
### Step 2: 需求完整性检查（3 个必问项）

检查以下信息：

| 信息 | 推断来源 | 缺失时行动 |
|------|---------|----------|
| **范式** | 文件名关键词（如 EPM, OFT, Shoaling） / 用户明示 | **推断失败 → `ask_clarification`** |
| **分组** | 文件名前缀（如 control_*, treatment_*） / 用户明示 | **无法推断 → `ask_clarification`** |
| **处理描述** | 用户消息中是否提及药物名/剂量/造模类型/基因型/年龄/性别等实质处理信息 | **分组为通用 label（control/treatment/对照/实验/ctrl/trt/组 1/组 2）且未提供处理描述 → `ask_clarification`** |
| 实验设计 | 关键词表（重复测量/配对/独立组） | 推断失败 → 走"自动判断" |
| 特殊需求 | 用户额外说明 | 缺失 → 走默认 |

**关键规则**：
- 范式 / 分组 / 处理描述缺失 → 立即 `ask_clarification`，不要进入后续步骤
- 处理描述已在本 session 早期追问过并写入 handoff_planning.json 的 `group_semantics` 字段 → 跳过追问
```

- [ ] **Step 3: 在 `ask_clarification` 示例段补一个处理描述的示例**

定位原文件第 65-83 行（两个现有 ask_clarification 示例之后），追加第三个示例：

```python
# 分组 label 通用但未提供处理描述
ask_clarification(
    question="请问实验组（Treatment）对应的具体处理是什么？例如药物剂量、造模类型、基因型差异、年龄等。",
    clarification_type="missing_info",
    context="分组 label 为通用名，需要具体处理描述以指导 Discussion 段解读",
    options=None  # 开放式，无预设选项
)
```

在这个示例后追加一段说明：

```markdown
**session 内去重**：用户回答后，把处理描述写入 handoff_planning.json 的 `group_semantics` 字段（格式：`{"control": "saline", "treatment": "30 mg/kg fluoxetine"}`）。下次 planning 先读该字段，已有则跳过追问。
```

- [ ] **Step 4: 验证**

```bash
grep -n "处理描述\|group_semantics" packages/agent/skills/custom/ethoinsight-planning/SKILL.md
grep -c "3 个必问项" packages/agent/skills/custom/ethoinsight-planning/SKILL.md
```

Expected:
- 看到 "处理描述"、"group_semantics" 多次出现
- `3 个必问项` 计数 1

- [ ] **Step 5: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-planning/SKILL.md
git commit -m "feat(skill): planning Step 2 新增'处理描述'检查

分组 label 为 control/treatment 等通用名且用户未提供具体处理描述时，
必须追问一次（药物/造模/基因型/年龄等）。对应 golden-case ANSWER 规则 4。
session 内通过 group_semantics 字段去重避免重复追问。"
```

---

## Task 7: 删除 quality-gates.md 第 30 行的 distance_moved 警告

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md:29-31`

**背景：** 当前文件第 29-31 行把 distance_moved > 200% 或 < 50% 当作质量警告触发条件，违反 golden case ANSWER 规则 2。

- [ ] **Step 1: 定位**

```bash
grep -n "总运动量\|运动亢进\|运动障碍" packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
```

Expected: 看到第 29-30 行的两条警告条目

- [ ] **Step 2: 删除两行**

将：
```
- 某只动物总运动量异常偏高（> 对照组 200%）→ 可能运动亢进
- 某只动物总运动量异常偏低（< 对照组 50%）→ 可能运动障碍
- 轨迹中断（missing data > 10%）
- 采样频率不一致
```

改为（删除前两条，保留后两条）：
```
- 轨迹中断（missing data > 10%）
- 采样频率不一致
- 某只动物的 mean_nnd 或象限分布明显偏离群体（shoaling 范式专属）→ 提示研究员检查该个体的生物学依据
```

- [ ] **Step 3: 验证**

```bash
grep -c "运动亢进\|运动障碍" packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
grep -c "mean_nnd 或象限分布" packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
```

Expected:
- `运动亢进|运动障碍` 计数 0
- `mean_nnd 或象限分布` 计数 1

- [ ] **Step 4: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md
git commit -m "fix(skill): quality-gates 删除 distance_moved 触发的质量警告

删除 '总运动量 > 200% / < 50%' 两条警告条目（违反 golden-case ANSWER 规则 2）。
新增 shoaling 专属警告：mean_nnd 或象限分布明显偏离群体时提示检查生物学依据。"
```

---

## Task 8: 重写 report_writer.py 的 system_prompt

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`（重写 system_prompt 字符串；保留 SubagentConfig 其他字段如 tools、disallowed_tools、model、max_turns、timeout_seconds 不变）

**背景：** 当前 system_prompt 定位为"APA 论文 Results + Discussion 撰写者"，和用户对 report-writer 的实际定位（给导师看的严肃结构化报告）严重不符。改动保留 `<language>`、`<contract>`、`<json_writing>`、`<failure>`、`<write_file_chunking>` 等基础设施段落，重写定位开头、`<structure>`、`<禁止的写法>`、`<workflow>`、`<formatting>` 段。

- [ ] **Step 1: 读当前 report_writer.py**

```bash
# 完整读一遍当前文件，重点注意 system_prompt 以外的字段（tools, disallowed_tools 等）
```

Expected: 对文件整体结构有认知，确保只改 system_prompt 字段。

- [ ] **Step 2: 定位 system_prompt 起止**

当前文件 system_prompt 从第 11 行 `system_prompt="""你是行为神经科学的科学报告撰写者。` 开始，到第 121 行 `让 lead agent 决定是否与用户重新沟通报告需求` 结束（三引号闭合）。

- [ ] **Step 3: 用新 system_prompt 替换整段**

将整段 `system_prompt="""..."""` 替换为：

```python
    system_prompt="""你是行为神经科学的研究报告撰写者。你的读者是研究员的导师 / 教授 / 学术监督者，他们会用 5-10 分钟阅读这份报告，判断这次实验做了什么、结论是什么、下一步该怎么走。

你写的不是期刊投稿论文，不套 APA 句式，不做文献综述。你写的是**一份严肃、结构化、可信的研究报告**，让导师扫一眼就能抓住重点，细看能追溯到每个数值。

<语言>
**输出语言必须与用户语言一致**：
- lead agent 派发任务时，会在 prompt 开头声明用户使用的语言
- 如果 lead 未明确声明，从任务描述中推断：中文任务用中文、英文任务用英文
- 所有输出（最终消息、write_file 内容、handoff_*.json 里的自由文本字段）
  都用同一种语言
- 统计术语、变量名、文件路径可以保留英文（它们是专有名词）
- 统计符号（M, SD, p, U, d 等）为国际通用，不需翻译
- handoff_report_writer.json 的 `sections_written` 字段值固定使用中文章节名
  （便于下游消费），不跟随用户语言变化
</语言>

<contract>
输入（两个 handoff 文件 + 可选数据快照）:
  - /mnt/user-data/workspace/handoff_code_executor.json —— 数据和统计原始结果
    （metrics_summary / per_subject / statistics / chart_paths ...）
  - /mnt/user-data/workspace/handoff_data_analyst.json —— 专业解读
    （key_findings / outlier_findings / method_warnings / excluded_metrics /
    recommendations）
  - /mnt/shared/code_summary.json —— 可选兜底，和 handoff_code_executor 重叠度高
  - /mnt/user-data/workspace/handoff_planning.json —— 若存在，可读 group_semantics
    字段获取处理描述（由 planning skill 追问得到）

输出（两样都要）:
  1. **/mnt/user-data/outputs/report.md** —— 结构化研究报告（见 <structure> 段）
  2. **/mnt/user-data/workspace/handoff_report_writer.json** —— 结构化交接文件

handoff_report_writer.json schema:
{
  "status": "completed" | "failed",
  "report_path": "/mnt/user-data/outputs/report.md",
  "sections_written": ["实验概况", "分析方法", "结果", "观察与洞察", "数据质量与局限", "下一步建议"],
  "errors": [str, ...]
}

工作范围:
  - 数据来源：handoff 文件（通过 read_file 读取）
  - 输出工具：write_file（写报告 + handoff JSON）和 ls（确认文件）
  - 图表已由 code-executor 生成，直接引用 chart_paths 中的路径（markdown 图片语法）
</contract>

<structure>
报告必须按以下 6 段骨架组织。章节编号保留，便于导师定位。

### 开头：一句话摘要
报告第一行是一句话摘要，以 blockquote 格式 `> ...` 呈现。格式示例：
> 本次分析了 X 条 [物种] 的 [范式] 行为，比较 [A 组] vs [B 组]，主要发现是 [核心结论]。样本量限制下结论为描述性。

### 1. 实验概况
- 范式：[从 handoff 读取]
- 受试个体：[总数、物种]
- 分组：[组名 (n=X): Subject X, Y, ...]
- 处理描述：[从 handoff_planning.json 的 group_semantics 字段读取；若未提供则**诚实写"用户未提供具体处理描述"**——不要编造]
- 数据来源：[EthoVision XT 导出 / Trial 数]

### 2. 分析方法
- 计算指标：[从 handoff.computed_metrics 列出]
- 统计方法：[t-test / Mann-Whitney U / ANOVA 等，从 handoff.statistics 读取]
- 方法选择依据：[若 method_warnings 非空，说明为何选此方法，例如"因 n<5 默认采用非参数 Mann-Whitney U"]
- 多重比较校正：[Bonferroni / Holm / 无]

### 3. 结果（仅陈述事实，不含解读）
本节只写数值和统计量。解读留到 §4。

#### 3.1 描述性统计
以表格呈现每组每指标的 M ± SD、n：
| 指标 | Control (n=X) | Treatment (n=Y) |
|-----|---------------|-----------------|

#### 3.2 组间比较
以 bullet 或小表列出每个指标的比较结果：
- mean_nnd: U = X, p = X.XX, Cohen's d = X.XX
- distance_moved: ...

不要写成 APA 句式（"t(10) = 2.34, p = .031, d = 0.85" 这种 inline 包装）。统计量直接列，让导师一眼看到数值。

#### 3.3 个体层面观察
仅陈述数值偏离的事实，不做行为学判断：
- ✅ "Subject 3 的 mean_nnd (70 mm) 明显高于同组其他个体 (36-40 mm)"  —— 事实
- ❌ "Subject 3 可能是造模失败" —— 这是解读，放 §4

#### 3.4 图表
引用 handoff.chart_paths 中的图表，用 markdown 图片语法：
- `![Figure 1: 组间 mean_nnd 箱线图](path/to/chart.png)`
- `![Figure 2: 轨迹图](path/to/trajectory.png)`

### 4. 观察与洞察（行为学解读）
本节整合 handoff_data_analyst 的 key_findings。用**自然段落**陈述解读，不用 APA 句式，不做文献引用（noldus-kb 未接入时）。

必须覆盖：
- **核心发现**：数据揭示了什么？组间差异的主要来源？
- **统计功效评估**：样本量是否允许下定论？（例如 "n=2 时 MWU 最小双尾 p=0.2，本设计下无法检测显著差异"）
- **关于离群个体**（若 handoff 中有 outlier_findings）：陈述偏离事实 + 建议研究员检查是否有造模失败 / 任务学习失败等生物学依据，**是否排除由研究员判断**。
  - ✅ "建议单独标注 Subject 3 并检查健康状态，是否纳入后续分析由研究员决定"
  - ❌ "建议排除 Subject 3"
  - ❌ "将 Subject 3 作为离群值剔除"
- **群体指标数据来源**（shoaling 范式）：若 mean_iid / mean_polarity 来自 X/Y 坐标兜底计算（非 EthoVision JS Continuous 变量），在此注明，提示解读时注意

### 5. 数据质量与局限
不是脚注，是让导师一眼看到的单独章节。

列出 handoff.data_quality_warnings 中的所有条目：
- 样本量限制：[具体到每组 n]
- 数据完整性：[Trial 数、missing data 比例]
- 指标适用性：[如 IID/Polarity 数据来源说明]
- 其他警告：[method_warnings / excluded_metrics]

### 6. 下一步建议
整合 handoff_data_analyst.recommendations。措辞克制——用"**可考虑的方向**"而非"**应该做**"，让导师保留决策权。

典型条目：
- 样本量扩充建议（基于功效分析估算目标 n）
- 补充实验建议（如补齐 Trial 2-N）
- 数据采集配置建议（如"若关注群体层面指标，建议在 EthoVision 项目中启用对应的 JS Continuous 自定义变量"）
- 分析方法建议（如后续可做的高级分析）

### 尾注
报告末尾加一行追溯信息：
---
*本报告由 EthoInsight 自动生成于 [日期] 的分析 session。结果与解读仅供研究参考，最终判断权在研究员与导师。*
</structure>

<禁止的写法>
本报告**不是**期刊论文，以下论文腔写法**禁用**：

- ❌ APA 句式包装："The treatment group showed significantly higher IID (M = 45.2, SD = 12.3) compared to controls (M = 32.1, SD = 15.7), t(10) = 2.34, p = .031, d = 0.85."
- ✅ 直接列数值：
    - Treatment mean_iid: 45.2 ± 12.3 mm
    - Control mean_iid: 32.1 ± 15.7 mm
    - Mann-Whitney U = X, p = X.XX, Cohen's d = 0.85

- ❌ 英文论文腔图表引用："As shown in Figure 1, the treatment group exhibited..."
- ✅ 中文自然描述："Figure 1 展示了组间 mean_nnd 的箱线图分布"

- ❌ 主动建议"排除"离群个体："建议将 Subject 3 作为离群值剔除后重新分析"
- ✅ "建议单独标注 Subject 3 并检查是否有生物学排除依据（如造模失败），是否纳入后续分析由研究员判断"

- ❌ 用绝对阈值判读："Treatment 组 mean_nnd 高于正常范围 (36-40 mm)，可能反映焦虑样行为"
- ✅ "Treatment 组 mean_nnd 高于 Control 组，但差异主要由 Subject 3 驱动，排除该个体后两组接近"

- ❌ 在 §3 结果段夹杂解读（Result 和 Discussion 必须分开）
- ✅ §3 只写数值，解读全部留到 §4

- ❌ 用 distance_moved 判定离群："Subject 3 的总运动距离仅为其他个体的 50%，应作为离群值"
- ✅ distance_moved 可在 §4 作为"混杂因素候选"提及，但不作为离群证据——离群判据用 mean_nnd 和象限分布

- ❌ 编造文献引用（noldus-kb 未接入时）
- ✅ §4 只做基于统计结果的行为学解读，不引文献
</禁止的写法>

<json_writing>
handoff_report_writer.json 必须是**合法的 JSON**——下游工具会 parse 它。
写字符串值时遵守以下规则，避免未转义的引号破坏 JSON 语法：

- 在字符串里想做**强调或引用短语**时：用中文全角引号 \"...\" 或书名号《》
- 需要**引用变量名、p 值表达式、参数**时：用单引号，例如 'p < 0.05'
- 真的必须写入半角双引号字符时：手动转义为 \\\"
- 不确定时就用中文引号——比半角安全

report.md（markdown 报告）本身不是 JSON，那里用什么引号都 OK。此规则只约束 handoff_report_writer.json 字符串值。
</json_writing>

<workflow>
1. read_file 两个 handoff 文件：
   - /mnt/user-data/workspace/handoff_code_executor.json（数据）
   - /mnt/user-data/workspace/handoff_data_analyst.json（解读）
   - 可选 read_file /mnt/user-data/workspace/handoff_planning.json 获取 group_semantics

2. 按 <structure> 段的 6 段骨架撰写报告：
   - 每段必须有，内容从对应 handoff 字段提取
   - §3 只写事实，§4 才做解读
   - 数据缺失时（如处理描述未提供）诚实写"未提供"，不编造

3. write_file /mnt/user-data/outputs/report.md 保存报告
   - 报告通常 3-8K 字符；超过 8000 时按 <write_file_chunking> 分段

4. write_file /mnt/user-data/workspace/handoff_report_writer.json 写交接文件

5. 最终 AIMessage：报告摘要（报告路径 + 各章节是否写全 + 任何失败条目）
</workflow>

<write_file_chunking>
结构化报告通常 3-8K 字符，一般单次写入足够。超过 write_file 单次 8000 字符上限时必须分段：
1. 第一次调用：append=False，写入 §开头摘要 + §1 + §2 + §3（约 6000-7500 字符）
2. 后续调用：append=True，写入 §4 + §5 + §6 + 尾注
3. 每次调用后读一次 write_file 返回值确认 "OK"，失败则调整切分点重试

write_file 若返回 "Error: Content exceeds 8000 chars..."，按错误消息里的指引分段。
</write_file_chunking>

<failure>
当 handoff_code_executor.json 或 handoff_data_analyst.json 读取失败，
或写入报告过程中反复出错：
- 仍然必须写出 handoff_report_writer.json，status 设为 "failed"，
  errors 字段记录失败原因
- 不要输出空报告或残缺报告
- 不要"假装"完成（比如把 data-analyst 的 key_findings 直接当作报告返回）
- 最终 AIMessage 明确声明失败：失败位置 + 原因
- 让 lead agent 决定是否与用户重新沟通报告需求
</failure>""",
```

**注意**：上述 Python 字符串中嵌入的转义（`\"`, `\\\"`）要如实保留，确保 Python 解析后得到的字符串内容正确。

- [ ] **Step 4: 验证 Python 语法**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate 2>/dev/null
python3 -c "from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG; print('OK', REPORT_WRITER_CONFIG.name, 'prompt len:', len(REPORT_WRITER_CONFIG.system_prompt))"
```

Expected: 输出 `OK report-writer prompt len: <4000-6000>` 量级，无 SyntaxError

- [ ] **Step 5: 跑 report_writer 相关测试（若有）**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ -k "report_writer" -v 2>&1 | tail -20
```

Expected: 相关测试通过（或无此测试，不报错即可）

- [ ] **Step 6: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
git commit -m "refactor(subagent): report_writer 从 APA 论文改为结构化严肃报告

- 定位：读者从'期刊审稿人'改为'研究员的导师/教授'
- 结构：新增 6 段骨架（一句话摘要 + 实验概况 + 分析方法 + 结果 + 观察与洞察 + 数据质量 + 下一步）
- Result/Discussion 严格分离（§3 只写事实，§4 才解读）
- 禁用 APA 句式包装、主动'排除'离群、绝对阈值判读、distance_moved 作离群依据
- 新增 handoff_planning.json 读取，使用 group_semantics 填充处理描述
- handoff schema sections_written 改中文章节名，删除 references_used 字段
- 保留 language/contract/json_writing/failure/chunking 基础设施段不变
- APA 格式和 PDF 输出留作未来 plugin"
```

---

## Task 9: 更新 golden case expected-analysis.yaml 回归锚点

**Files:**
- Modify: `golden-cases/case-001-shoaling-baseline/expected-analysis.yaml`

**背景：** 当前 `should_not_contain` 列表没覆盖本次 e2e 发现的违规行为（"建议排除"、"正常范围"等）。离群检测条目的 metrics 字段 `["mean_nnd", "distance_moved"]` 应收窄到只有 `["mean_nnd"]`，对应 golden case ANSWER 规则 2。

- [ ] **Step 1: 改 outlier_detection 条目**

定位第 58-65 行的 Subject 3 outlier_detection 条目：

```yaml
  - type: outlier_detection
    subject: "Subject 3"
    metrics: ["mean_nnd", "distance_moved"]
    reasoning: "TODO(行为学同事): Subject 3 的 mean_nnd=70.02 mm 远高于群体其余个体均值(约 37.5 mm)，且 distance_moved=12518 mm 远低于其余个体(约 24000 mm)。低运动量 + 高 NND 的组合是否指向某种表型？"
    severity: moderate
    required_keywords: ["Subject 3", "70"]
```

替换为：

```yaml
  - type: outlier_detection
    subject: "Subject 3"
    metrics: ["mean_nnd"]
    reasoning: "Subject 3 的 mean_nnd=70.02 mm 远高于群体其余个体均值(约 37.5 mm)，单独构成离群证据。distance_moved 差异仅作为 §4 Discussion 段的混杂因素候选，不独立成立离群结论（对应 golden-case ANSWER 规则 2）"
    severity: moderate
    required_keywords: ["Subject 3", "NND"]
```

- [ ] **Step 2: 扩充 should_not_contain**

定位第 97-103 行，将：

```yaml
should_not_contain:
  - "Subject 6"           # 数据里只有 5 个受试者
  - "Subject 7"
  - "p < 0.01"            # 样本量只有 5，不应有极小 p 值
  - "显著差异"             # 统计上不显著
  - "药物显著"             # 无实际药物处理
  - "处理效应显著"
```

替换为：

```yaml
should_not_contain:
  # 既有：数据一致性
  - "Subject 6"
  - "Subject 7"
  - "p < 0.01"
  - "显著差异"
  - "药物显著"
  - "处理效应显著"
  # 新增：离群个体处理越界（对应 ANSWER 规则 3）
  - "建议排除"
  - "排除后重新分析"
  - "作为离群值排除"
  - "将其剔除"
  # 新增：绝对阈值判读（对应 ANSWER 规则 1）
  - "正常范围"
  - "典型值"
  - "高于常模"
  - "低于常模"
  # 新增：APA 论文腔（对应报告结构指南）
  - "As shown in Figure"
  # 新增：distance_moved 作离群依据（对应 ANSWER 规则 2）
  - "总运动距离仅为"        # 通常是 'Subject X 的总运动距离仅为...' 作离群证据的措辞
```

- [ ] **Step 3: 验证 YAML 合法性**

```bash
cd /home/qiuyangwang/noldus-insight
python3 -c "import yaml; d = yaml.safe_load(open('golden-cases/case-001-shoaling-baseline/expected-analysis.yaml')); print('OK', len(d['should_not_contain']), 'forbidden claims')"
```

Expected: 输出 `OK 15 forbidden claims`（原 6 + 新增 9）

- [ ] **Step 4: 跑 golden-case 校验脚本**

```bash
cd /home/qiuyangwang/noldus-insight
python3 scripts/validate_golden_case.py golden-cases/case-001-shoaling-baseline/
```

Expected: schema 校验通过

- [ ] **Step 5: 提交**

```bash
cd /home/qiuyangwang/noldus-insight
git add golden-cases/case-001-shoaling-baseline/expected-analysis.yaml
git commit -m "test(golden-case): case-001 回归锚点对齐 ANSWER 规则

- outlier_detection.metrics 从 [mean_nnd, distance_moved] 收窄到 [mean_nnd]
  （ANSWER 规则 2：distance_moved 不作独立离群判据）
- should_not_contain 新增 9 条：
  * 离群处理越界（'建议排除'等 4 条，对应 ANSWER 规则 3）
  * 绝对阈值判读（'正常范围'等 4 条，对应 ANSWER 规则 1）
  * APA 论文腔与 distance_moved 作离群依据各 1 条"
```

---

## Task 10: 运行端到端验证

**Files:**
- Create: `docs/e2e/2026-04-24-斑马鱼鱼群轨迹分析-post-alignment.md`（新建，记录改动后 e2e 结果）

**背景：** 所有改动合入后，用 case-001 的同一份 raw data 重跑 e2e，对比改动前（`docs/e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md`）和改动后的 agent 行为，验证 7 个关键违规是否都被修复。

- [ ] **Step 1: 启动完整应用**

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make stop 2>&1 | tail -5
make dev 2>&1 | tail -10 &
sleep 15
curl -s http://localhost:2026/api/health 2>&1 | head -3
```

Expected: `make dev` 启动成功，nginx 能响应 health check

- [ ] **Step 2: 通过前端上传 case-001 raw data 并触发分析**

用浏览器打开 `http://localhost:2026`，在新 thread 中：
1. 上传 `golden-cases/case-001-shoaling-baseline/raw-data/` 下的 5 个轨迹文件
2. 发送消息："请分析这批斑马鱼 shoaling 数据"
3. 观察 agent 的第一次响应——是否在 planning 阶段就追问了"处理描述"（Task 6 的新增 Gate）

Expected: Planning skill 触发 ask_clarification，追问 Treatment 对应的处理描述

- [ ] **Step 3: 回答"沿用相同分组（Control 1,2 / Treatment 3,4,5），为演示分组无实际药物处理"**

触发完整流水线：code-executor → data-analyst → report-writer

- [ ] **Step 4: 检查生成的 report.md**

```bash
cat /home/qiuyangwang/noldus-insight/packages/agent/backend/.deer-flow/threads/<new-thread-id>/user-data/outputs/report.md
```

按 7 个检查项逐条对比：

| # | 检查项 | 改动前行为 | 期望改动后行为 |
|---|---|---|---|
| 1 | 是否追问处理描述 | 未追问 | ✅ planning 阶段追问一次 |
| 2 | 是否用 distance_moved 判离群 | 有 | ❌ 只引 mean_nnd |
| 3 | 是否建议"排除 Subject 3" | 有 | ❌ 改为"建议单独标注并检查生物学依据" |
| 4 | 是否引用"正常范围"或"典型值" | 未引（侥幸） | ❌ 继续不引用 |
| 5 | 报告是否有 §3 §4 清晰分离 | 混杂 | ✅ 严格 §3 只有数值、§4 才解读 |
| 6 | 是否用 APA 句式 "t(10) = 2.34, p = .031" 这种 | 有 | ❌ 直接列数值 |
| 7 | 是否包含 §6 "下一步建议" | 较弱 | ✅ 明确的"可考虑的方向" |

- [ ] **Step 5: 跑 golden-case 回归校验**

（假设项目已有 golden-case 跑 agent 实际输出的比对脚本；若没有，手动把 report.md 内容粘贴到脚本）

```bash
# 如项目有此脚本：
python3 scripts/check_golden_case.py \
    --case golden-cases/case-001-shoaling-baseline/ \
    --actual <thread-id>/user-data/outputs/report.md
# 如无此脚本，手动 grep：
grep -E "建议排除|排除后重新分析|正常范围|典型值|As shown in Figure|总运动距离仅为" \
    <thread-id>/user-data/outputs/report.md
```

Expected: 无任何 should_not_contain 命中

- [ ] **Step 6: 记录 e2e 结果**

新建 `docs/e2e/2026-04-24-斑马鱼鱼群轨迹分析-post-alignment.md`，按 `docs/e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md` 的相同格式记录本次 session 的完整对话和 agent 输出。文件末尾加一段"与 deepseek-fix 版本对比"小结，逐条列出 7 个检查项的实际表现。

- [ ] **Step 7: 提交 e2e 结果 + 停机**

```bash
cd /home/qiuyangwang/noldus-insight
git add docs/e2e/2026-04-24-斑马鱼鱼群轨迹分析-post-alignment.md
git commit -m "test(e2e): case-001 改动后端到端验证

对 Task 1-9 的所有改动运行端到端测试：
- 前端上传 case-001 raw-data，触发完整 planning → code-executor →
  data-analyst → report-writer 流水线
- 验证 7 个关键违规全部被修复（详见文档内对比表）
- golden-case should_not_contain 回归无命中"

cd /home/qiuyangwang/noldus-insight/packages/agent
make stop
```

---

## Self-Review

### 1. Spec Coverage（本计划覆盖了哪些 golden case 规则）

| Golden Case ANSWER 规则 | 对应 Task |
|---|---|
| 规则 1：不用常模/baseline | Task 1, 2, 5, 8, 9 |
| 规则 2：离群判据不用 distance_moved | Task 2, 3, 4, 7, 8, 9 |
| 规则 3：不主动建议排除个体 | Task 2, 5, 8, 9 |
| 规则 4：裸 label 追问处理描述 | Task 6, 8 |
| 规则 5：Result/Discussion 分离 | Task 5, 8 |
| 规则 6：IID/NND 是 JS 变量，数据来源说明 | Task 2, 5, 8 |

所有 6 条规则都有对应任务实施，且报告撰写（Task 8）+ 回归锚点（Task 9）+ e2e 验证（Task 10）三层兜底。

### 2. Placeholder Scan

快速扫描文本，确认无：
- "TBD" / "TODO" / "implement later" —— 无
- "add appropriate error handling" —— 无
- "Similar to Task N" —— 无，每个任务都有完整 diff
- 未定义的引用 —— 所有引用的文件路径、行号、grep 模式都具体到字符串

### 3. Type Consistency

检查跨 Task 的一致性：
- `handoff_planning.json` 的 `group_semantics` 字段：Task 6 定义、Task 8 消费 —— 名字一致 ✅
- report.md 的 6 段章节名："实验概况 / 分析方法 / 结果 / 观察与洞察 / 数据质量与局限 / 下一步建议" —— Task 5 指南、Task 8 handoff schema、Task 10 验证表格 三处命名一致 ✅
- skill reference 文件重命名 `apa-reporting-format.md` → `report.md`：Task 5 Step 3 同步改 `SKILL.md:42` 引用路径 ✅
- "处理描述 / group_semantics"术语：Task 6 用中文字段名，Task 8 Python 字符串中也用 `group_semantics` —— 通过 ✅

### 4. 额外检查：受保护文件边界

确认没有碰任何受保护文件（见 CLAUDE.md）：
- `agents/lead_agent/prompt.py` ❌ 未改
- `subagents/builtins/__init__.py` ❌ 未改
- `mcp/tools.py` ❌ 未改
- `sandbox/tools.py` ❌ 未改
- `subagents/builtins/report_writer.py` ✅ 已改（用户确认是 Noldus 自建非 DeerFlow 原生，非受保护）

---

## 交接执行

**Plan 完成，保存于 `docs/plans/2026-04-24-golden-case-alignment-plan.md`。**

两种执行方案：

**1. Subagent-Driven（推荐）** —— 每个 Task 派遣独立 subagent，Task 间 review，fast iteration。适合本计划因为 10 个 Task 彼此相对独立（除了 Task 1→2, Task 5 内部 rename→SKILL.md 引用更新、Task 10 依赖前 9 个完成）。

**2. Inline Execution** —— 在当前 session 依次执行，checkpoints 在 Task 5（rename）、Task 8（python syntax check）、Task 10（e2e）三处。

**推荐：Subagent-Driven**，理由：
- Task 2（paradigm-interpretation 整份重写）、Task 5（report.md 整份重写）、Task 8（system_prompt 重写）这三个大改动各自独立，适合并行或串行 subagent
- 每个 Task 有独立的验证步骤（grep / python syntax check / yaml.safe_load），subagent 可自检后 commit
- Task 10 必须在前 9 个都完成后由 user 或 fresh subagent 执行

**等你决定 execution 方式再启动。**
