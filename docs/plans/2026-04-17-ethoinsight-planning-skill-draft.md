# ethoinsight-planning Skill 草案

> 日期: 2026-04-17
> 状态: 草案（未实装）
> 目的: 将隐式、硬编码、分散的规划逻辑抽象成一个显式 skill，让模型有"领域专家式的任务拆解能力"

---

## 设计原则

1. **规划是 skill（方法论），TodoList 是 tool（状态）** — 两者配合使用
2. **领域特定 > 通用** — 这个 skill 只管行为学数据分析的规划，不是通用任务规划
3. **显式优于隐式** — 强制在开始工作前输出结构化计划
4. **分支逻辑显式化** — 把"什么情况下跳过 data-analyst"等判断从 prompt 里的 if-else 移到 skill 里成为显式决策树
5. **用户可见可改** — 计划作为第一条用户可见消息输出，用户可以打断重定向

---

## SKILL.md 草案

```markdown
---
name: ethoinsight-planning
description: >
  Task planning skill for behavioral data analysis workflows. Use BEFORE
  delegating to subagents when the user requests end-to-end analysis, follow-up
  questions on existing results, or mixed-intent tasks. Produces a structured
  plan that covers paradigm identification, group assignment, analysis scope,
  quality gates, and expected output. Always use this when `<uploaded_files>`
  contains new data files AND the user requests analysis.
version: 0.1.0
author: noldus-insight
---

# EthoInsight Planning — 行为学分析任务规划

## 何时使用此 skill

**必须使用**：
- 用户上传数据 + 请求分析（端到端流水线）
- 用户要求"重新分析"或"换种方式分析"已有数据
- 用户一次请求中混合多个意图（如"分析数据 + 解释概念"）

**可跳过**：
- 纯知识问答（直接派遣 knowledge-assistant）
- 追问已有分析结果的单一问题
- 闲聊、确认、感谢

## 核心原则

1. **规划先于行动** — 在派遣第一个 subagent 之前，必须输出完整计划
2. **计划对用户可见** — 用自然语言写给用户看，不是写给自己
3. **分支逻辑显式** — 说明每一步的前置条件和降级策略
4. **样本量 / 数据质量前置判断** — 不要等 code-executor 发现问题才处理

## Workflow

### Step 1: 意图分类

从用户消息中识别**主意图**和**副意图**：

| 主意图 | 触发信号 | 后续动作 |
|--------|---------|---------|
| 端到端分析 | 新上传数据 + "分析"/"统计"/"看看" | 进入 Step 2 |
| 追问已有结果 | 无新文件 + 指代已有结果的代词 | 直接派遣 knowledge-assistant |
| 范式/术语问答 | 无新文件 + 概念性问题 | 直接派遣 knowledge-assistant |
| 仅重写报告 | "重新写报告"/"换个格式" | 仅派遣 report-writer |
| 混合意图 | 同时包含多个上述信号 | 按主意图规划 + 末尾加副意图处理 |

### Step 2: 需求完整性检查

在开始规划前，检查以下信息是否齐全：

| 信息 | 推断来源 | 缺失时 |
|------|---------|-------|
| 范式 | 文件名关键词 / 用户明示 | ask_clarification（给 options） |
| 分组定义 | 用户消息 / 文件名前缀 | ask_clarification（必须） |
| 实验设计类型 | 关键词表（见 references/design-type-keywords.md） | 默认"自动判断" |
| 特殊需求 | 用户额外说明 | 默认"无" |

**关键规则**：范式或分组缺失 → 立即 ask_clarification，不要进入规划。

### Step 3: 制定分析计划

按范式和数据规模决定执行路径。详见 `references/planning-templates.md`。

**标准 3 步流水线（默认路径）**：
1. code-executor → 数据解析、指标计算、统计检验、图表生成
2. data-analyst → 结合领域知识解读 + 查询 noldus-kb
3. report-writer → APA 格式报告 + 文献引用

**可简化的场景**：
- 样本量 < 3/组 → 跳过 data-analyst（统计解读意义有限，直接给原始指标 + 警告）
- 用户明确说"只要数据，不要解读" → 跳过 data-analyst 和 report-writer
- 用户明确说"先不写报告，我想先看数据" → 只跑 code-executor，等用户确认

**需加强的场景**：
- 多范式同时分析 → Step 1 并行派遣多个 code-executor
- 重复测量 / 学习曲线 → 在 code-executor prompt 中明确标注设计类型
- 跨会话追问 → Step 2 之前先用 knowledge-assistant 回答上下文问题

### Step 4: 质量门控点规划

每个流水线阶段后的检查点：

| 阶段 | 检查项 | 降级策略 |
|------|--------|---------|
| code-executor 完成 | data_quality_warnings 字段 | 有警告 → ask_clarification，无警告 → 继续 |
| code-executor 失败 | 失败类型（范式不支持 / 数据格式 / 超时） | 按 lead_agent prompt 中的失败处理规则 |
| data-analyst 超时 | 返回内容是否为空 | 空 → 直接把 code_summary 展示给用户 |
| report-writer 超时 | 返回内容是否为空 | 空 → 用 data-analyst 的摘要作为最终输出 |

### Step 5: 输出计划

用**自然语言**把规划呈现给用户，格式：

```
我将按以下步骤分析你的 <范式名> 数据：

1. 📊 数据解析与统计 — 解析 N 个文件，计算 <关键指标>，执行 <统计方法> 检验
2. 🧠 专业解读 — 结合行为学知识评估效应量和生物学意义
3. 📝 科学报告 — 生成 APA 格式报告，包含效应量和参考文献

预计 ~X 分钟。如需调整（如跳过某步骤、换统计方法），请告诉我。
```

**关键要求**：
- 不要把计划写成代码或 JSON
- 用表情符号标识阶段，用户一眼能看懂
- 主动给出"可选调整"的提示，让用户有控制感
- **如果打算走非标准路径**（跳过步骤、特殊处理），必须解释原因

### Step 6: 执行计划

规划完成后，按 lead_agent prompt 中 `<orchestration_guide>` 描述的流程派遣 subagent。

---

## 决策清单（快速参考）

遇到以下情况时的标准应对：

```
Q: 用户上传数据但没说范式？
A: ask_clarification，提供 Top 5 范式 options

Q: 用户说"随便看看"？
A: ask_clarification，让用户选"完整分析"还是"先看数据"

Q: 用户一次传了 3 种不同范式的数据？
A: 并行派遣 3 个 code-executor（受 max_concurrent=3 限制）

Q: 每组只有 2 个样本？
A: 计划中明示"样本量小，仅做描述性统计，不做推断性检验"

Q: 用户之前分析过，这次说"再换个图表类型"？
A: 仅派遣 code-executor，跳过 data-analyst 和 report-writer

Q: 用户问"这个 p 值什么意思"？
A: 不进入规划，直接派遣 knowledge-assistant
```

---

## 反模式（不要做）

- ❌ 不要在没有完整信息时就开始规划（先澄清）
- ❌ 不要把计划藏在 thinking 里（必须对用户可见）
- ❌ 不要为每次简单追问都做复杂规划（知识问答直接派遣）
- ❌ 不要跳过质量门控点（每个阶段后必须检查）
- ❌ 不要在规划里写死"一定 3 步"（根据场景裁剪）

---

## 引用资源

- `references/intent-classification.md` — 意图分类决策树（完整版）
- `references/planning-templates.md` — 各范式的标准规划模板
- `references/design-type-keywords.md` — 实验设计类型关键词表
- `references/quality-gates.md` — 质量门控点详细规则
- `references/failure-recovery.md` — 失败场景降级策略
```

---

## 引用文件草案

### `references/planning-templates.md`（示例）

```markdown
# 范式分析规划模板

## Shoaling（斑马鱼群体行为）

**典型样本**：5 鱼/组，每组 2-4 轨迹文件
**核心指标**：NND、IID、群聚面积、运动速度
**统计方法**：Mann-Whitney U（非参数）或 t-test
**报告重点**：群聚性、社交焦虑

**标准规划输出**：
"我将执行 Shoaling 群体行为分析：
1. 计算 NND（最近邻距离）、IID（个体间距离）、群聚面积、运动速度
2. 对两组数据做正态性检验 → 自动选择参数/非参数方法
3. 生成群聚性热图 + 轨迹图 + 小提琴图
4. 解读群聚性变化的生物学意义
5. 生成 APA 格式对比报告"

## EPM（高架十字迷宫）

**典型样本**：8-12 只/组
**核心指标**：开臂时间比例、开臂进入次数比例、总进臂次数
**统计方法**：t-test 或 Mann-Whitney U
**报告重点**：焦虑样行为

**标准规划输出**：
"我将执行 EPM 焦虑评估：
1. 计算开臂时间比例、开臂进入次数、总活动量
2. 做两组独立对比的统计检验
3. 生成臂停留时间柱状图 + 轨迹热图
4. 评估焦虑水平变化（注意排除活动水平混杂）
5. 生成 APA 格式焦虑评估报告"

## Open Field（旷场）

**典型样本**：8-12 只/组
**核心指标**：中心区停留时间、总距离、直立次数
**统计方法**：t-test 或 Mann-Whitney U
**报告重点**：自主活动 + 焦虑（中心区回避）

（模板略）
```

---

## 实装要点

1. **放置位置**：`packages/agent/skills/custom/ethoinsight-planning/SKILL.md`
2. **启用方式**：在 `extensions_config.json` 中加 `"ethoinsight-planning": { "enabled": true }`
3. **默认启用** — Phase 0 期间就上，不需要等微调
4. **与 Plan Mode 的关系**：
   - Plan Mode（`write_todos` tool）是状态载体
   - 本 skill 是方法论指导
   - 建议在用户上传数据 + 请求分析时**自动开启 Plan Mode**（需要改路由逻辑）

---

## 与现有 prompt 的重构计划

这个 skill 上线后，lead_agent prompt 应做以下精简：

| prompt 中的内容 | 去向 |
|----------------|------|
| Step 0-4 派遣流程（orchestration_guide） | 保留（执行细节） |
| 实验设计关键词表 | 移到 skill 的 `references/design-type-keywords.md` |
| 失败处理规则 | 移到 skill 的 `references/failure-recovery.md` |
| 路由判断（if-else） | skill 的"意图分类"章节覆盖（更细致） |
| 澄清场景列表 | 保留（通用澄清原则） |

目标：**prompt 负责"怎么执行"，skill 负责"怎么规划"。**

---

## 验收标准

实装后应观察到以下改善：

1. ✅ 用户能看到完整计划（而不是 agent 默默开始）
2. ✅ 同类任务的计划输出格式稳定
3. ✅ 小样本、多范式、非标准场景的规划合理（不会机械走 3 步）
4. ✅ 规划错误时用户能在早期打断重定向
5. ✅ lead_agent prompt 精简 30% 以上

---

## 风险与取舍

1. **延迟增加**：每次分析前多一轮 LLM 输出（规划本身），约 +3-5s
2. **GLM-5.1 能否稳定输出**：现阶段 GLM-5.1 对复杂结构化任务不稳定，可能每次规划格式漂移
   - **应对**：在 skill 里给出**完整的输出模板**，而非让模型自由发挥
3. **用户可能不想看计划**：研究员想"扔数据拿报告"
   - **应对**：保持计划**简短**（3-5 行），用户可以忽略

---

## 下一步

- [ ] 用户确认 skill 设计方向
- [ ] 补全 5 个 references/ 文件
- [ ] 实装到 skills/custom/ethoinsight-planning/
- [ ] 端到端测试（用现有 demo-data）
- [ ] 精简 lead_agent prompt
- [ ] 评估对 GLM-5.1 的延迟影响
