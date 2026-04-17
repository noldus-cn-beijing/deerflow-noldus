# ethoinsight-planning Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 EthoInsight 中实装完整的三层数据分析任务规划机制——Lead agent prompt 核心原则 + `ethoinsight-planning` skill（完整模板和决策树）+ DeerFlow 原生 `write_todos` 状态机，通过用户手动开启 Plan Mode 激活。

**Architecture:** 三层协同——Prompt 层让模型"记起要规划"（5-10 行核心原则 + 明示引用 skill 路径），Skill 层提供"规划得像专家"的完整方法论（SKILL.md + 5 个 references 文件），TodoList 层作为规划产物的可视化状态机（仅在 Plan Mode 开启时激活）。

**Tech Stack:**
- Skill 文件：Markdown + YAML frontmatter（放在 `packages/agent/skills/custom/ethoinsight-planning/`）
- Prompt 改动：[prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 中的 `SYSTEM_PROMPT_TEMPLATE`（受保护文件，需登记 `scripts/sync-deerflow.sh`）
- 配置：`extensions_config.json` 注册新 skill
- 测试：pytest（`packages/agent/backend/tests/`）

**验收标准（用户手动 E2E 验证）：**
1. 上传 shoaling 数据 + "帮我分析"，agent 输出"单行计划摘要"后派遣 code-executor
2. 上传文件名无范式信息 + "分析一下"，agent 调用 `ask_clarification` 问范式
3. 上传 2 个 subject 文件，agent 提醒"小样本量"并询问是否继续
4. 用户手动开启 Plan Mode 后，进度以 TodoList 形式可见

---

## Task 1: 登记 `prompt.py` 为受保护文件（防止下次 sync 被覆盖）

**Files:**
- Modify: `scripts/sync-deerflow.sh`

**Step 1: 确认当前受保护清单**

Run: `grep -A 20 'PROTECTED_FILES=' scripts/sync-deerflow.sh | head -25`

Expected: 能看到 `agents/lead_agent/prompt.py` 已经在列表里（草案 2026-04-17-roadmap 中提过）。

**Step 2: 如果 `prompt.py` 不在受保护清单，则加入**

若 Step 1 已确认存在，**跳过此 Task，直接进入 Task 2**。

若不存在，在 `scripts/sync-deerflow.sh` 的 `PROTECTED_FILES` 数组中加入一行：

```bash
PROTECTED_FILES=(
    # 高侵入 - Noldus 核心业务逻辑
    "agents/lead_agent/prompt.py"
    # ... 其余保持不变
)
```

**Step 3: 提交**

```bash
git add scripts/sync-deerflow.sh
git commit -m "chore: 确认 prompt.py 在受保护文件清单中"
```

(如 Step 1 已确认在列表中，此 Task 无改动，跳过提交。)

---

## Task 2: 创建 skill 目录骨架 + SKILL.md

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/SKILL.md`

**Step 1: 创建目录**

Run:
```bash
mkdir -p packages/agent/skills/custom/ethoinsight-planning/references
```

**Step 2: 写 SKILL.md**

Create `packages/agent/skills/custom/ethoinsight-planning/SKILL.md`:

```markdown
---
name: ethoinsight-planning
description: >
  Task planning framework for EthoVision behavioral data analysis. MUST be
  loaded BEFORE delegating any subagent when the current message contains
  uploaded data files AND the user requests analysis, re-analysis, or
  report-only operations. Provides intent classification, completeness
  checking, paradigm-specific planning templates, and single-line user
  contract output. Skip for pure knowledge questions (no uploaded data).
version: 0.1.0
author: noldus-insight
---

# EthoInsight Planning — 行为学分析任务规划

## 何时使用此 skill

**必须使用（进入规划流程）**：
- 本轮消息 `<uploaded_files>` 有新上传文件 + 用户请求分析/统计/可视化/报告
- 用户要求"重新分析"/"换种方式分析"已有数据
- 用户仅要求"重写报告"（跳过 code-executor 的特殊路径）

**可跳过（不规划）**：
- 纯知识问答（无新文件 + 概念性问题）
- 追问已有分析结果（"这个 p 值什么意思"）
- 闲聊、确认、感谢

## 核心原则

1. **规划先于行动** — 派遣第一个 subagent 之前，必须完成意图分类和需求完整性检查
2. **单行计划可见** — 用 1-2 行自然语言把方向告诉用户，不要复杂格式
3. **低召回反问** — 仅在范式无法推断或分组无法推断时才 `ask_clarification`；其他情况走默认
4. **分支显式** — 样本量小、数据质量异常等场景走非标准路径时必须解释原因

## Workflow（规划 6 步）

### Step 1: 意图分类

根据本轮消息的 `<uploaded_files>` 和用户文本，确定主意图。详见 `references/intent-classification.md`。

快速判断表：

| 有新文件 | 用户要求 | 主意图 | 动作 |
|---------|---------|-------|------|
| 是 | 分析/统计/可视化/报告 | 端到端分析 | 进入 Step 2 |
| 是 | 仅打招呼或无明确指令 | 需澄清 | `ask_clarification` 询问分析意图 |
| 否 | 指代已有结果（"这个"、"刚才"）| 追问 | 派遣 knowledge-assistant |
| 否 | 概念性问题 | 知识问答 | 派遣 knowledge-assistant |
| 是 | "重新分析"/"换图表" | 重做分析 | 进入 Step 2，但可能简化 |
| 是 | "只重写报告" | 仅报告 | 仅派遣 report-writer |

### Step 2: 需求完整性检查（仅 2 个必问项）

检查以下信息：

| 信息 | 推断来源 | 缺失时行动 |
|------|---------|----------|
| **范式** | 文件名关键词（如 EPM, OFT, Shoaling） / 用户明示 | **推断失败 → `ask_clarification`** |
| **分组** | 文件名前缀（如 control_*, treatment_*） / 用户明示 | **无法推断 → `ask_clarification`** |
| 实验设计 | 关键词表（重复测量/配对/独立组） | 推断失败 → 走"自动判断" |
| 特殊需求 | 用户额外说明 | 缺失 → 走默认 |

**关键规则**：范式或分组缺失 → 立即 `ask_clarification`，不要进入后续步骤。

`ask_clarification` 示例：

```python
# 范式推断失败
ask_clarification(
    question="请问这批数据来自哪种实验范式？",
    clarification_type="missing_info",
    context="文件名无法推断范式，需要确认",
    options=["高架十字迷宫 (EPM)", "旷场 (Open Field)", "斑马鱼群体行为 (Shoaling)", "其他"]
)

# 分组无法推断
ask_clarification(
    question="请问哪些 Subject 是对照组，哪些是实验组？",
    clarification_type="missing_info",
    context="文件名无命名规律，需要分组定义"
)
```

### Step 3: 选择规划模板

根据范式和数据规模选择执行路径。详见 `references/planning-templates.md`。

**标准 3 步流水线（默认）**：code-executor → data-analyst → report-writer

**可简化**：
- 样本量 < 3/组 → 跳过 data-analyst + 提醒用户（Step 4 会触发）
- 用户明确"只要数据" → 跳过 data-analyst 和 report-writer
- 用户明确"仅重写报告" → 仅派遣 report-writer

**可并行**：
- 多范式同时分析 → Step 1 并行派遣多个 code-executor（受 max_concurrent=3 限制）

### Step 4: 质量门控点规划

详见 `references/quality-gates.md`。关键点：

| 触发条件 | 行动 |
|---------|------|
| 样本量 < 3/组（规划阶段从文件数判断） | 规划摘要中注明"小样本量，仅描述性统计" + `ask_clarification` 确认 |
| `data_quality_warnings` 非空（code-executor 返回后）| `ask_clarification` 询问排除异常还是继续 |
| code-executor 失败 | 按失败类型分支（见 `references/failure-recovery.md`）|
| data-analyst 超时/空返回 | 跳过，直接把 code_summary 展示给用户 |
| report-writer 超时/空返回 | 用 data-analyst 摘要作最终输出 |

### Step 5: 输出单行计划给用户

格式：**一句话说明将做什么 + 预计用时**

示例：
- "将对 EPM 数据执行统计分析 + 生成 APA 报告，约 2 分钟"
- "检测到 3 种范式数据，将并行分析并生成对比报告，约 5 分钟"
- "样本量较小（每组 2 只），将仅做描述性统计，不做推断性检验"

**禁止**：
- 不写多步列表（太啰嗦，违背"低召回"）
- 不写 JSON 或代码格式
- 不重复用户已经说过的需求

### Step 6: 执行

按 lead_agent prompt 中 `<orchestration_guide>` 描述的流程派遣 subagent。

---

## 决策清单（快速参考）

| 场景 | 应对 |
|------|-----|
| 用户上传数据但没说范式（文件名也看不出） | `ask_clarification` 给 options |
| 用户上传数据但没说分组（文件名无规律） | `ask_clarification`（无 options，开放式）|
| 用户一次传了 3 种范式 | 并行派遣 3 个 code-executor |
| 每组只有 2 个样本 | 计划中明示"小样本" + `ask_clarification` 确认 |
| 用户说"换个图表类型" | 仅 code-executor，跳过 data-analyst + report-writer |
| 用户问"这个 p 值什么意思"（无新文件）| 不规划，直接派遣 knowledge-assistant |

## 反模式

- ❌ 不要在信息不全时就开始规划（先澄清）
- ❌ 不要把计划写成多行列表（单行摘要）
- ❌ 不要为每次简单追问做复杂规划
- ❌ 不要跳过质量门控点
- ❌ 不要在规划里写死"一定 3 步"（按场景裁剪）

## 引用资源

- `references/intent-classification.md` — 意图分类决策树（完整版）
- `references/planning-templates.md` — 各范式标准规划模板
- `references/design-type-keywords.md` — 实验设计类型关键词表
- `references/quality-gates.md` — 质量门控点详细规则
- `references/failure-recovery.md` — 失败场景降级策略
```

**Step 3: 验证文件能被 skills loader 读取**

Run:
```bash
cd packages/agent/backend && source .venv/bin/activate
python -c "from deerflow.skills import load_skills; skills = list(load_skills(enabled_only=False)); names = [s.name for s in skills]; print('ethoinsight-planning' in names)"
```

Expected: `True`（但此时 enabled=False 是默认状态，因为还没注册到 extensions_config.json）

**Step 4: 提交**

```bash
git add packages/agent/skills/custom/ethoinsight-planning/SKILL.md
```

注：`skills/custom/` 是 gitignored（见 [CLAUDE.md](CLAUDE.md)），所以 `git add` 不会真的加入。这是预期行为——skill 文件不进 git，但要在本地文件系统存在。跳过 commit。

---

## Task 3: 创建 `references/intent-classification.md`

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/references/intent-classification.md`

**Step 1: 写 references 文件**

```markdown
# 意图分类决策树

## 信号来源

1. `<uploaded_files>` 中 "uploaded in this message" 部分（本轮新上传）
2. 用户消息文本
3. 已有会话历史（workspace 中是否有 analysis_report.md / metrics.csv）

## 决策树

```
本轮是否有新上传数据文件？
├── 是
│   ├── 用户消息包含分析词汇（"分析"/"统计"/"看看"/"处理"/"可视化"/"报告"）？
│   │   ├── 是 → 主意图: 端到端分析
│   │   └── 否（仅打招呼/无分析指令） → 主意图: 需澄清
│   │       → 动作: ask_clarification("用户刚上传了文件 X，但未提出分析请求，是否要分析？")
│   └── (已处理)
└── 否
    ├── 用户消息指代已有结果（代词"这个"/"刚才"/"上面"/提及具体指标如"p 值"/"NND"/"Cohen's d"）？
    │   ├── 是 → 主意图: 追问已有结果
    │   │       → 动作: 派遣 knowledge-assistant，附 workspace/analysis_report.md 路径
    │   └── 否
    │       └── 消息是概念性问题（"什么是 X"/"怎么做 X"/"X 和 Y 的区别"）？
    │           ├── 是 → 主意图: 知识问答
    │           │       → 动作: 派遣 knowledge-assistant
    │           └── 否 → 主意图: 闲聊/确认
    │                   → 动作: 自己回复，不派遣
```

## 特殊路径

### "重新分析"/"换种方式"
- 触发词：re, 重新, 换, different, 不一样
- 动作: 端到端分析，但在 code-executor prompt 中注明"重新分析"
- 如果用户具体指了某一步（如"只换图表类型"）→ 仅派遣 code-executor

### "只重写报告"
- 触发词：重写, 换个格式, 翻译, 精简, APA, 中文/英文
- 前置条件: workspace 中有 analysis_report.md 或 code_summary.json
- 动作: 仅派遣 report-writer，传入原分析结果路径
- 无前置数据 → `ask_clarification` 询问是否先做分析

### 混合意图
- 例: "帮我分析这批数据，顺便解释下什么是 NND"
- 处理: 主意图（端到端分析）优先 + 末尾追加一个 knowledge-assistant 回答副意图
- 或者在主流水线完成后，用 report-writer 的报告中自然包含术语解释

## 边界 case

| 情况 | 判断 |
|------|------|
| 有新文件但是非数据文件（如 PDF、图片） | 先 `ask_clarification` 询问用途 |
| 有新文件 + "?"（纯问号） | 等价"无明确指令"，`ask_clarification` |
| 有新文件 + 用户说"先不分析" | 不规划，直接回复"收到，需要分析时告诉我" |
| 无新文件 + "继续上次的分析" | 如果 workspace 有数据 → 根据上次的 paradigm 继续 |
```

**Step 2: 提交**

```bash
git add packages/agent/skills/custom/ethoinsight-planning/references/intent-classification.md
```

(skill 目录 gitignored，无实际提交；文件仅需落盘)

---

## Task 4: 创建 `references/planning-templates.md`

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/references/planning-templates.md`

**Step 1: 写模板文件**

```markdown
# 范式分析规划模板

每个范式包含：典型样本量、核心指标、统计方法、单行计划输出示例。

## Shoaling（斑马鱼群体行为）

- **典型样本**: 5 鱼/组，每组 2-4 个轨迹文件
- **核心指标**: NND（最近邻距离）、IID（个体间距离）、群聚面积、运动速度
- **统计方法**: ethoinsight 自动选择（正态 → t-test，非正态 → Mann-Whitney U）
- **单行计划**: "将对 Shoaling 数据执行群聚性分析（NND / IID / 群聚面积）+ 生成对比报告，约 2 分钟"

## EPM（高架十字迷宫）

- **典型样本**: 8-12 只/组
- **核心指标**: 开臂时间比例、开臂进入次数比例、总进臂次数
- **统计方法**: t-test / Mann-Whitney U（两组独立对比）
- **单行计划**: "将对 EPM 数据执行焦虑评估（开臂时间/进入比例）+ 生成 APA 报告，约 2 分钟"

## Open Field（旷场）

- **典型样本**: 8-12 只/组
- **核心指标**: 中心区停留时间、总距离、直立次数
- **统计方法**: t-test / Mann-Whitney U
- **报告重点**: 自主活动水平 + 焦虑（中心区回避）
- **单行计划**: "将对旷场数据执行活动/焦虑双维度分析 + 生成 APA 报告，约 2 分钟"

## Novel Object / Y-Maze（学习记忆）

- **典型样本**: 10+ 只/组
- **核心指标**: 识别指数、自发交替率
- **统计方法**: t-test / Mann-Whitney U，或 one-sample t-test 对比 50%
- **单行计划**: "将对 <范式名> 数据执行学习记忆评估 + 生成报告，约 2 分钟"

## 多范式并行

- **识别**: `<uploaded_files>` 中有多组文件，前缀不同（如 EPM_*.txt 和 OFT_*.txt）
- **计划**: 并行派遣多个 code-executor（≤ 3 个/轮），最后用 report-writer 综合对比
- **单行计划**: "检测到 <N> 种范式数据，将并行分析 + 生成对比报告，约 <N*2> 分钟"

## 降级路径

### 小样本量（< 3/组）
- **单行计划**: "样本量较小（每组 <N> 只），将仅做描述性统计，跳过推断性检验和 APA 报告"
- **执行**: 只派遣 code-executor，用简化 skill 获得 basic_metrics 而不跑统计；不派遣 data-analyst 和 report-writer
- **必须 `ask_clarification` 确认**：用户可能想继续或补数据

### 重复测量
- **识别**: 关键词"多天"、"Day 1-5"、"训练曲线"、"learning curve"
- **传递给 code-executor**: `实验设计: 重复测量（同一动物多次测量）`

### 配对设计
- **识别**: "给药前后"、"baseline vs post"
- **传递**: `实验设计: 配对设计（同一动物前后对比）`

### 多组（3+）
- **识别**: "3组"、"多剂量"、"low/mid/high"
- **传递**: `实验设计: 多组独立设计` → ethoinsight 会走 ANOVA
```

**Step 2: 落盘**

文件已创建，跳过 git commit。

---

## Task 5: 创建 `references/design-type-keywords.md`

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/references/design-type-keywords.md`

**Step 1: 写关键词表**

```markdown
# 实验设计类型关键词表

规划时从用户描述推断实验设计，传递给 code-executor。

| 关键词/信号 | 设计类型 | code-executor prompt 标注 |
|-----------|---------|--------------------------|
| "训练曲线"/"多天"/"Day 1-5"/"learning curve"/"habituation" | 重复测量 | 实验设计: 重复测量（同一动物多次测量） |
| "给药前后"/"baseline vs post"/"pre-post"/"before-after" | 配对 | 实验设计: 配对设计（同一动物前后对比） |
| "3组"/"多剂量"/"low/mid/high"/"dose-response"/"gradient" | 多组独立 | 实验设计: 多组独立设计 |
| "对照 vs 实验"/"control vs treatment"/"KO vs WT"/"sham vs model" | 两组独立 | 实验设计: 两组独立设计 |
| 无上述信号 | 自动判断 | 实验设计: 自动判断（由 ethoinsight 根据数据结构推断）|

## 推断优先级

1. 用户在本轮消息中明示 → 使用明示
2. 用户历史消息中提及 → 使用历史
3. 文件名包含关键词（如 "Day1_.txt", "pre_.txt"）→ 按文件名推断
4. 以上都无 → "自动判断"

## 不确定时的原则

**不要向用户反问实验设计**——这不是"必问项"（范式和分组才是）。让 ethoinsight 走"自动判断"路径，它会从数据结构推断。
```

**Step 2: 落盘**

完成，无 commit。

---

## Task 6: 创建 `references/quality-gates.md`

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md`

**Step 1: 写质量门控规则**

```markdown
# 质量门控点

在规划和执行的关键节点触发检查，必要时 `ask_clarification`。

## Gate 1: 规划阶段 — 样本量检查

**触发时机**: Step 2 需求完整性检查之后，Step 5 输出计划之前

**检查**: 每组样本数（从 `<uploaded_files>` 数量和分组定义推断）

| 每组样本数 | 行动 |
|-----------|------|
| ≥ 5 | 标准流水线，无特殊处理 |
| 3-4 | 执行标准流水线，但在 data-analyst prompt 中注明"样本量偏小，谨慎解读" |
| < 3 | **`ask_clarification`**：告知用户样本量不足，询问：(a) 仅做描述性统计 (b) 先补数据 (c) 强行跑统计（不推荐）|

## Gate 2: code-executor 返回后 — 数据质量警告

**触发时机**: code-executor 完成，返回 handoff JSON

**检查**: handoff JSON 中的 `data_quality_warnings` 字段

| 状态 | 行动 |
|------|------|
| 空数组 | 继续流水线 |
| 非空 | **`ask_clarification`**：列出警告项，询问：(a) 排除异常个体并重算 (b) 保留并继续 (c) 查看详情 |

**常见 warnings**:
- 某只动物总运动量异常偏高（> 对照组 200%）→ 可能运动亢进
- 某只动物总运动量异常偏低（< 对照组 50%）→ 可能运动障碍
- 轨迹中断（missing data > 10%）
- 采样频率不一致

## Gate 3: code-executor 失败

**触发时机**: code-executor 返回 status=failed 或超时

**行动**: 按失败类型分支，详见 `failure-recovery.md`

## Gate 4: data-analyst 超时/空返回

**触发时机**: data-analyst 完成后检查返回内容

| 状态 | 行动 |
|------|------|
| 返回正常解读 | 继续流水线，写 analysis_summary.md |
| 超时 | 跳过 report-writer，直接把 code_summary.json 的统计摘要展示给用户 |
| 返回空或仅重复统计 | 视为超时处理 |

## Gate 5: report-writer 超时/空返回

**触发时机**: report-writer 完成后检查返回内容

| 状态 | 行动 |
|------|------|
| 返回正常报告 | 用 present_files 展示给用户 |
| 超时 | 用 data-analyst 的 analysis_summary.md 作为最终输出 |

## 通用原则

- 每个 Gate 的 `ask_clarification` 必须给 options（用户不用思考）
- 连续 2 个 Gate 失败 → 必须 `ask_clarification` 整体方向，不能继续盲目流水线
- Gate 触发的信息必须如实告诉用户，不要隐瞒质量问题
```

**Step 2: 落盘**

---

## Task 7: 创建 `references/failure-recovery.md`

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-planning/references/failure-recovery.md`

**Step 1: 写失败恢复文档**

```markdown
# 失败场景降级策略

## code-executor 失败类型分类

根据失败信息的关键词分类：

| 关键词 | 类型 | 策略 |
|--------|------|------|
| "范式不支持"/"尚未支持"/"无模板"/"not supported" | 能力边界 | 降级选项 A |
| "文件解析失败"/"编码错误"/"No trajectory files found"/"parse error" | 数据格式 | 降级选项 B |
| "分组信息缺失"/"groups"/"missing groups" | 参数不足 | 降级选项 C |
| 超时无输出 / timeout | 执行复杂度 | 降级选项 D |
| 其他 | 未知 | 降级选项 E |

## 降级选项

### A. 能力边界（范式不支持）

```python
ask_clarification(
    question="该范式的自动分析流程尚未完善（当前支持：shoaling, epm, open_field, ...）。可选方案：",
    clarification_type="approach_choice",
    context="code-executor 返回：<原错误信息>",
    options=[
        "尝试基础指标计算（移动距离、区域停留时间等，可能不完整）",
        "展示数据结构，我来指定分析内容",
        "暂时跳过"
    ]
)
```

### B. 数据格式问题

```python
ask_clarification(
    question="数据文件解析失败：<具体错误>。可能原因：",
    clarification_type="missing_info",
    context="code-executor 返回：<原错误信息>",
    options=[
        "文件不是 EthoVision XT 导出格式 → 重新导出",
        "编码问题（UTF-16 等）→ 尝试重新读取",
        "文件损坏 → 重新上传"
    ]
)
```

### C. 参数不足

重新检查分组定义，通常是 lead agent 自己的 prompt 传递问题。回到 Step 2 重新询问分组。

### D. 执行超时

```python
ask_clarification(
    question="分析耗时较长，可能因为样本量大或数据复杂。可选方案：",
    clarification_type="approach_choice",
    context="code-executor 超时",
    options=[
        "继续等待（再次派遣，给更长超时）",
        "简化分析（只跑核心指标，跳过复杂统计）",
        "分批分析（先做对照组）"
    ]
)
```

### E. 未知错误

```python
ask_clarification(
    question="分析遇到未预期的问题：<错误信息>。需要您帮助判断下一步：",
    clarification_type="approach_choice",
    context="code-executor 未知错误",
    options=[
        "重试",
        "跳过这次分析",
        "联系技术支持"
    ]
)
```

## 绝对禁止

- ❌ 同一轮对话中重新派遣 code-executor 执行相同范式（用户明确指示除外）
- ❌ 自己用 bash/read_file 替代 code-executor 完成整个分析流程
- ❌ 假设"换个参数"能解决范式不支持的问题
- ❌ 不告知用户就静默重试

## 连续失败处理

- 同一范式连续失败 2 次 → 必须放弃当前路径，询问用户整体方向
- 不同 subagent 连续失败 2 次 → 必须 `ask_clarification`，不能继续盲目流水线
```

**Step 2: 落盘**

---

## Task 8: 注册 skill 到 `extensions_config.json`

**Files:**
- Modify: `packages/agent/extensions_config.json`

**Step 1: 读取现有配置**

Run: `cat packages/agent/extensions_config.json`

**Step 2: 在 skills 对象中添加 ethoinsight-planning**

用 Edit 工具在 [extensions_config.json](packages/agent/extensions_config.json) 中 `"ethoinsight": { "enabled": true },` 下方插入：

```json
    "ethoinsight-planning": { "enabled": true },
```

**Step 3: 验证 JSON 仍合法**

Run: `python3 -c "import json; json.load(open('packages/agent/extensions_config.json'))" && echo OK`

Expected: `OK`

**Step 4: 验证 skill 被加载为 enabled**

Run:
```bash
cd packages/agent/backend && source .venv/bin/activate
python -c "from deerflow.skills import load_skills; skills = list(load_skills(enabled_only=True)); names = [s.name for s in skills]; print('ethoinsight-planning' in names)"
```

Expected: `True`

**Step 5: 提交**

```bash
git add packages/agent/extensions_config.json
git commit -m "feat: 注册 ethoinsight-planning skill"
```

---

## Task 9: 在 lead_agent prompt 中加入规划核心原则（Layer 1）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

**Step 1: 确定插入位置**

Layer 1 的核心原则应放在 `orchestration_guide` 的开头（[prompt.py:866-941](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L866-L941)），在"Step 0: 确认需求"之前，作为"规划先于派遣"的总原则。

**Step 2: 用 Edit 工具插入新章节**

在 [prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 中，找到：

```python
    if subagent_enabled and has_noldus_agents:
        orchestration_guide = """<orchestration_guide>
## EthoVision 数据分析派遣流程

当用户上传 EthoVision 数据并请求分析时，按以下流程派遣 subagent：
```

替换为：

```python
    if subagent_enabled and has_noldus_agents:
        orchestration_guide = """<orchestration_guide>
## 规划先于派遣（MANDATORY）

当本轮消息 `<uploaded_files>` 包含新上传的数据文件 **且** 用户请求分析/统计/可视化/报告时，
你 **必须** 先加载 `ethoinsight-planning` skill 并按它的流程规划：

1. **立即调用**: `read_file("/mnt/skills/ethoinsight-planning/SKILL.md")`
2. **遵循 6 步规划流程**: 意图分类 → 完整性检查 → 选模板 → 质量门控 → 单行摘要 → 执行
3. **仅两种情况必须反问用户**:
   - 范式推断失败（文件名看不出范式）
   - 分组无法推断（无命名规律且用户未明示）
   - 其他情况走默认，**不要过度反问**
4. **输出单行计划给用户**（格式：`将对 <范式> 数据执行 <操作>，约 X 分钟`）
5. **执行时遵循本文档后续的派遣流程**

**跳过规划的场景**（直接派遣 knowledge-assistant）：
- 无新上传文件 + 追问已有结果或概念问题
- 用户闲聊、确认、感谢

**规划本身不占用 `task` 调用配额**——它只是读 skill + 可能的 `ask_clarification`。

## EthoVision 数据分析派遣流程

当用户上传 EthoVision 数据并请求分析时，按以下流程派遣 subagent：
```

**Step 3: 跑单元测试确保 prompt 仍能正确生成**

Run:
```bash
cd packages/agent/backend && source .venv/bin/activate
python -c "from deerflow.agents.lead_agent.prompt import apply_prompt_template; p = apply_prompt_template(subagent_enabled=True); assert '规划先于派遣' in p; assert 'ethoinsight-planning' in p; print('OK')"
```

Expected: `OK`

**Step 4: 跑全部测试确保无回归**

Run: `cd packages/agent/backend && source .venv/bin/activate && make test 2>&1 | tail -5`

Expected: 测试全部通过，无新增失败。

**Step 5: 提交**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "feat: 在 lead_agent prompt 加入规划先于派遣原则"
```

---

## Task 10: 写单元测试验证 skill 加载

**Files:**
- Create: `packages/agent/backend/tests/test_ethoinsight_planning_skill.py`

**Step 1: 写失败的测试**

Create `packages/agent/backend/tests/test_ethoinsight_planning_skill.py`:

```python
"""Tests for ethoinsight-planning skill integration."""

import json
from pathlib import Path

from deerflow.skills import load_skills


def _find_skill(name: str):
    for skill in load_skills(enabled_only=False):
        if skill.name == name:
            return skill
    return None


def test_planning_skill_is_discovered():
    """ethoinsight-planning skill should be discoverable by skills loader."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None, "ethoinsight-planning skill not found in skills/"


def test_planning_skill_is_enabled_in_config():
    """extensions_config.json should enable ethoinsight-planning."""
    repo_root = Path(__file__).resolve().parents[4]
    config_path = repo_root / "extensions_config.json"
    assert config_path.exists(), f"extensions_config.json not found at {config_path}"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    skill_state = config.get("skills", {}).get("ethoinsight-planning")
    assert skill_state is not None, "ethoinsight-planning not registered in extensions_config.json"
    assert skill_state.get("enabled") is True, "ethoinsight-planning must be enabled"


def test_planning_skill_has_required_references():
    """Planning skill should ship 5 reference files for progressive loading."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None

    skill_dir = Path(skill.path).parent
    references_dir = skill_dir / "references"
    assert references_dir.is_dir(), f"references/ directory missing at {references_dir}"

    required = {
        "intent-classification.md",
        "planning-templates.md",
        "design-type-keywords.md",
        "quality-gates.md",
        "failure-recovery.md",
    }
    actual = {p.name for p in references_dir.iterdir() if p.is_file()}
    missing = required - actual
    assert not missing, f"Missing reference files: {missing}"


def test_planning_skill_description_mentions_trigger_conditions():
    """SKILL.md description should state when to load (uploaded data + analysis request)."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None

    description = skill.description.lower()
    assert "uploaded" in description or "data" in description
    assert "analysis" in description or "analyze" in description
```

**Step 2: 运行测试确认失败（如果前面任何步骤没做完）或通过**

Run:
```bash
cd packages/agent/backend && source .venv/bin/activate
pytest tests/test_ethoinsight_planning_skill.py -v
```

Expected:
- 如果 Task 2-8 都已完成 → 全部 PASS
- 如果有缺失 → 相应测试 FAIL，按错误信息回查

**Step 3: 若失败，回查并修复**

常见问题：
- `references/` 目录路径错误 → 查 Task 2-7 的文件路径
- `skill.path` 结构与假设不同 → 读 `deerflow.skills.types.Skill` 源码调整断言

**Step 4: 提交**

```bash
git add packages/agent/backend/tests/test_ethoinsight_planning_skill.py
git commit -m "test: 覆盖 ethoinsight-planning skill 加载与配置"
```

---

## Task 11: 写单元测试验证 prompt 包含规划指令

**Files:**
- Create: `packages/agent/backend/tests/test_lead_agent_planning_prompt.py`

**Step 1: 写测试**

Create `packages/agent/backend/tests/test_lead_agent_planning_prompt.py`:

```python
"""Tests for planning instructions in lead_agent prompt."""

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.subagents import get_available_subagent_names


def test_prompt_contains_planning_directive_when_noldus_agents_present():
    """Prompt should mention ethoinsight-planning skill when Noldus subagents are registered."""
    names = set(get_available_subagent_names())
    has_noldus = bool({"code-executor", "data-analyst", "report-writer", "knowledge-assistant"} & names)
    if not has_noldus:
        import pytest
        pytest.skip("Noldus subagents not registered in this environment")

    prompt = apply_prompt_template(subagent_enabled=True)

    assert "ethoinsight-planning" in prompt, "Prompt must reference the planning skill"
    assert "规划先于派遣" in prompt, "Prompt must enforce planning before delegation"


def test_prompt_lists_mandatory_clarification_cases():
    """Prompt should state only paradigm and group inference failures require clarification."""
    names = set(get_available_subagent_names())
    has_noldus = bool({"code-executor", "data-analyst", "report-writer", "knowledge-assistant"} & names)
    if not has_noldus:
        import pytest
        pytest.skip("Noldus subagents not registered in this environment")

    prompt = apply_prompt_template(subagent_enabled=True)

    assert "范式推断失败" in prompt
    assert "分组无法推断" in prompt


def test_prompt_without_subagents_has_no_planning_directive():
    """When subagents are disabled, planning directive should not appear."""
    prompt = apply_prompt_template(subagent_enabled=False)
    assert "规划先于派遣" not in prompt
```

**Step 2: 运行测试**

Run: `cd packages/agent/backend && source .venv/bin/activate && pytest tests/test_lead_agent_planning_prompt.py -v`

Expected: 3 个测试全部 PASS

**Step 3: 提交**

```bash
git add packages/agent/backend/tests/test_lead_agent_planning_prompt.py
git commit -m "test: 覆盖 lead_agent prompt 中的规划指令"
```

---

## Task 12: 跑全量测试确保无回归

**Step 1: 运行 backend 全部测试**

Run:
```bash
cd packages/agent/backend && source .venv/bin/activate && make test 2>&1 | tail -20
```

Expected: 所有测试通过，包括新增的 2 个测试文件。

**Step 2: 运行 ethoinsight 测试**

Run:
```bash
cd packages/ethoinsight && source ../agent/backend/.venv/bin/activate && pytest tests/ -v 2>&1 | tail -15
```

Expected: 全部通过（此次改动不触及 ethoinsight 库）。

**Step 3: ruff 检查**

Run: `cd packages/agent/backend && source .venv/bin/activate && make lint 2>&1 | tail -5`

Expected: 无新增 warning 或 error。

**Step 4: 如果有失败，回查并修复**

**Step 5: 如已跑绿，无需额外提交**

---

## Task 13: 更新 CLAUDE.md 和 docs/roadmap.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/roadmap.md`

**Step 1: CLAUDE.md — 在 "项目定制 skill" 相关段落提及新 skill**

用 Edit 工具修改 [CLAUDE.md](CLAUDE.md)：

找到 `skills/custom/ 是 gitignored` 条目附近（约在"重要注意事项"章节），在合适位置补充：

在 `- **skills/custom/ 是 gitignored**` 这一行之后加入：

```
   - 项目定制 skill 包括：`ethoinsight`, `ethoinsight-analysis`, `ethoinsight-charts`, `ethoinsight-planning`（规划框架，2026-04-17 新增）
```

（Edit old_string 要精确匹配现有那行原文，参照 Task 开始前 Read CLAUDE.md 的结果。）

**Step 2: docs/roadmap.md — 在 Phase 0 中提及 planning skill**

用 Edit 工具修改 [docs/roadmap.md](docs/roadmap.md)，在 Phase 0 的 "M0.1 鲁棒性验证" 或 "M0.4 基础设施" 里合适位置补一条：

```
- **ethoinsight-planning skill**: 端到端分析的规划框架（3 层：prompt 核心原则 + skill 模板 + TodoList 状态机），2026-04-17 实装
```

**Step 3: 提交**

```bash
git add CLAUDE.md docs/roadmap.md
git commit -m "docs: 登记 ethoinsight-planning skill 到 CLAUDE.md 和 roadmap"
```

---

## Task 14: 写交接文档

**Files:**
- Create: `docs/handoffs/2026-04-17-planning-skill-implementation-handoff.md`

**Step 1: 写交接文档**

Create the handoff with sections:
- 当前任务目标
- 当前进展（表格列出 14 个 Task）
- 本次改动的文件列表
- 关键架构决策（引用 brainstorming 的 7 个设计问题）
- 未完成事项（E2E 手动验证 + 可能的 prompt 精简）
- 建议接手路径（手动 E2E 测试 4 个场景）
- 风险与注意事项（GLM-5.1 是否会稳定调用 skill）

格式参考 [docs/handoffs/2026-04-17-roadmap-and-strategy-handoff.md](docs/handoffs/2026-04-17-roadmap-and-strategy-handoff.md)。

**Step 2: 提交**

```bash
git add docs/handoffs/2026-04-17-planning-skill-implementation-handoff.md
git commit -m "docs: 添加规划 skill 实装交接文档"
```

---

## 验收（用户手动 E2E 测试指南）

实装完成后，用户手动在 agent UI 中测试以下 4 个场景：

### 场景 1: 标准端到端分析

1. 启动 `cd packages/agent && make dev`
2. 打开 localhost:2026
3. 上传 `demo-data/DemoData/<shoaling 相关文件>` 下 5 个轨迹文件
4. 发送："帮我分析这批 shoaling 数据，Subject 1-2 是对照组，Subject 3-5 是实验组"

**预期行为**：
- Agent 输出一句话计划："将对 Shoaling 数据执行群聚性分析 + 生成对比报告，约 2 分钟"
- 随后派遣 code-executor → data-analyst → report-writer
- 最终给出报告

### 场景 2: 范式推断失败

1. 上传文件名为 `Subject 1.txt` ~ `Subject 6.txt` 的数据（无范式关键词）
2. 发送："帮我分析一下"

**预期行为**：
- Agent 调用 `ask_clarification`，问"请问这批数据来自哪种实验范式？"
- 给出 options（EPM / OFT / Shoaling / 其他）
- **不**自己开始执行

### 场景 3: 小样本量提醒

1. 上传 2 个文件（每组 1 只）
2. 发送："分析这批 EPM 数据，Subject 1 对照，Subject 2 实验"

**预期行为**：
- Agent 计划摘要明示："样本量较小（每组 1 只），将仅做描述性统计"
- `ask_clarification` 询问是否继续

### 场景 4: 纯知识问答（不规划）

1. 不上传文件
2. 发送："什么是 NND？"

**预期行为**：
- Agent **不**加载 ethoinsight-planning skill
- 直接派遣 knowledge-assistant
- 返回 NND 的定义

### 场景 5（可选）: 手动开启 Plan Mode

1. 前端 UI 切换 Plan Mode 开关（DeerFlow 原生按钮）
2. 重复场景 1
3. **预期**：规划产物以 TodoList 形式可见（3 条：数据分析/解读/报告），每步完成后状态更新

---

## 执行交接

Plan complete and saved to `docs/plans/2026-04-17-ethoinsight-planning-skill-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
