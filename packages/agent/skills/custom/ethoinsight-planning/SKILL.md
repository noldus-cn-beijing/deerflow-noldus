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

# 分组 label 通用但未提供处理描述
ask_clarification(
    question="请问实验组（Treatment）对应的具体处理是什么？例如药物剂量、造模类型、基因型差异、年龄等。",
    clarification_type="missing_info",
    context="分组 label 为通用名，需要具体处理描述以指导 Discussion 段解读",
    options=None  # 开放式，无预设选项
)
```

**session 内去重**：用户回答后，把处理描述写入 handoff_planning.json 的 `group_semantics` 字段（格式：`{"control": "saline", "treatment": "30 mg/kg fluoxetine"}`）。下次 planning 先读该字段，已有则跳过追问。

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
| data-analyst 超时/空返回 | 跳过，lead 自己 read_file handoff_code_executor.json，把统计摘要（metrics_summary + statistics）转述给用户 |
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
