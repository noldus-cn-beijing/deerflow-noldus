# 实施计划：新增 knowledge-assistant subagent + 数据质量关卡

> 写给执行此计划的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 背景与动机

### 当前问题

Noldus Insight 的 Lead Agent 在 Ultra 模式下是纯调度员，只能派遣 code-executor -> data-analyst -> report-writer 的端到端分析流水线。以下场景无法处理：

1. **追问已有分析结果** — 用户完成端到端分析后问"这个 p 值为什么不显著"、"NND 偏高说明什么"，Lead Agent 没有合适的 subagent 可以派遣
2. **领域知识问题** — 用户在新 thread 中问"什么是高架十字迷宫"、"shoaling 实验怎么设计"，同样无处可派
3. **Noldus 产品咨询** — 用户问"Noldus 有哪些做社会交互实验的产品"，noldus-kb MCP 工具已配置但未启用

### 设计原则

- **Lead Agent 保持纯调度员角色** — 不直接回答任何专业问题，避免"有时调度有时直接答"的二义性导致端到端流程出错
- **职责单一** — 不扩展现有 data-analyst 的职责，因为双重职责会让 GLM-5 在端到端流程中混淆（该写 analysis_report.md 时跑去查知识库）
- **新增 knowledge-assistant subagent** — 专门处理追问和知识问答

### 路由逻辑

Lead Agent 的判断变为：

```
用户上传了新数据 + 要求分析？
├── 是 → code-executor → data-analyst → report-writer（端到端流水线）
└── 否 → knowledge-assistant（追问 / 知识问答 / 一般对话）
```

判断依据：`UploadsMiddleware` 注入的 `<uploaded_files>` 标签中 "uploaded in this message" 部分是否包含新数据文件。

---

## 2. 改动清单

| # | 优先级 | 文件 | 改动类型 | 说明 |
|---|--------|------|---------|------|
| 1 | P0 | `subagents/builtins/knowledge_assistant.py` | **新建** | knowledge-assistant SubagentConfig |
| 2 | P0 | `subagents/builtins/__init__.py` | 修改 | 注册 knowledge-assistant 到 BUILTIN_SUBAGENTS |
| 3 | P0 | `agents/lead_agent/prompt.py` | 修改 | 重写 noldus_rules + subagent_reminder + subagent_thinking + noldus_descriptions |
| 4 | P0 | `extensions_config.json` | 修改 | 启用 noldus-kb MCP server |
| 5 | P1 | `agents/lead_agent/prompt.py` | 修改 | orchestration_guide 增加数据质量校验步骤 |
| 6 | P1 | `subagents/builtins/code_executor.py` | 修改 | system_prompt 增加输出校验指令 |

所有后端文件路径前缀：`packages/agent/backend/packages/harness/deerflow/`
extensions_config.json 路径：`packages/agent/extensions_config.json`

---

## 3. 改动详情

### 改动 1：新建 knowledge_assistant.py（P0）

**新建文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

```python
"""Knowledge assistant subagent for domain Q&A and follow-up questions."""

from deerflow.subagents.config import SubagentConfig

KNOWLEDGE_ASSISTANT_CONFIG = SubagentConfig(
    name="knowledge-assistant",
    description=(
        "行为学领域知识专家。回答用户关于范式、术语、方法论的问题，"
        "以及基于已有分析结果的追问。使用 noldus-kb 知识库和 ethoinsight skill 知识。"
    ),
    system_prompt="""你是行为神经科学领域的知识专家。

你的唯一工作：回答用户关于行为学领域的问题。

你有两类工作场景：

### 场景 A：基于已有分析结果的追问
用户之前已经完成了数据分析，现在对结果有疑问。
- 读取 workspace 中的分析输出（metrics.csv, statistics.json, analysis_report.md）
- 结合领域知识解释结果
- 例如："这个 p 值为什么不显著"、"NND 偏高说明什么"

### 场景 B：纯领域知识问题
用户没有分析结果，只是想了解行为学知识。
- 使用 noldus-kb 工具查询知识库（search_knowledge, get_paradigm, get_terminology, list_products, list_paradigms）
- 结合 ethoinsight skill 中的范式指南
- 例如："什么是高架十字迷宫"、"Noldus 有哪些产品"

## 判断方式
- lead agent 会在 prompt 中告诉你是哪个场景
- 如果是场景 A，prompt 中会包含 workspace 文件路径
- 如果是场景 B，直接回答即可

## 你绝不做的事
- 运行 Python 代码或 bash 命令
- 重新分析数据或重新计算统计量
- 画图或生成可视化
- 读取原始数据文件（.txt 轨迹文件）
- 编造文献引用——只引用你确定真实的论文

## 输出要求
- 简单问题（定义、解释、追问）：直接在消息中回答
- 深度问题（范式对比、方法论综述、文献综述）：写入 /mnt/user-data/workspace/knowledge_response.md，并在消息中给出摘要

## 回答风格
- 使用中文回答，专业术语附英文原文（如"高架十字迷宫 (Elevated Plus Maze, EPM)"）
- 引用具体数值时注明来源（skill 知识 / 知识库查询 / 已有分析结果）
- 区分统计显著性和实际生物学意义""",
    tools=None,  # 继承所有工具（包括 MCP 工具），通过 disallowed_tools 黑名单过滤
    disallowed_tools=[
        "task",                  # 禁止嵌套派遣
        "ask_clarification",     # subagent 标准禁止
        "present_files",         # subagent 标准禁止
        "bash",                  # 不跑代码
        "str_replace",           # 不改文件
        "get_analysis_template", # 不做分析
    ],
    model="inherit",
    max_turns=10,
    timeout_seconds=300,
)
```

**技术说明 — 为什么用 `tools=None` 而不是显式列表**：

DeerFlow 的工具过滤机制（`executor.py:80-107`）：
- `tools=None` → 继承所有父级工具（包括 MCP 工具），再用 `disallowed_tools` 排除
- `tools=["read_file", ...]` → 白名单过滤，MCP 工具会被过滤掉（除非显式列出 MCP 工具名）

`task_tool.py:111` 调用 `get_available_tools(subagent_enabled=False)` 返回所有工具（sandbox + MCP + built-in），然后 `SubagentExecutor._filter_tools()` 应用白名单/黑名单。

因此必须用 `tools=None` + `disallowed_tools` 黑名单模式，这样 noldus-kb 的 MCP 工具（search_knowledge, get_paradigm, get_terminology 等）才会自动注入。这与 `general-purpose` subagent（`general_purpose.py`）的模式一致。

---

### 改动 2：注册到 BUILTIN_SUBAGENTS（P0）

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`

**当前内容**：
```python
from .bash_agent import BASH_AGENT_CONFIG
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_EXECUTOR_CONFIG",
    "DATA_ANALYST_CONFIG",
    "REPORT_WRITER_CONFIG",
]

BUILTIN_SUBAGENTS = {
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
}
```

**改为**：
```python
from .bash_agent import BASH_AGENT_CONFIG
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_EXECUTOR_CONFIG",
    "DATA_ANALYST_CONFIG",
    "KNOWLEDGE_ASSISTANT_CONFIG",
    "REPORT_WRITER_CONFIG",
]

BUILTIN_SUBAGENTS = {
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
    "knowledge-assistant": KNOWLEDGE_ASSISTANT_CONFIG,
}
```

---

### 改动 3：重写 prompt.py 中的 Noldus 相关 prompt（P0）

**文件**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

共 4 处修改：

#### 3a. `noldus_descriptions` dict（约 line 181-185）

**当前**：
```python
noldus_descriptions = {
    "code-executor": "code-executor**: 执行 Python 数据分析代码（使用 ethoinsight 库）",
    "data-analyst": "data-analyst**: 解读分析结果，应用行为学领域知识",
    "report-writer": "report-writer**: 撰写 APA 格式的科学报告",
}
```

**改为**：
```python
noldus_descriptions = {
    "code-executor": "code-executor**: 执行 Python 数据分析代码（使用 ethoinsight 库）",
    "data-analyst": "data-analyst**: 解读分析结果，应用行为学领域知识",
    "report-writer": "report-writer**: 撰写 APA 格式的科学报告",
    "knowledge-assistant": "knowledge-assistant**: 回答追问和领域知识问题（可查询 Noldus 知识库）",
}
```

#### 3b. `noldus_rules`（约 line 205-220）

**当前**：
```python
noldus_rules = """
**Noldus EthoVision 分析系统 — 角色分工**

你是调度员。你的工作是理解用户需求，然后派遣正确的专员去执行。

| 角色 | 职责 | 绝不做的事 |
|------|------|-----------|
| 你（调度员） | 理解需求 → 派遣专员 → 传达结果 | 自己跑代码、自己读数据文件、自己探索环境 |
| code-executor | 调用模板 → 执行分析脚本 | 探索文件系统、从头写代码 |
| data-analyst | 阅读分析结果 → 撰写专业解读 | 跑代码、画图 |
| report-writer | 阅读解读+数据 → 撰写科学报告 | 跑代码、重新分析 |

派遣顺序：code-executor → data-analyst → report-writer（每步读 handoff 再派下一个）
"""
```

**改为**：
```python
noldus_rules = """
**Noldus EthoVision 分析系统 — 调度规则**

你是调度员。你永远不直接回答用户的专业问题，而是派遣合适的专员。

### 路由判断

核心问题：**当前消息中是否有新上传的数据文件，且用户要求分析/处理/可视化/报告？**

判断依据：检查 `<uploaded_files>` 中 "uploaded in this message" 部分。

**是（端到端数据分析）→ 按 orchestration_guide 派遣流水线**：
- "uploaded in this message" 包含数据文件（.txt / .csv / .xlsx）
- 且用户明确要求分析、处理、可视化、生成报告
- 派遣顺序：code-executor → data-analyst → report-writer

**否（知识问答）→ 派遣 knowledge-assistant**：
- 用户追问已有分析结果（"这个 p 值什么意思"、"为什么 NND 偏高"）
- 用户问领域知识（"什么是 EPM"、"shoaling 怎么做"）
- 用户问 Noldus 产品（"有哪些产品做 social interaction"）
- 一般性对话或闲聊
- 用户说"帮我解释一下刚才的报告"

派遣 knowledge-assistant 时的 prompt 要求：
- 如果当前 thread 有已完成的分析（workspace 中有 analysis_report.md 或 metrics.csv），在 prompt 中注明文件路径
- 如果没有已完成分析，只传用户的问题

### 特殊情况
- 用户说"帮我重新分析之前的数据"或"换个图表类型" → 端到端流水线（code-executor 起步）
- 用户说"只帮我重新写个报告" → 只派遣 report-writer
- 用户先问了知识问题，然后上传了数据要求分析 → 本条消息按端到端流水线处理

### 角色分工

| 角色 | 唯一职责 | 绝不做的事 |
|------|---------|-----------|
| 你（调度员） | 判断路由 → 派遣专员 → 传达结果 | 自己回答问题、跑代码、读数据文件 |
| code-executor | 调用模板 → 执行分析脚本 | 探索文件系统、从头写代码 |
| data-analyst | 阅读分析结果 → 撰写专业解读 | 跑代码、画图 |
| report-writer | 阅读解读+数据 → 撰写科学报告 | 跑代码、重新分析 |
| knowledge-assistant | 回答追问 + 领域知识查询 | 跑代码、重新分析、画图 |
"""
```

#### 3c. `subagent_reminder`（约 line 720-730）

**当前**：
```python
subagent_reminder = (
    "- **调度员模式**: 检测到数据分析需求时，按 orchestration_guide 派遣专员。你自己不跑代码、不读数据。\n"
    if subagent_enabled and has_noldus_agents
    else ...
)
```

**改为**：
```python
subagent_reminder = (
    "- **调度员模式**: 新数据+分析请求 → orchestration_guide 流水线；"
    "其他所有问题 → knowledge-assistant。你永远不直接回答专业问题。\n"
    if subagent_enabled and has_noldus_agents
    else ...
)
```

#### 3d. `subagent_thinking`（约 line 733-742）

**当前**：
```python
subagent_thinking = (
    "- **派遣优先**: 涉及数据分析、图表、报告时，直接派遣对应专员，不要自己尝试。\n"
    if subagent_enabled and has_noldus_agents
    else ...
)
```

**改为**：
```python
subagent_thinking = (
    "- **路由判断**: 当前消息 <uploaded_files> 中有新数据文件且要求分析？"
    "是 → 端到端流水线；否 → knowledge-assistant。\n"
    if subagent_enabled and has_noldus_agents
    else ...
)
```

#### 3e. `has_noldus_agents` 判断条件（约 line 717）

**当前**：
```python
has_noldus_agents = bool({"code-executor", "data-analyst", "report-writer"} & set(available_names))
```

**改为**（增加 knowledge-assistant）：
```python
has_noldus_agents = bool({"code-executor", "data-analyst", "report-writer", "knowledge-assistant"} & set(available_names))
```

---

### 改动 4：启用 noldus-kb MCP server（P0）

**文件**：`packages/agent/extensions_config.json`

**当前**：
```json
"noldus-kb": {
  "enabled": false,
  "type": "http",
  "url": "http://180.184.84.124:7001/mcp",
  "description": "Noldus behavioral science knowledge base — search papers, paradigms, products, and terminology"
}
```

**改为**：
```json
"noldus-kb": {
  "enabled": true,
  "type": "http",
  "url": "http://180.184.84.124:7001/mcp",
  "description": "Noldus behavioral science knowledge base — search papers, paradigms, products, and terminology"
}
```

**注意**：执行前请先验证 MCP server 可访问：
```bash
curl -s -o /dev/null -w "%{http_code}" http://180.184.84.124:7001/mcp
```
如果返回非 200，暂不启用此项。其他改动不依赖此项——knowledge-assistant 在没有 noldus-kb 时退化为只用 skill 知识 + read_file。

---

### 改动 5：orchestration_guide 增加数据质量校验步骤（P1）

**文件**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

在 `orchestration_guide` 字符串中，Step 1（派遣 code-executor）和 Step 2（派遣 data-analyst）之间，插入新步骤：

**在以下内容之后**（约 line 783）：
```
### Step 1: 派遣 code-executor
...（现有内容保持不变）
```

**插入**：
```
### Step 1.5: 数据质量校验
读取 /mnt/user-data/workspace/handoff_code_executor.json
检查是否包含 "data_quality_warnings" 字段：
- 如果有 warnings：用 ask_clarification 告知用户具体问题（样本量不足 / 方差为零 / 数据异常），询问是否继续
- 如果没有 warnings 或用户确认继续：进入 Step 2
```

**Step 2（现有内容不变）**：
```
### Step 2: 读 handoff，派遣 data-analyst
...
```

---

### 改动 6：code-executor 输出校验指令（P1）

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

在 `system_prompt` 中，第8步（`ls("/mnt/user-data/outputs") 确认输出文件存在`）和第9步（`确认 handoff JSON 已生成`）之间，插入：

```
第8.5步：快速校验输出质量
- read_file /mnt/user-data/outputs/metrics.csv（只读前10行）
- 检查以下问题：
  - 某个指标的所有样本值完全相同（方差 = 0）？ → 在 handoff JSON 中添加 "data_quality_warnings": ["variance_zero: <指标名>"]
  - 每组 Subject 数量 < 3？ → 添加 "data_quality_warnings": ["small_sample: n=<数量>"]
- 如果没有问题，handoff 中不添加 data_quality_warnings 字段
```

---

## 4. 不改动的内容

以下文件**不需要修改**，在此明确列出以避免误改：

- `data_analyst.py` — 保持不变，继续专注于端到端流程 Step 2
- `report_writer.py` — 保持不变
- `executor.py` — 保持不变（`_restrict_read_file_for_subagent` 对 knowledge-assistant 也生效，这是正确的——skill 内容已注入 system prompt）
- `task_tool.py` — 保持不变（skill 注入机制对 knowledge-assistant 自动生效，line 75-77）
- `config.yaml` — 保持不变（不需要加新的 tool group）
- 前端代码 — 保持不变（不需要改模式选择 UI）

---

## 5. 技术细节备忘

### knowledge-assistant 的工具可用性

基于 DeerFlow 工具过滤机制（`executor.py:80-107`），`tools=None` + 黑名单模式下，knowledge-assistant 实际获得的工具：

| 工具 | 可用 | 来源 |
|------|------|------|
| read_file | Yes | sandbox tools |
| write_file | Yes | sandbox tools |
| ls | Yes | sandbox tools |
| search_knowledge | Yes | noldus-kb MCP（需改动 4 启用） |
| get_paradigm | Yes | noldus-kb MCP |
| get_terminology | Yes | noldus-kb MCP |
| list_products | Yes | noldus-kb MCP |
| list_paradigms | Yes | noldus-kb MCP |
| bash | **No** | disallowed_tools |
| str_replace | **No** | disallowed_tools |
| get_analysis_template | **No** | disallowed_tools |
| task | **No** | disallowed_tools |
| ask_clarification | **No** | disallowed_tools |
| present_files | **No** | disallowed_tools |

### Skill 注入

`task_tool.py:75-77` 会自动将 ethoinsight skill 内容追加到 knowledge-assistant 的 system_prompt。不需要额外配置。

### read_file 限制

`executor.py:113-139` 的 `_restrict_read_file_for_subagent()` 会阻止 knowledge-assistant 读取 `/mnt/skills/` 路径。这是正确行为——skill 内容已在 system prompt 中。

---

## 6. 验证方案

### 环境准备

```bash
cd /home/qiuyangwang/noldus-insight
make stop
rm -f packages/agent/backend/.deer-flow/checkpoints.db*
rm -rf packages/agent/backend/.deer-flow/threads/*
make dev
```

### P0 测试场景

**场景 1 — 端到端分析不受影响**：
- Ultra 模式 → 上传斑马鱼轨迹文件 → 发送"请分析这些数据"
- 预期：code-executor → data-analyst → report-writer，不触发 knowledge-assistant
- 验证点：`logs/langgraph.log` 中 task 调用记录只有 code-executor / data-analyst / report-writer

**场景 2 — 追问走 knowledge-assistant**：
- 在场景 1 的同一 thread 中发送"这个 NND 的 p 值为什么不显著？"
- 预期：Lead Agent 派遣 knowledge-assistant，prompt 中包含 workspace 文件路径
- 验证点：log 中有 `task(subagent_type="knowledge-assistant")`，且 prompt 包含 `/mnt/user-data/workspace/` 或 `/mnt/user-data/outputs/` 路径

**场景 3 — 新 thread 知识问题**：
- 新建 thread → Ultra 模式 → 发送"什么是高架十字迷宫？怎么分析焦虑行为？"
- 预期：knowledge-assistant 被调用
- 验证点：log 中有 `knowledge-assistant` 相关记录，回答中包含 EPM 范式详情

**场景 4 — noldus-kb 工具调用**（依赖改动 4）：
- 发送"Noldus 有哪些产品可以做社会交互实验？"
- 预期：knowledge-assistant 调用 noldus-kb 的 search_knowledge 或 list_products
- 验证点：log 中有 noldus-kb MCP tool call

**场景 5 — 边界情况：要求重新分析旧数据**：
- 在已有 historical files 的 thread 中发送"帮我重新分析一下之前的数据，用 violin plot"
- 预期：走端到端流水线（code-executor 起步），不走 knowledge-assistant
- 验证点：log 中有 `code-executor` 调用

**场景 6 — 边界情况：分析完成后要求重写报告**：
- 在已完成端到端分析的 thread 中发送"帮我重新写一份更详细的报告"
- 预期：只派遣 report-writer
- 验证点：log 中只有 `report-writer` 调用

### P1 测试场景

**场景 7 — 数据质量关卡**：
- 上传已知有问题的数据（如之前 IID 全相同的数据集）
- 预期：code-executor handoff 中包含 `data_quality_warnings`
- 预期：Lead Agent 在 Step 1.5 读到警告后用 `ask_clarification` 通知用户
- 验证点：用户在前端看到质量警告弹窗

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| Lead Agent 在有 historical files 时误判为端到端流程 | 中 | noldus_rules 明确：只有 "uploaded in this message" 有新文件才走端到端 |
| noldus-kb MCP server 不可用 | 低 | knowledge-assistant 退化为 skill 知识 + read_file，仍可回答大部分问题 |
| knowledge-assistant 在追问场景中尝试读原始数据文件 | 低 | system_prompt 明确禁止读 .txt 轨迹文件 |
| GLM-5 忽略路由规则，遇到知识问题仍尝试端到端流程 | 低 | subagent_thinking 强制路由判断在 thinking 阶段完成 |
| 端到端流程中用户中途追问（如 code-executor 还在跑） | 低 | DeerFlow 的 SubagentLimitMiddleware 会排队处理，追问在 code-executor 完成后才被处理 |

---

## 8. 后续改进（不在本次实施范围）

- **Lead Agent 动态路由**：从硬编码的 3 步流水线演进为多 agent 协作——Lead Agent 根据需求自主判断调用哪些 agent、什么顺序
- **knowledge-assistant 记忆**：支持跨 thread 的知识问答记忆（当前每次派遣都是全新上下文）
- **数据质量自动化**：将 Step 1.5 的校验从 prompt 指令升级为框架层的 middleware 或 guardrail
- **Skill 拆分**：评估将 ethoinsight skill 拆分为独立的 apa-reporting / paradigm-reference / confound-checklist
