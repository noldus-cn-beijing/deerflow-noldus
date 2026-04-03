# EthoInsight：基于 DeerFlow Fork 的二次开发计划

## Context

EthoInsight 是一个垂直领域的 AI 数据分析 Agent。

DeerFlow 的配置驱动模式能做的事：注册自定义工具（config.yaml）、注入 Skills prompt（skills/custom/）、接入 GLM-5.1（config.yaml models）、调用 noldus-kb（Agent 内置 bash 工具）。

但配置驱动做不到的事也不少。选择 fork 有两方面原因：

**前端：** DeerFlow 自带通用 Agent 对话界面，不符合 Noldus 商业产品的需要。我们需要定制数据上传界面、实验配置表单、品牌化的报告展示页面。

**后端：** DeerFlow 的 System Prompt 基础角色定义是硬编码的、中间件链是有序硬编码列表不支持插件式扩展、ThreadState 没有领域字段、文件上传不认识 EthoVision 格式、记忆提取的 fact prompt 是通用的、子 Agent 注册表只有两种内置类型。这些都需要改源码。

DeerFlow 提供的基础设施（LangGraph Agent 循环、Docker 沙箱、Session 记忆、前端骨架）仍然是我们的起点。

## Fork 策略

```bash
git clone https://github.com/bytedance/deer-flow.git noldus-insight-agent
cd noldus-insight-agent
git remote rename origin upstream   # 保留 upstream 方便偶尔同步
git remote add origin <our-repo>    # 我们自己的 repo
```

**上游同步原则：** 不主动跟进 DeerFlow 版本更新。只在需要某个具体 bugfix 或新功能时，cherry-pick 特定 commit。fork 后这是我们的产品，不是 DeerFlow 的插件。

---

## 改动总览：配置 vs 源码 vs 新增

### 纯配置（不改源码）

| 做什么 | 在哪配 | 说明 |
|--------|--------|------|
| 注册 EthoInsight 工具 | `config.yaml` tools section | 指向我们写的 Python 模块 |
| 接入 GLM-5.1 | `config.yaml` models section | `langchain_openai:ChatOpenAI` + 智谱 base_url |
| 沙箱模式选择 | `config.yaml` sandbox section | MVP 用 local，后切 Docker |
| 记忆开关和参数 | `config.yaml` memory section | storage_dir, debounce, max_facts |
| noldus-kb 调用 | 不需要配置 | Agent 用内置 bash 工具 + prompt 引导 |

### 后端源码修改

| 改什么 | 路径 | 为什么配置不够 | 具体改动 |
|--------|------|---------------|----------|
| **System Prompt 基础角色** | `backend/app/agents/lead_agent/` | `apply_prompt_template()` 中的角色定义是硬编码的，Skills 只能注入额外内容，无法替换基础角色 | 将通用 AI 助手角色改为"EthoInsight 行为学数据分析专家"，包含分析原则、工具使用指南 |
| **中间件链** | `backend/app/agents/lead_agent/agent.py` | 12 个中间件是有序硬编码列表，不是插件系统 | 新增 `ExperimentContextMiddleware`：从上传的 EthoVision 文件自动提取实验元数据（范式、物种、分组、时长）注入 Agent 上下文 |
| **ThreadState** | `backend/app/agents/thread_state.py` | 只有通用字段（sandbox, artifacts, todos, uploaded_files） | 扩展加入 `experiment_context`（范式/分组/物种）和 `analysis_results`（结构化分析结果，供追问时使用） |
| **文件上传处理** | `backend/app/gateway/` + `UploadsMiddleware` | 上传 handler 用 `markitdown` 转换 PDF/PPT/Excel/Word，不认识 EthoVision 的 UTF-16 分号分隔格式 | 在上传处理逻辑中加 EthoVision 格式检测：识别 UTF-16 BOM + 分号分隔 + 特征 header，跳过 markitdown 转换，保留原始数据 |
| **记忆 fact 提取 prompt** | `backend/app/agents/memory/` | 默认提取通用对话 facts | 修改 LLM 提取 prompt，让它专注于保存：实验参数（范式、分组、样本量）、统计发现（显著差异的指标、p值、效应量）、分析结论、用户的研究假设 |
| **子 Agent 注册** | `backend/app/subagents/registry.py` | 只有 `general-purpose` 和 `bash` 两种内置子 Agent | 注册领域子 Agent：`statistics-analyst`（专门做统计分析的子 Agent，工具集只包含 run_statistics + assess_results） |

### 前端源码修改

| 改什么 | 路径 | 具体改动 |
|--------|------|----------|
| **数据上传界面** | `frontend/src/` | 新增 EthoVision 数据上传组件：拖拽多文件、自动识别范式、显示文件元数据（动物数、采样率、时长） |
| **实验配置表单** | `frontend/src/` | 新增实验上下文输入：范式选择、分组定义（对照组/实验组/剂量组）、物种品系、研究假设 |
| **报告展示** | `frontend/src/` | 报告 markdown 渲染优化：出版级图表内嵌、统计结果表格高亮、论文引用格式、一键导出 |
| **品牌化** | `frontend/src/` | Noldus 品牌色、Logo、产品名称 |

### 新增代码（不涉及改 DeerFlow 源码）

| 新增什么 | 路径 | 说明 |
|----------|------|------|
| **EthoVision 解析工具** | `backend/app/tools/ethoinsight/parse_ethovision.py` | 解析 UTF-16 分号分隔的 raw data，提取 metadata + time-series DataFrame |
| **指标计算工具** | `backend/app/tools/ethoinsight/compute_metrics.py` | 从 time-series 计算范式指标（scipy/pandas） |
| **统计分析工具** | `backend/app/tools/ethoinsight/run_statistics.py` | 自动选择统计检验（Shapiro-Wilk → t-test/ANOVA/Mann-Whitney）+ 效应量 |
| **图表生成工具** | `backend/app/tools/ethoinsight/generate_chart.py` | matplotlib 出版级图表：box plot, bar chart, trajectory（沙箱执行） |
| **结果评估工具** | `backend/app/tools/ethoinsight/assess_results.py` | 基于组间比较结果评估效应大小、检查混杂因素 |
| **YAML 知识库** | `knowledge/` | 统计方法选择规则、混杂因素检查清单、效应量解读参考 |
| **EthoInsight Skill** | `skills/custom/ethoinsight/SKILL.md` | 行为学数据分析工作流定义（补充 System Prompt） |

---

## config.yaml 关键配置

```yaml
models:
  - name: glm-5.1
    display_name: GLM-5.1
    use: langchain_openai:ChatOpenAI
    model: glm-5.1
    base_url: https://open.bigmodel.cn/api/paas/v4
    api_key: $GLM_API_KEY
    max_tokens: 4096
    temperature: 0.1

tools:
  # DeerFlow 内置沙箱工具（保留）
  - name: bash
    module: app.sandbox.tools:bash_tool
    group: sandbox
  - name: read_file
    module: app.sandbox.tools:read_file_tool
    group: sandbox
  - name: write_file
    module: app.sandbox.tools:write_file_tool
    group: sandbox

  # EthoInsight 自定义工具
  - name: parse_ethovision
    module: app.tools.ethoinsight.parse_ethovision:parse_ethovision
    group: ethoinsight
  - name: compute_metrics
    module: app.tools.ethoinsight.compute_metrics:compute_metrics
    group: ethoinsight
  - name: run_statistics
    module: app.tools.ethoinsight.run_statistics:run_statistics
    group: ethoinsight
  - name: generate_chart
    module: app.tools.ethoinsight.generate_chart:generate_chart
    group: ethoinsight
  - name: assess_results
    module: app.tools.ethoinsight.assess_results:assess_results
    group: ethoinsight

tool_groups:
  - name: sandbox
    enabled: true
  - name: ethoinsight
    enabled: true

sandbox:
  provider: app.sandbox.local:LocalSandboxProvider  # MVP 先用 local，后切 Docker

memory:
  enabled: true
  storage_dir: .ethoinsight/memory
```

## System Prompt 核心内容

```markdown
你是 EthoInsight，一位专业的动物行为学数据分析助手。

## 你的能力
- 解析 EthoVision XT 导出的原始轨迹数据
- 计算行为学范式的关键指标（组间比较，非绝对阈值判定）
- 自动选择正确的统计检验方法
- 生成出版级统计图表
- 撰写可用于论文的分析报告

## 工具使用指南
- `parse_ethovision`: 上传数据后首先调用，解析 EthoVision 文件
- `compute_metrics`: 从原始数据计算行为指标
- `run_statistics`: 对指标做组间统计比较
- `generate_chart`: 生成统计图表（在沙箱中执行）
- `assess_results`: 评估组间差异的生物学意义
- `bash` + `noldus-kb --format json`: 检索论文、范式、术语

## 分析原则
- 行为学没有绝对的"常模"。核心方法论是组间对比（对照组 vs 实验组）
- 所有统计计算使用确定性代码（scipy），不要自己编造数字
- 报告中引用 noldus-kb 检索到的真实论文
- 检查混杂因素（如运动量异常可能影响焦虑指标）
- 用户的最终目的是发论文。输出要能直接用于 Results 和 Discussion
```

---

## 实施顺序

| 阶段 | 内容 | 验证标准 |
|------|------|----------|
| **P0: Fork + GLM-5.1** | Fork DeerFlow，配置 GLM-5.1，验证基础 Agent 能对话 | 能对话、能调 bash、沙箱能跑 Python |
| **P1: parse_ethovision + 上传适配** | EthoVision 解析工具 + 修改上传 handler 识别 EthoVision 格式 | 能解析斑马鱼 demo 数据的 35 个 UTF-16 轨迹文件 |
| **P2: compute_metrics + generate_chart** | 指标计算 + 图表生成 | Agent 解析数据 → 计算群体指标 → 画 box plot |
| **P3: run_statistics + assess_results** | 统计检验 + 结果评估 | 完整的组间比较分析流程 |
| **P4: System Prompt + 中间件 + ThreadState** | 改 prompt 角色、加 ExperimentContextMiddleware、扩展 ThreadState | Agent 表现为行为学专家，上传数据后自动提取实验上下文 |
| **P5: noldus-kb + 记忆调优** | 沙箱安装 noldus-kb CLI + 修改记忆提取 prompt | Agent 引用论文、记忆保存实验参数和统计发现 |
| **P6: 前端定制** | 数据上传、实验配置、报告展示、品牌化 | 完整的 Web 交互流程 |
| **P7: 部署** | Docker Compose（EthoInsight + noldus-kb + PostgreSQL） | 本地一键启动 |

---

## 验证

1. **P0**: `make dev` 启动，GLM-5.1 对话正常，沙箱执行 `python -c "import scipy; print(scipy.__version__)"`
2. **P1**: 上传斑马鱼轨迹文件 → parse_ethovision 正确返回 DataFrame 描述（不被 markitdown 错误处理）
3. **P2-P3**: Agent 自主调用 parse → compute → statistics → chart → assess 完整链路
4. **P4**: Agent 自我介绍为 EthoInsight；上传文件后 ExperimentContextMiddleware 自动注入实验元数据
5. **P5**: Agent 报告引用 noldus-kb 论文；结束对话后记忆中保存了实验参数和统计发现（非通用对话 facts）
6. **端到端**: 上传斑马鱼 demo 数据 → 描述实验 → Agent 完成完整分析 → 报告含图表+文献引用 → 追问能基于记忆回答

---

## 关键文件路径（DeerFlow 中需要了解的）

### 需要改源码的文件

| 文件 | 改什么 |
|------|--------|
| `backend/app/agents/lead_agent/agent.py` | `apply_prompt_template()` 角色定义 + 中间件链顺序 |
| `backend/app/agents/thread_state.py` | 扩展 ThreadState 加 experiment_context, analysis_results |
| `backend/app/gateway/` (upload handler) | EthoVision 文件格式识别，跳过 markitdown |
| `backend/app/agents/middlewares/` | 新增 ExperimentContextMiddleware |
| `backend/app/agents/memory/` | 修改 fact 提取 prompt（行为学领域特化） |
| `backend/app/subagents/registry.py` | 注册 statistics-analyst 子 Agent |
| `frontend/src/` | 数据上传、实验配置、报告展示、品牌化 |

### 只需配置的文件

| 文件 | 配什么 |
|------|--------|
| `config.yaml` | GLM-5.1 model、EthoInsight 工具注册、沙箱、记忆参数 |
| `skills/custom/ethoinsight/SKILL.md` | 行为学分析工作流（补充 System Prompt） |

### 纯新增的文件

| 文件 | 内容 |
|------|------|
| `backend/app/tools/ethoinsight/*.py` | 5 个自定义工具 |
| `backend/app/agents/middlewares/experiment_context.py` | 实验上下文中间件 |
| `knowledge/*.yaml` | 统计规则、混杂因素、效应量解读 |
