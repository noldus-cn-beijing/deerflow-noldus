# EthoInsight：Agent 架构设计（v2）

## Context

Noldus 的动物行为学数据分析产品。用户上传 EthoVision 导出的 raw time-series 数据 + 输入实验上下文和期望输出，Agent 自动完成分析并生成报告，用户可以对结果追问。

关于为什么选择 Agent 方案而非原始 PRD 的可视化平台方案，参见 [why-agent-approach.md](./2026-04-01-why-agent-approach.md)。

**核心设计决策：**

| 决策项 | 选择 | 原因 |
|--------|------|------|
| Agent 框架 | DeerFlow（字节开源） | 内置 LangGraph Agent 编排、Docker 沙箱、Session 记忆、React 前端。一人开发不应该重复造这些轮子 |
| LLM | GLM-4 云端 API | 公司现有资源。通过 `langchain_openai:ChatOpenAI` + 智谱 `base_url` 接入 DeerFlow |
| LLM 角色 | **自主调用工具**（而非固定 pipeline） | 用户输入多样（不同范式、不同问题），LLM 根据上下文决定调哪些工具、按什么顺序 |
| 工具执行 | Docker 沙箱（DeerFlow 内置） | 图表生成、数据分析代码必须隔离执行，防止恶意代码或意外副作用 |
| 记忆 | Session 级 MD 文档 | 保存实验上下文、分析结果、对话历史，方便用户跨对话追问 |
| 知识存储 | 结构化 YAML + **noldus-kb** | YAML 存确定性规则（统计方法选择、混杂因素检查）。noldus-kb 提供 6200+ 篇论文、9 个 Noldus 产品文档、15 个范式定义、行为学术语库——Agent 通过 CLI 调用 |
| 语言 | 中英双语 | |
| 部署 | 先本地（Docker Compose），后云 | |
| 数据格式 | EthoVision raw data（UTF-16 分号分隔 time-series） | Agent 从原始数据计算指标 |

---

## 架构总览

### 之前 vs 现在的关键变化

之前的设计是**固定 7 步 pipeline**（parse → validate → metrics → stats → classify → cross-paradigm → report），LLM 只在最后一步参与。

现在的设计是**LLM 驱动的 Agent 循环**：LLM 根据用户的实验上下文和问题，自主决定调用哪些工具。这更灵活——用户可能只想看轨迹图，可能想做完整统计分析，可能想比较两组数据，可能想追问上次分析的某个细节。固定 pipeline 无法应对这种多样性。

但确定性原则不变：**所有科学计算在工具内部用确定性代码完成**，LLM 只决定调哪个工具、传什么参数，以及把结果写成报告。

### 架构图

```
用户
 │
 │  上传数据 + 实验上下文 + 问题
 ▼
┌─────────────────────────────────────────────────────────┐
│                    DeerFlow 框架层                        │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │              React 前端（DeerFlow 内置）         │     │
│  │  数据上传 / 实验配置 / 分析结果 / 对话追问       │     │
│  └──────────────────┬─────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼─────────────────────────────┐     │
│  │           LangGraph Agent 编排                   │     │
│  │                                                  │     │
│  │   GLM-4 ──tool_call──▶ 工具注册表               │     │
│  │     ▲                      │                     │     │
│  │     │                      ▼                     │     │
│  │   result ◀────────── Docker 沙箱执行             │     │
│  │     │                                            │     │
│  │     ▼                                            │     │
│  │   继续推理 or 返回结果                            │     │
│  └──────────────────┬─────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼─────────────────────────────┐     │
│  │           Session 记忆（MD 文档）                │     │
│  │  实验上下文 / 分析结果 / 对话历史                 │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
┌───────────┐  ┌────────────┐  ┌──────────────────┐
│ EthoInsight│  │  YAML      │  │  noldus-kb 服务   │
│ Tools      │  │  知识库    │  │  (PostgreSQL)     │
│ (我们开发)  │  │  (分析规则) │  │                  │
│            │  │            │  │  6200+ 论文/手册  │
│ parse      │  └────────────┘  │  9 产品 / 15 范式 │
│ compute    │                  │  15 行为学术语    │
│ statistics │                  │                  │
│ chart(沙箱)│                  │  CLI: noldus-kb  │
└───────────┘                  └──────────────────┘
```

---

## noldus-kb：Agent 的领域知识大脑

noldus-kb（`/home/qiuyangwang/noldus-kb`）是我们同步开发的行为科学知识检索服务。它包含：

- **6200+ 篇论文和手册**：行为神经科学领域的学术论文 + Noldus 产品技术文档
- **9 个 Noldus 产品**：EthoVision XT、CatWalk XT、DanioVision 等完整产品信息
- **15 个实验范式**：OFT、EPM、MWM、FST、NOR 等，含 setup、measured_parameters、related_products
- **15 个行为学术语**：distance moved、thigmotaxis、freezing 等，含定义、单位、计算方法

**接入方式：Agent 在沙箱中通过 CLI 调用 noldus-kb**

```bash
# Agent 需要了解某个范式的分析方法
noldus-kb paradigm "Elevated Plus Maze"

# Agent 为报告查找相关论文
noldus-kb search "elevated plus maze anxiety drug treatment" --limit 5

# Agent 需要解释某个指标
noldus-kb term thigmotaxis

# Agent 需要了解 EthoVision 的参数含义
noldus-kb product EthoVision
```

**noldus-kb 和 YAML 知识库的分工：**

| | YAML 知识库 | noldus-kb |
|---|---|---|
| 用途 | 确定性分析规则 | 领域知识检索 |
| 内容 | 统计方法选择规则、混杂因素检查清单 | 论文、产品文档、范式详情、术语 |
| 调用方式 | 工具内部直接读取 | Agent bash 调用 CLI |
| 特点 | 小、精确、确定性 | 大、丰富、语义检索 |
| 场景 | "该用什么统计检验" | "有没有相关论文支持这个结论" |

**部署：** noldus-kb 服务（PostgreSQL + FastAPI）随 EthoInsight 的 Docker Compose 一起启动，沙箱中的 CLI 通过 `NOLDUS_KB_API_URL` 连接。

---

## DeerFlow 提供什么 vs 我们开发什么

### DeerFlow 提供（直接用）

| 组件 | 说明 |
|------|------|
| **Agent 循环** | LangGraph 状态机，处理 tool_call → execution → observation → 继续推理的循环 |
| **Docker 沙箱** | 图表生成、数据分析代码在隔离容器中执行，支持本地/Docker/K8s 三种模式 |
| **Session 记忆** | 跨对话的长期记忆，持久化到本地文件 |
| **React 前端** | 对话界面、文件上传、结果展示 |
| **LLM 接入** | 通过 `langchain_openai:ChatOpenAI` + `base_url` 配置 GLM-4 |
| **IM 通道** | 可选：飞书/Slack/Telegram 集成 |

### 我们开发（EthoInsight 业务层）

| 组件 | 说明 | 沙箱执行 |
|------|------|----------|
| **Tool: parse_ethovision** | 解析 EthoVision UTF-16 raw data，提取 metadata + time-series | 否（纯解析） |
| **Tool: compute_metrics** | 从 time-series 计算范式指标（如 open_arm_time_pct） | 是 |
| **Tool: run_statistics** | 自动选择统计检验（Shapiro-Wilk → t-test/ANOVA/Mann-Whitney）+ 效应量 | 是 |
| **Tool: classify_phenotype** | 基于组间比较结果，识别显著差异的指标，评估效应量，检查混杂因素（如运动量） | 否（规则引擎） |
| **Tool: generate_chart** | matplotlib 生成出版级图表（bar, box, radar, heatmap, trajectory） | **是（必须沙箱）** |
| **Tool: query_knowledge** | 检索 YAML 知识库（统计方法选择规则、混杂因素检查清单） | 否 |
| **Tool: noldus-kb CLI** | Agent 在沙箱中 bash 调用 `noldus-kb search/paradigm/term/product`，获取论文、范式详情、术语定义、产品参数 | 是（bash 沙箱） |
| **Tool: cross_paradigm_analysis** | 多范式综合分析、综合评分 | 是 |
| **知识库 (YAML)** | 范式定义、分析规则、统计检验选择、报告模板、文献 | — |
| **System Prompt** | 行为学专家角色、分析流程引导、YAML 知识库使用指南、noldus-kb CLI 使用指南（何时搜论文、何时查范式） | — |

---

## Agent 工作流

### 典型流程（LLM 自主编排）

```
用户: "我上传了斑马鱼鱼群行为的 EthoVision 数据，5条鱼，7组实验。
      请帮我分析鱼群的群体凝聚度和运动同步性，比较不同组之间的差异。"

Agent 思考: 用户要分析斑马鱼群体行为，需要解析数据、计算群体指标、做组间统计比较

Agent → tool_call: parse_ethovision(files=[...])
  返回: {35个轨迹文件, 5只鱼×7组, 采样率30Hz, 时长5min}

Agent → tool_call: query_knowledge(paradigm="zebrafish_shoaling")
  返回: {指标定义: 鱼间距离, 极化度, 群体速度, 凝聚指数...}

Agent → bash: noldus-kb search "zebrafish shoaling group cohesion" --limit 3
  返回: {3篇相关论文的摘要和方法论参考}

Agent → tool_call: compute_metrics(data=..., metrics=["inter_fish_distance", "polarization", "group_velocity"])
  返回: {每组每个trial的群体指标汇总}

Agent → tool_call: run_statistics(metrics=..., groups=[...], comparison="between_groups")
  返回: {Kruskal-Wallis H=12.3, p=0.003, post-hoc Dunn's test...}

Agent → tool_call: generate_chart(type="box_plot", data=..., groups=..., significance=...)
  [沙箱执行] 返回: {figure_path: "/output/shoaling_comparison.png"}

Agent → bash: noldus-kb search "zebrafish shoaling drug treatment group comparison" --limit 3
  返回: {相关论文，用于报告中的文献引用}

Agent → 生成报告（自然语言总结所有结果 + 引用 noldus-kb 检索到的论文）
```

### 追问流程（利用 Session 记忆）

```
用户: "group 3 的凝聚度为什么特别低？能看一下轨迹图吗？"

Agent 从 Session 记忆加载: 之前的分析结果 + 实验上下文

Agent → tool_call: generate_chart(type="trajectory", data=..., group=3, overlay=True)
  [沙箱执行] 返回: {轨迹图}

Agent → 基于之前的统计结果 + 轨迹可视化，给出解释
```

---

## Session 记忆设计

每个分析 session 对应一个 MD 文档，存储在 DeerFlow 的记忆系统中：

```markdown
# Session: zebrafish-shoaling-20260402

## 实验上下文
- 物种: 斑马鱼
- 范式: 鱼群行为（Shoaling）
- 数据: 35个轨迹文件, 5只鱼/组, 7组
- 采样率: 30Hz
- 时长: 5分钟/trial
- 用户目标: 分析群体凝聚度和运动同步性的组间差异

## 分析结果
### 数据解析
- 总数据点: 315,000
- 缺失率: 0%
- 坐标范围: X[-30, 180]mm, Y[120, 175]mm

### 指标计算
| 指标 | Group 1 | Group 2 | Group 3 | ... |
|------|---------|---------|---------|-----|
| 鱼间距离(mm) | 25.3±4.2 | 28.1±5.1 | 42.7±8.3 | ... |
| 极化度 | 0.72±0.08 | 0.68±0.11 | 0.45±0.15 | ... |

### 统计检验
- Kruskal-Wallis: H=12.3, p=0.003
- Post-hoc: Group 3 vs others, p<0.01

### 生成的图表
- /output/shoaling_comparison_boxplot.png
- /output/group3_trajectory.png

## 对话历史
- [用户] 分析鱼群凝聚度...
- [Agent] 已完成分析，Group 3 凝聚度显著低于其他组...
- [用户] Group 3 为什么低？
- [Agent] 从轨迹图可以看到...
```

---

## 知识库结构

```
knowledge/
├── paradigms/
│   ├── zebrafish_shoaling.yaml       # 斑马鱼群体行为指标
│   ├── epm.yaml                      # 高架十字迷宫
│   ├── open_field.yaml               # 旷场实验
│   ├── o_maze.yaml                   # O迷宫
│   ├── light_dark_box.yaml           # 明暗箱
│   ├── nsf.yaml                      # 新奇抑制摄食
│   ├── morris_water_maze.yaml        # 莫里斯水迷宫
│   ├── forced_swim.yaml              # 强迫游泳
│   ├── novel_object_recognition.yaml # 新物体识别
│   └── ...                           # 按 DemoData 中的范式扩展
├── rules/
│   ├── statistical_test_selection.yaml
│   ├── phenotype_classification.yaml
│   └── strain_adjustments.yaml
├── report_templates/
│   └── interpretation_guidelines.yaml
└── ethovision/
    └── column_mappings.yaml          # EthoVision 列名 → 标准 ID 映射
```

---

## GLM-4 接入配置

DeerFlow 的 `config.yaml`：

```yaml
models:
  - name: glm-4
    display_name: GLM-4
    use: langchain_openai:ChatOpenAI
    model: glm-4
    base_url: https://open.bigmodel.cn/api/paas/v4
    api_key: $GLM_API_KEY
    max_tokens: 4096
    temperature: 0.1    # 科学分析场景低温度
```

---

## EthoVision 数据格式

基于实际 demo 数据 (`DemoData/斑马鱼鱼群行为/`) 观察到的格式：

- **编码**: UTF-16 Little-Endian with BOM
- **分隔符**: 分号 (`;`)
- **Header**: 37-39 行 metadata（实验名、试验名、对象名、观察区、检测设置、开始时间、持续时间、缺失率等）
- **列**: 试用时间、录制时间、X中心、Y中心、区域、面积变化、伸长、移动距离、Velocity、Result 1
- **单位**: 秒、毫米、毫米²、毫米/秒
- **缺失值**: `"-"` (字符串 dash)
- **采样率**: ~30Hz (0.033s 间隔)

`parse_ethovision` 工具需要处理这些格式特性。

---

## 项目结构

```
noldus-insight/
├── docs/                             # 文档（已有）
│   ├── plans/                        # 架构设计文档（本文件）
│   └── EthoInsight-技术路径&背景/     # 原始讨论文档
│
├── knowledge/                        # 结构化知识库 (YAML)
│   ├── paradigms/
│   ├── rules/
│   ├── report_templates/
│   └── ethovision/
│
├── tools/                            # EthoInsight Agent 工具（注册到 DeerFlow）
│   ├── parse_ethovision.py           # 解析 EthoVision raw data
│   ├── compute_metrics.py            # 范式指标计算
│   ├── run_statistics.py             # 统计检验
│   ├── classify_phenotype.py         # 表型分类
│   ├── generate_chart.py             # 图表生成（沙箱执行）
│   ├── query_knowledge.py            # 知识库检索
│   └── cross_paradigm.py             # 跨范式分析
│
├── prompts/                          # System Prompt 和模板
│   ├── system_prompt.md              # Agent 角色定义 + 工具使用指南
│   └── report_template.md            # 报告生成模板
│
├── deerflow/                         # DeerFlow 框架（git submodule 或 fork）
│   └── config.yaml                   # GLM-4 配置、沙箱配置、记忆配置
│
└── tests/
    ├── fixtures/                     # 测试用 EthoVision 数据
    └── ...
```

---

## 实施顺序

| 阶段 | 内容 | 产出 |
|------|------|------|
| **P0: DeerFlow + noldus-kb** | clone DeerFlow，配置 GLM-4，验证 Agent 循环 + 沙箱。在沙箱中安装 noldus-kb CLI，验证能调通 | Agent 能对话、能在沙箱执行 Python 和 noldus-kb 命令 |
| **P1: 数据解析工具** | `parse_ethovision` — 解析斑马鱼 demo 数据 | 能正确读取 UTF-16 文件，输出标准化 DataFrame |
| **P2: 知识库 + 第一个范式** | 斑马鱼群体行为 YAML + `compute_metrics` + `generate_chart` | Agent 能解析数据 → 计算群体指标 → 画图 |
| **P3: 统计分析** | `run_statistics` + `classify_phenotype` | Agent 能做完整的统计检验和分类 |
| **P4: System Prompt + 报告** | 行为学专家 prompt + 报告生成 | Agent 能输出结构化的分析报告 |
| **P5: Session 记忆** | 配置 DeerFlow 记忆系统，设计 session MD 格式 | 用户能跨对话追问 |
| **P6: 扩展范式** | 焦虑5范式 YAML + tools 适配（EPM, OF, O-Maze, LDB, NSF） | 支持更多范式 |
| **P7: 打磨** | 错误处理、边界情况、中英双语、Docker Compose 部署 | 可交付的产品 |

**关键变化：** 先用斑马鱼 demo 数据跑通端到端，而不是先做焦虑范式（因为我们有现成的斑马鱼数据可以验证）。

---

## 验证方式

1. **P0 验证**: DeerFlow + GLM-4 能跑通 tool_call，沙箱能执行 `matplotlib` 生成图表
2. **P1 验证**: 解析 35 个斑马鱼轨迹文件，输出和原始数据一致
3. **P2 验证**: Agent 对话中能从上传数据到输出群体指标 + 图表
4. **P3 验证**: 统计结果与手动用 scipy 计算的一致
5. **端到端**: 用户上传斑马鱼数据 → 描述实验 → Agent 完成完整分析 → 输出报告 → 用户追问 → Agent 基于记忆回答
