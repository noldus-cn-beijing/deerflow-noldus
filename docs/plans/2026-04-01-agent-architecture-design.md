# EthoInsight: End-to-End Behavioral Analysis Agent

## Context

Noldus 已有详细的动物行为学数据分析平台 PRD 和文档（16+ 范式、可视化规范、前端设计），但原始设计偏向传统 web 可视化平台。我们转向 **Agent 驱动**方案：用户上传 EthoVision 导出的 raw time-series 数据，Agent 自动完成从数据解析、指标计算到报告生成的全流程，帮助没有数据分析经验的实验人员直接获得结果。

**为什么选择 Agent 方案而非原方案：**
- 原方案是"给专家更好的工具"，Agent 方案是"让非专家也能得到专家级结果"——后者市场更大、差异化更强
- 实验人员的核心需求不是"更多图表"，而是"告诉我实验结果是什么"
- Noldus 的护城河在行为学专业知识（指标、阈值、解读规则），Agent 方案直接将这些知识编码变现
- 符合 AI 时代从"工具"到"助手"的产品趋势

**核心设计决策：**
- 交互方式：Form + Agent pipeline（配置式），用户可对结果追问
- 知识存储：结构化 YAML 知识库（非 RAG），确定性地存储指标、阈值、分析规则
- LLM 角色：Workflow 编排 + 报告生成，所有统计计算由确定性 Python 代码完成
- 技术栈：Python (FastAPI) + 极简前端 + SQLite + YAML 知识库
- LLM：云端 API（Claude/GPT）
- 部署：先本地（Docker），后云
- 语言：中英双语
- MVP 范围：焦虑样行为 5 个范式（EPM, Open Field, O-Maze, Light-Dark Box, NSF）
- 数据输入：EthoVision raw time-series data（Agent 从原始数据计算指标）
- 团队：一人先做 prototype

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│           极简前端 (or CLI for prototype)              │
│  [选范式] → [传数据] → [看进度] → [读报告] → [追问]     │
└───────────────────────┬──────────────────────────────┘
                        │ REST API
┌───────────────────────▼──────────────────────────────┐
│                 FastAPI Backend                        │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │         Agent Orchestrator (Pipeline)         │     │
│  │  parse_raw → validate → compute_metrics →     │     │
│  │  stats → classify → cross_paradigm → report   │     │
│  └──┬──────────┬───────────────┬────────────────┘     │
│     │          │               │                       │
│  ┌──▼──────┐ ┌▼───────────┐ ┌▼───────────┐          │
│  │Analysis │ │Knowledge   │ │LLM Client  │          │
│  │Engine   │ │Base (YAML) │ │(Report+Q&A)│          │
│  │+ Parser │ └────────────┘ └────────────┘          │
│  │(scipy)  │                                         │
│  └─────────┘                                         │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
noldus-insight/
├── knowledge/                        # 结构化知识库
│   ├── paradigms/                    # 每范式一个 YAML
│   │   ├── epm.yaml                  # 指标定义、阈值、severity bands、区域定义、文献
│   │   ├── open_field.yaml
│   │   ├── o_maze.yaml
│   │   ├── light_dark_box.yaml
│   │   └── nsf.yaml
│   ├── rules/
│   │   ├── phenotype_classification.yaml   # 跨范式判定规则
│   │   ├── statistical_test_selection.yaml # 统计检验自动选择
│   │   └── strain_adjustments.yaml         # 品系特异性修正
│   └── report_templates/
│       └── interpretation_guidelines.yaml  # 报告生成指南（中英双语模板）
│
├── backend/
│   ├── main.py                       # FastAPI entry
│   ├── config.py
│   │
│   ├── api/
│   │   ├── routes/                   # upload, analysis, paradigms, reports, figures, chat
│   │   └── schemas/                  # Pydantic models
│   │
│   ├── parser/                       # ★ EthoVision raw data 解析器
│   │   ├── ethovision.py             # 解析 EthoVision 导出格式（header metadata + time-series）
│   │   ├── column_mapper.py          # 列名标准化映射
│   │   └── detector.py               # 自动检测文件格式和范式
│   │
│   ├── agent/
│   │   ├── orchestrator.py           # Sequential pipeline controller
│   │   ├── state.py                  # AnalysisState dataclass
│   │   └── steps/                    # parse_raw, validate, compute_metrics, stats, classify, cross_paradigm, report
│   │
│   ├── analysis/
│   │   ├── base.py                   # Abstract ParadigmAnalyzer
│   │   ├── registry.py
│   │   ├── paradigms/                # epm.py, open_field.py, o_maze.py, light_dark_box.py, nsf.py
│   │   │                             # 每个 analyzer 从 raw time-series 计算范式指标
│   │   ├── statistics/               # descriptive, comparison, effect_size
│   │   └── classification/           # threshold + cross-paradigm
│   │
│   ├── knowledge/                    # YAML loader with caching
│   ├── figures/                      # matplotlib publication-quality generators
│   ├── report/                       # LLM report builder (中英双语)
│   ├── storage/                      # SQLite + file store
│   └── llm/                          # Provider-agnostic LLM client
│
├── cli.py                            # ★ CLI 入口 (prototype 阶段主要交互方式)
│
├── frontend/                         # 极简前端（M6 再做）
│
├── Dockerfile
├── docker-compose.yaml
│
└── tests/
    ├── fixtures/                     # EthoVision raw data samples
    └── ...
```

---

## EthoVision Raw Data 解析（核心模块）

EthoVision 导出的 raw data 是带 header metadata 的 time-series CSV。需要：

1. **Header 解析**：提取实验元数据（范式、动物 ID、组别、实验时长、采样率等）
2. **列名标准化**：EthoVision 列名可能是 "In zone(Open arm 1)/center-point"、"Distance moved" 等，需映射到标准 ID
3. **Time-series → 指标计算**：从原始时间序列计算汇总指标（如从 zone occupancy time-series 计算 open_arm_time_pct）
4. **范式自动检测**：根据列名和区域名称推断实验范式

每个范式 YAML 中增加 `raw_data_schema` 段，定义期望的 EthoVision 列和区域映射。

---

## Agent Pipeline (7 Steps)

| Step | Input | 执行逻辑 | Output |
|------|-------|----------|--------|
| **1. Parse Raw** | EthoVision 文件 | 解析 header + time-series，标准化列名，检测范式 | 标准化 DataFrames + metadata |
| **2. Validate** | DataFrames + YAML schema | 检查必要列、数据类型、采样率、最小样本量 | 验证通过的 DataFrames |
| **3. Compute Metrics** | raw time-series | 从 time-series 计算各范式汇总指标 + 描述统计 | metrics dict + descriptive stats |
| **4. Statistical Test** | metrics + YAML 规则 | Shapiro-Wilk → Levene → 选择检验 → 效应量 | test results + significance + figures |
| **5. Classify** | metrics + severity bands | 按 YAML 阈值分级，检查运动混杂 | per-subject phenotype labels |
| **6. Cross-Paradigm** | classifications (>1范式时) | 综合判定规则 + 雷达图 | composite anxiety score |
| **7. Generate Report** | all results + guidelines | LLM 生成中/英文自然语言报告 | markdown report (bilingual) |

**关键原则：** Pipeline 是确定性的顺序执行。LLM 仅在 Step 7（报告生成）和后续 Q&A 中参与。统计计算全部由 scipy/statsmodels 完成，保证可复现性。

---

## API Endpoints

```
GET  /api/paradigms                    # 范式列表
POST /api/upload                       # 上传 EthoVision 文件
POST /api/analysis/{id}/start          # 启动异步分析
GET  /api/analysis/{id}/status         # 轮询进度
GET  /api/analysis/{id}/results        # 分析结果 + 报告
GET  /api/figures/{figure_id}          # 图片
POST /api/chat                         # Follow-up Q&A (中英双语)
```

---

## CLI (Prototype 阶段)

一个人做 prototype 时，CLI 是最快验证核心分析能力的方式：

```bash
# 基本使用
ethoinsight analyze --paradigm epm --data ./epm_raw.csv --lang zh

# 多范式
ethoinsight analyze --paradigm epm,open_field --data ./data/ --lang en

# 交互式追问
ethoinsight chat --analysis-id abc123 "为什么 treatment 组被判定为高焦虑？"
```

CLI 和 API 共享同一个 analysis engine，CLI 只是跳过了 web 层直接调用 orchestrator。

---

## Implementation Order (一人开发优化)

| Milestone | Scope | 说明 |
|-----------|-------|------|
| **M1: 基础 + 知识库** | 项目脚手架、5 个范式 YAML、知识库 loader | 把文档中的指标/阈值编码为 YAML |
| **M2: EthoVision 解析器** | raw data parser、列名映射、范式检测 | 需要 raw data 样本来开发 |
| **M3: 分析引擎** | EPM analyzer 端到端 → 统计模块 → 分类 → 其余 4 个 | 先跑通一个范式再横向扩展 |
| **M4: Pipeline + CLI** | Orchestrator、CLI 入口、端到端测试 | 此时可以 CLI 跑通完整流程 |
| **M5: LLM 报告** | 报告生成（中英双语）、Q&A | 接入 Claude API |
| **M6: API + 极简前端** | FastAPI routes、简单 web UI | 把 CLI 能力包装为 web 服务 |
| **M7: Docker + 打磨** | Dockerfile、错误处理、Demo 数据 | 可交付的本地部署包 |

---

## Verification

1. **EthoVision 解析：** 用真实 raw data 样本验证解析正确性（列识别、指标计算与 EthoVision 汇总结果对比）
2. **单元测试：** 每个 analyzer 用 fixture 数据验证指标计算
3. **统计正确性：** 用已知数据验证 t-test/ANOVA 结果与 R/SPSS 一致
4. **端到端：** CLI 输入 raw data → 验证完整输出（metrics + stats + classification + report）
5. **知识库一致性：** YAML 与文档 `小鼠焦虑样行为范式-20260112.md` 完全一致
6. **双语报告：** 同一分析结果分别生成中英文报告，验证质量

---

## 下一步需要提供的

1. **EthoVision raw data 样本**（至少 EPM 的）— 这是 M2 的前提
2. **确认 EthoVision 导出格式**：header 结构、列命名约定、编码等

---

## Critical Files (existing docs)

- `docs/EthoInsight-技术路径&背景/小鼠焦虑样行为范式-20260112.md` — 5 个范式的指标、阈值、文献
- `docs/EthoInsight-技术路径&背景/总洞察范围-20260112.md` — 跨范式分析框架、表型分类逻辑
- `docs/EthoInsight-技术路径&背景/小鼠焦虑样-可视化呈现方案-20260112.md` — 前端设计 tokens
- `docs/prd.md` — 总体产品需求
