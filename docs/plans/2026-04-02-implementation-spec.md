# EthoInsight 二次开发实施文档

> 状态：待审批
> 版本：v3（2026-04-03 修订）
> 作者：Claude Opus 4.6 + qiuyangwang

---

## 1. 项目概述

### 1.1 目标

将 DeerFlow fork 改造为 EthoInsight——Noldus 的动物行为学数据分析 Agent。

### 1.2 核心架构

#### Agent 体系

```
Lead Agent (EthoInsight Orchestrator)
│  职责：理解需求 → 主动提问澄清 → 按流程串行派遣 → 读 handoff → 整合回复
│
│  ──── Step 1 ────
├── code-executor subagent
│   输入：lead agent 的任务描述（范式、分组、文件路径、用户需求）
│   工作：选预制模板 → 读数据文件开头确认格式 → 微调代码 → 沙箱执行
│   兜底：执行失败 → 查原因 → 读数据前500行 → 修代码 → 重试
│   输出：
│     - handoff JSON（执行状态 + 结果文件路径摘要）→ 给 lead agent
│     - 数据 output 文件（metrics CSV、statistics JSON、图表 PNG）→ 给下游 agent 读
│
│  ──── Step 2（lead agent 读 handoff 后派遣）────
├── data-analyst subagent
│   输入：lead agent 的任务 + code-executor 的数据 output 文件路径
│   工作：读实际数据结果 → 用 Noldus 领域 skills 洞察 → 可选调用 noldus-kb 查文献
│   输出：
│     - handoff JSON（分析结论 + 洞察）→ 给 lead agent
│     - 分析文档（详细分析内容）→ 给 report-writer 读
│
│  ──── Step 3（lead agent 读 handoff 后派遣）────
└── report-writer subagent
    输入：lead agent 的任务 + code-executor 的数据 output + data-analyst 的分析文档
    工作：读原始数据结果 + 分析结论 → 用 report skills 写论文级报告
    输出：
      - handoff JSON（报告文件路径）→ 给 lead agent
      - 报告文件（markdown/PDF）

Lead Agent 读所有 handoff → 整合 → 返回给用户
```

**执行顺序严格串行**：code-executor → data-analyst → report-writer。每一步由 lead agent 协调。

#### DeerFlow 现有 Agent（改造基础）

DeerFlow 只有 **1 个主 Agent + 2 个内置 subagent**：

| Agent | 角色 | 工具集 | 我们是否保留 |
|---|---|---|---|
| **Lead Agent** | 主 Agent，与用户对话，通过 `task()` 工具委派 | 所有工具 + task | 保留，改造为 orchestrator |
| **general-purpose** | 通用执行者 | 继承所有工具（无 task） | 保留 |
| **bash** | 命令执行 | bash, ls, read_file, write_file, str_replace | 保留 |

我们新增 3 个专业 subagent：code-executor、data-analyst、report-writer。

#### 数据处理方式

**不注册固定 tool**。code-executor 在 Docker 沙箱中执行代码：

1. ethoinsight 库提供**完整可执行模板脚本**（每种实验 setup 一套）
2. code-executor 根据范式选模板 → 读数据开头确认格式 → 微调参数和代码 → bash 执行
3. 用户需求不同（相关性分析、特定区域停留时间、自定义指标等），code-executor 据此调整代码

**为什么不注册为 DeerFlow tool**：
- 格式固定但**用户需求不可预测**（用户 A 要相关性，用户 B 要别的）
- LLM 需要**微调代码**适配需求，固定 tool 参数无法表达这种灵活性
- 预制模板节省 LLM 写代码时间，微调保留灵活性

#### 兜底策略

code-executor 执行失败时的处理链：

```
执行失败
  → 读 stderr/错误信息
  → 判断错误类型：
    ├── 格式问题 → read_file 数据文件前 500 行 → 理解实际格式 → 修改代码 → 重试
    ├── 依赖缺失 → bash pip install → 重试
    ├── 逻辑错误 → 分析代码问题 → 修复 → 重试
    └── 数据问题（缺失值、异常值）→ 添加错误处理代码 → 重试
  → 最多重试 3 次
  → 仍然失败 → 在 handoff 中标记 failed + 错误详情 → lead agent 告知用户
```

#### Handoff 机制

**handoff JSON** = subagent 完成后写入 workspace 的结构化交接文件：

```
/mnt/user-data/workspace/
├── handoff_code_executor.json       # code-executor 的交接
├── handoff_data_analyst.json        # data-analyst 的交接
├── handoff_report_writer.json       # report-writer 的交接
│
├── output/                          # code-executor 的数据 output（实际数据文件）
│   ├── metrics.csv                  # 指标数据
│   ├── statistics.json              # 统计结果
│   ├── iid_boxplot.png              # 图表
│   ├── nnd_boxplot.png
│   └── ...
│
└── analysis/                        # data-analyst 的详细分析
    └── analysis_report.md           # 详细分析内容
```

**handoff JSON 格式示例**（code-executor）：

```json
{
  "status": "completed",
  "summary": "35个轨迹文件解析成功，5个指标计算完成，3个显著",
  "output_files": {
    "metrics": "/mnt/user-data/workspace/output/metrics.csv",
    "statistics": "/mnt/user-data/workspace/output/statistics.json",
    "charts": [
      "/mnt/user-data/workspace/output/iid_boxplot.png",
      "/mnt/user-data/workspace/output/nnd_boxplot.png"
    ]
  },
  "metadata": {
    "paradigm": "shoaling",
    "n_files": 35,
    "n_subjects": 5,
    "groups": {"control": ["Subject 1", "Subject 2"], "treatment": ["Subject 3", "Subject 4", "Subject 5"]}
  },
  "errors": []
}
```

**data-analyst 和 report-writer 读的是 output 文件**（metrics.csv、statistics.json、图表 PNG），不是 handoff。handoff 只是给 lead agent 看状态摘要的。

#### Lead Agent 的主动提问

Lead Agent 在派遣 subagent 前，**必须确认**：

1. **范式**：什么实验？（可从文件名/ExperimentContextMiddleware 推断，但要确认）
2. **分组**：哪些 Subject 是对照组/实验组？
3. **需求**：用户想看什么？（组间比较、相关性、特定区域分析、自定义指标...）
4. **缺失信息**：如果无法推断，必须先问

DeerFlow 已实现 `ask_clarification` 工具，Lead Agent 可以用它中断执行、向用户提问。

#### 预制代码模板

每种实验 setup 一个完整可执行的 Python 脚本，放在 `ethoinsight/templates/` 下：

```python
# templates/shoaling.py — 斑马鱼群体行为分析模板
"""
code-executor 读取此模板，根据用户需求修改后在沙箱中执行。
参数区在脚本顶部，方便 LLM 定位和修改。
"""
import json, glob
from ethoinsight import parse, metrics, statistics, charts

# ===== 参数区（code-executor 修改这里）=====
FILE_PATTERN = "/mnt/user-data/uploads/轨迹*.txt"
PARADIGM = "shoaling"
GROUPS = {
    "control": ["Subject 1", "Subject 2"],
    "treatment": ["Subject 3", "Subject 4", "Subject 5"]
}
METRICS_TO_COMPUTE = ["inter_individual_distance", "nearest_neighbor_distance",
                      "group_polarity", "swimming_speed"]
CHART_TYPES = ["box_plot"]  # box_plot, violin_plot, bar_chart
OUTPUT_DIR = "/mnt/user-data/workspace/output/"
HANDOFF_PATH = "/mnt/user-data/workspace/handoff_code_executor.json"
# ===== 参数区结束 =====

# 1. 解析
data = parse.parse_batch(FILE_PATTERN)
print(parse.get_summary(data))

# 2. 计算指标
m = metrics.compute_paradigm_metrics(data, PARADIGM, groups=GROUPS,
                                      metrics=METRICS_TO_COMPUTE)

# 3. 统计检验
stat_results = statistics.compare_groups(m, list(GROUPS.keys()))

# 4. 图表
chart_paths = []
for metric in METRICS_TO_COMPUTE:
    for chart_type in CHART_TYPES:
        fn = getattr(charts, chart_type)
        path = fn(m, [metric], significance=stat_results,
                  output_path=f"{OUTPUT_DIR}{metric}_{chart_type}.png")
        chart_paths.append(path)

# 5. 保存数据 output
metrics.save_to_csv(m, f"{OUTPUT_DIR}metrics.csv")
with open(f"{OUTPUT_DIR}statistics.json", "w") as f:
    json.dump(stat_results, f, ensure_ascii=False, indent=2, default=str)

# 6. 写 handoff
handoff = {
    "status": "completed",
    "summary": stat_results.get("summary", ""),
    "output_files": {
        "metrics": f"{OUTPUT_DIR}metrics.csv",
        "statistics": f"{OUTPUT_DIR}statistics.json",
        "charts": chart_paths,
    },
    "metadata": {"paradigm": PARADIGM, "n_files": data["summary"]["total_files"],
                  "groups": GROUPS},
    "errors": []
}
with open(HANDOFF_PATH, "w") as f:
    json.dump(handoff, f, ensure_ascii=False, indent=2, default=str)
print(f"HANDOFF: {HANDOFF_PATH}")
```

**模板覆盖的实验 setup**（11 种范式）：

| 模板 | 实验 | 关键指标 |
|------|------|---------|
| `shoaling.py` | 斑马鱼群体行为 | IID、NND、群体极性、游泳速度 |
| `open_field.py` | 旷场实验 | 中心区时间比、总距离、速度、趋触性 |
| `epm.py` | 高架十字迷宫 | 开臂时间比、开臂进入次数、总距离 |
| `novel_object.py` | 新物体识别 | 探索时间比、鉴别指数 |
| `y_maze.py` | Y迷宫 | 自发交替率、臂进入次数 |
| `forced_swim.py` | 强迫游泳 | 不动时间、游泳时间、攀爬时间 |
| `barnes_maze.py` | 巴恩斯迷宫 | 逃避潜伏期、错误数 |
| `morris_water_maze.py` | Morris 水迷宫 | 逃避潜伏期、平台区停留时间 |
| `light_dark.py` | 明暗箱 | 暗箱时间比、穿梭次数 |
| `social_interaction.py` | 社会互动 | 接触时间、接触次数 |
| `o_maze.py` | O迷宫 | 开放区时间比、进入次数 |

---

## 2. Monorepo 结构

```
noldus-insight/                         # 主仓库
├── docs/                               # 项目文档
├── packages/
│   ├── ethoinsight/                    # Python 数据分析库
│   │   ├── pyproject.toml
│   │   ├── ethoinsight/
│   │   │   ├── __init__.py
│   │   │   ├── parse.py                # EthoVision 文件解析
│   │   │   ├── metrics.py              # 行为指标计算
│   │   │   ├── statistics.py           # 统计检验
│   │   │   ├── charts.py              # 出版级图表
│   │   │   ├── assess.py              # 结果评估（被 data-analyst 用）
│   │   │   ├── utils.py               # 通用工具
│   │   │   └── templates/             # 预制分析脚本
│   │   │       ├── shoaling.py
│   │   │       ├── open_field.py
│   │   │       ├── epm.py
│   │   │       └── ...（11 种范式）
│   │   └── tests/
│   └── agent/                          # DeerFlow fork（嵌套 git repo）
│       ├── backend/
│       │   ├── app/gateway/            # 上传 handler 改动
│       │   └── packages/harness/
│       │       └── deerflow/           # Agent 核心改动
│       │           ├── agents/lead_agent/   # Lead Agent prompt
│       │           ├── agents/middlewares/   # ExperimentContextMiddleware
│       │           ├── agents/thread_state.py
│       │           ├── agents/memory/       # 记忆 prompt
│       │           └── subagents/builtins/  # 3 个新 subagent
│       ├── frontend/
│       ├── skills/custom/ethoinsight/  # EthoInsight Skills
│       └── config.yaml
├── knowledge/                          # YAML 知识库
└── demo-data/                          # 测试数据 symlink
```

---

## 3. DeerFlow 关键发现

### 3.1 代码路径

核心 Agent 框架在 `packages/agent/backend/packages/harness/deerflow/`（以下简称 `deerflow/`）。

| 组件 | 路径 |
|---|---|
| Lead Agent prompt | `deerflow/agents/lead_agent/prompt.py` |
| 中间件链 | `deerflow/agents/lead_agent/agent.py` → `_build_middlewares()` |
| ThreadState | `deerflow/agents/thread_state.py` |
| 上传处理 | `packages/agent/backend/app/gateway/routers/uploads.py` |
| 记忆提取 | `deerflow/agents/memory/prompt.py` |
| Subagent 注册 | `deerflow/subagents/builtins/__init__.py` → `BUILTIN_SUBAGENTS` |
| Subagent 配置 | `deerflow/subagents/config.py` → `SubagentConfig` |
| Subagent 执行 | `deerflow/subagents/executor.py` |
| Task 工具 | `deerflow/tools/builtins/task_tool.py` |

### 3.2 Subagent 机制（DeerFlow 已实现）

- Lead Agent 通过 `task()` 工具委派，参数：`description`、`prompt`、`subagent_type`
- 每个 subagent 有独立的 agent 实例（`create_agent()`），不继承主对话上下文
- subagent 继承沙箱状态和 thread workspace（同一个文件系统）
- 工具通过 allowlist/denylist 过滤
- 最多 3 个 subagent 并行（我们用串行，不受此限制）
- subagent 完成后返回字符串结果给 task() 工具

**lead agent 给 subagent 的任务 = task() 的 `prompt` 参数**。我们需要在 lead agent 的 system prompt 中指导它如何组织清晰的任务描述。

### 3.3 EthoVision 数据实测

| 范式 | 文件数 | 总行数 | 单文件大小 | 格式 |
|---|---|---|---|---|
| 斑马鱼 | 35 | 316K | 1-1.5 MB | UTF-16 LE + BOM + 分号分隔 |
| O迷宫 | 32 | 400K | ~2.5 MB | 同上 |
| 社会互动 | 2 | 90K | ~12 MB | 同上 |

格式固定：前 N 行 header（元数据），之后是时间序列数据。不同范式 header 行数和数据列不同，但同一范式格式一致。

### 3.4 GLM-5.1 Prompt Cache

自动前缀缓存，缓存输入 1/5 价格。需要 prompt 中静态内容前置、动态内容后置。

---

## 4. 环境配置（P0）

### 4.1 系统依赖

| 依赖 | 当前 | 需要 | 安装 |
|---|---|---|---|
| Node.js | 20.20.0 | 22+ | `nvm install 22` |
| nginx | 未安装 | 需要 | `sudo apt install -y nginx` |
| pnpm | 10.33.0 | ✅ | — |
| uv | 0.10.2 | ✅ | — |

### 4.2 config.yaml

已创建在 `packages/agent/config.yaml`，配置了 GLM-5.1 模型。

### 4.3 启动验证

```bash
cd packages/agent
export GLM_API_KEY=<key>
make install && make dev
```

验证：GLM-5.1 对话正常 + 沙箱能跑 Python + `import ethoinsight` 可用。

---

## 5. ethoinsight Python 库（P1-P3）

### 5.1 parse.py — EthoVision 解析（P1）

```python
def detect_ethovision(file_path: str) -> bool:
    """检测 UTF-16 LE BOM + 特征 header"""

def parse_header(file_path: str) -> dict:
    """解析 header 元数据"""

def parse_trajectory(file_path: str) -> pd.DataFrame:
    """解析单个轨迹文件为 DataFrame（列名统一为英文 snake_case）"""

def parse_batch(file_paths: "list[str] | str") -> dict:
    """批量解析，返回 {metadata, subjects, all_data, summary}"""

def get_summary(parsed_data: dict, max_chars: int = 2000) -> str:
    """生成数据摘要（给 Agent 看，控制大小）"""
```

### 5.2 metrics.py — 指标计算（P2）

通用指标 + 每种范式的专用指标：

```python
# 通用
def compute_distance_moved(df) -> float
def compute_velocity_stats(df) -> dict

# Shoaling
def compute_inter_individual_distance(subjects_dfs) -> pd.DataFrame
def compute_nearest_neighbor_distance(subjects_dfs) -> pd.DataFrame
def compute_group_polarity(subjects_dfs) -> pd.DataFrame

# Open Field
def compute_center_time_ratio(df, center_zone) -> float
def compute_thigmotaxis_index(df, arena_radius) -> float

# EPM
def compute_open_arm_time_ratio(df, open_arm_zones) -> float

# 范式调度器
def compute_paradigm_metrics(parsed_data, paradigm, groups, metrics=None) -> dict

# 导出
def save_to_csv(metrics_result, path) -> str
```

### 5.3 statistics.py — 统计检验（P3）

```python
def test_normality(values, alpha=0.05) -> dict
def compare_two_groups(g1, g2, alpha=0.05, paired=False) -> dict
def compare_multiple_groups(groups, alpha=0.05) -> dict
def compare_groups(metrics, groups, metrics_to_test=None, alpha=0.05, correction="bonferroni") -> dict
def compute_cohens_d(g1, g2) -> dict
def compute_eta_squared(groups) -> dict
```

### 5.4 charts.py — 图表生成（P2）

```python
def box_plot(metrics, metrics_to_plot, significance=None, output_path=...) -> str
def bar_chart(metrics, metrics_to_plot, error_type="sem", ...) -> str
def violin_plot(metrics, metrics_to_plot, ...) -> str
def trajectory_plot(df, ...) -> str
def add_significance_markers(ax, comparisons) -> None
```

### 5.5 assess.py — 结果评估（P3）

被 data-analyst subagent 的 skill 引用，也可被代码直接调用：

```python
def assess_results(statistics_results, paradigm, knowledge_dir="knowledge/") -> dict
```

### 5.6 上传 handler 修改（P1）

`packages/agent/backend/app/gateway/routers/uploads.py`：在 markitdown 转换前检测 EthoVision 格式，跳过转换。

---

## 6. Subagent 注册（P3）

### 6.1 code-executor

```python
# deerflow/subagents/builtins/code_executor.py

CODE_EXECUTOR_CONFIG = SubagentConfig(
    name="code-executor",
    description=(
        "Code execution specialist for behavioral data analysis. "
        "Selects pre-built analysis templates, adapts code to specific data and requirements, "
        "executes in sandbox, and writes structured handoff with output file paths."
    ),
    system_prompt="""You are a code execution specialist for behavioral data analysis.

<workflow>
1. Read the task from the lead agent — understand paradigm, groups, file paths, user requirements
2. Read the appropriate template:
   read_file("/path/to/ethoinsight/templates/{paradigm}.py")
3. Read the first 20 lines of one data file to confirm format:
   read_file("/mnt/user-data/uploads/轨迹-xxx.txt", max_lines=20)
4. Adapt the template:
   - Modify the PARAMETERS section (groups, metrics, output paths)
   - Add/modify analysis code for custom user requirements
   - Write the adapted script to workspace
5. Execute: bash("python /mnt/user-data/workspace/analysis_script.py")
6. Check output files exist
7. Write handoff JSON to /mnt/user-data/workspace/handoff_code_executor.json

IMPORTANT: Return the handoff file path and a brief summary as your final message.
</workflow>

<error_handling>
If execution fails:
1. Read the error message carefully
2. If format issue: read_file first 500 lines of data file, understand actual format, fix code
3. If dependency issue: pip install in sandbox
4. If logic error: analyze and fix
5. Retry up to 3 times
6. If still failing: write handoff with status "failed" and error details
</error_handling>

<principles>
- Never guess data values — all computation via code
- Template parameters at top of script for easy modification
- Always verify output files exist before writing handoff
</principles>""",
    tools=["bash", "read_file", "write_file", "ls", "str_replace"],
    disallowed_tools=["task", "ask_clarification", "present_files",
                       "web_search", "web_fetch", "image_search"],
    model="inherit",
    max_turns=25,
    timeout_seconds=600,
)
```

### 6.2 data-analyst

```python
# deerflow/subagents/builtins/data_analyst.py

DATA_ANALYST_CONFIG = SubagentConfig(
    name="data-analyst",
    description=(
        "Behavioral data analysis expert. Interprets code execution results using "
        "Noldus domain knowledge, noldus-kb literature, and statistical expertise. "
        "Produces analytical insights and conclusions."
    ),
    system_prompt="""You are a behavioral data analysis expert with deep Noldus domain knowledge.

<workflow>
1. Read the task from lead agent — understand what analysis is needed
2. Read code-executor's output files (NOT the handoff, the actual data):
   - read_file metrics CSV, statistics JSON
   - Understand the statistical results
3. Apply domain knowledge (from your skills) to interpret results:
   - Are the effect sizes practically meaningful?
   - Are there confounding factors?
   - How do results compare with published literature?
4. Optionally search noldus-kb for relevant papers:
   bash("noldus-kb search 'keywords' --format json")
5. Write detailed analysis to /mnt/user-data/workspace/analysis/analysis_report.md
6. Write handoff JSON to /mnt/user-data/workspace/handoff_data_analyst.json

IMPORTANT: Return the handoff file path and a brief summary as your final message.
</workflow>

<principles>
- 行为学核心方法论是组间对比，不是绝对阈值
- 检查混杂因素（运动量异常可能影响焦虑指标）
- 引用 noldus-kb 中的真实论文，不编造引用
- 区分统计显著和实际意义
</principles>""",
    tools=["bash", "read_file", "write_file", "ls",
           "web_search", "web_fetch"],
    disallowed_tools=["task", "ask_clarification", "present_files"],
    model="inherit",
    max_turns=20,
    timeout_seconds=600,
)
```

### 6.3 report-writer

```python
# deerflow/subagents/builtins/report_writer.py

REPORT_WRITER_CONFIG = SubagentConfig(
    name="report-writer",
    description=(
        "Scientific report writer. Reads data analysis outputs and analytical insights, "
        "writes publication-ready Results and Discussion sections."
    ),
    system_prompt="""You are a scientific report writer for behavioral neuroscience.

<workflow>
1. Read the task from lead agent
2. Read code-executor's data outputs:
   - metrics CSV, statistics JSON, chart file paths
3. Read data-analyst's analysis document:
   - /mnt/user-data/workspace/analysis/analysis_report.md
4. Write publication-ready report:
   - Results section: APA-format statistical reporting, reference figures
   - Discussion section: interpret findings, compare with literature, note limitations
5. Save report to /mnt/user-data/workspace/output/report.md
6. Write handoff JSON to /mnt/user-data/workspace/handoff_report_writer.json

IMPORTANT: Return the handoff file path and a brief summary as your final message.
</workflow>

<formatting>
Statistical results: "The treatment group showed significantly higher IID
(M = 45.2, SD = 12.3) compared to controls (M = 32.1, SD = 15.7),
t(10) = 2.34, p = .031, d = 0.85."

Figure references: "As shown in Figure 1, ..."
</formatting>""",
    tools=["read_file", "write_file", "bash", "ls"],
    disallowed_tools=["task", "ask_clarification", "present_files",
                       "web_search", "web_fetch"],
    model="inherit",
    max_turns=15,
    timeout_seconds=300,
)
```

### 6.4 注册

修改 `deerflow/subagents/builtins/__init__.py`：

```python
from .bash_agent import BASH_AGENT_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
}
```

---

## 7. Lead Agent 改造（P4）

### 7.1 System Prompt 修改

**文件**: `deerflow/agents/lead_agent/prompt.py`

**改动 1**: `agent_name` 默认值 `"DeerFlow 2.0"` → `"EthoInsight"`

**改动 2**: 新增 `<orchestration_guide>` 段落，指导 lead agent 如何派遣：

```markdown
<orchestration_guide>
## 数据分析派遣流程

当用户上传 EthoVision 数据并请求分析时：

### Step 0: 确认需求
- 确认实验范式（从文件名/上下文推断，但要向用户确认）
- 确认分组定义（哪些 Subject 是对照/实验组）
- 确认用户具体需求（标准组间比较？相关性？自定义指标？）
- 如果信息不足，使用 ask_clarification 工具提问

### Step 1: 派遣 code-executor
task(subagent_type="code-executor", description="执行数据分析代码",
     prompt="任务描述，包含：\n- 范式: {paradigm}\n- 文件路径: {file_pattern}\n- 分组: {groups_json}\n- 用户需求: {requirements}\n- 模板路径: /path/to/ethoinsight/templates/{paradigm}.py")

### Step 2: 读 handoff，派遣 data-analyst
读取 /mnt/user-data/workspace/handoff_code_executor.json
确认 status == "completed"
task(subagent_type="data-analyst", description="分析实验数据",
     prompt="任务描述 + output 文件路径列表")

### Step 3: 读 handoff，派遣 report-writer
读取 /mnt/user-data/workspace/handoff_data_analyst.json
task(subagent_type="report-writer", description="撰写分析报告",
     prompt="任务描述 + output 文件路径 + analysis_report.md 路径")

### Step 4: 整合返回用户
读取 /mnt/user-data/workspace/handoff_report_writer.json
读取报告内容，整合为用户回复
</orchestration_guide>
```

**改动 3**: subagent 列表从硬编码改为动态生成

**改动 4**: prompt 段落顺序优化（静态前置 → 利用 GLM-5.1 前缀缓存）

### 7.2 ThreadState 扩展

`deerflow/agents/thread_state.py` 新增：

```python
experiment_context: NotRequired[dict | None]      # 实验元数据
analysis_results: NotRequired[list[dict] | None]  # 分析结果历史
```

### 7.3 ExperimentContextMiddleware

新文件 `deerflow/agents/middlewares/experiment_context.py`：

从上传的 EthoVision 文件自动提取 header 元数据（实验名、范式、Subject 列表等），注入 Agent 上下文。帮助 lead agent 在提问前就有基本信息。

插入 `_build_middlewares()` 中 UploadsMiddleware 之后。

---

## 8. 记忆调优 + Skills（P5）

### 8.1 记忆提取 prompt

`deerflow/agents/memory/prompt.py` 追加行为学领域 fact 提取规则：实验参数、统计发现、分析结论、用户假设。

### 8.2 Skills

`skills/custom/ethoinsight/SKILL.md`：

data-analyst subagent 的领域 skill，包含：
- Noldus 独有的分析方法论
- 各范式的解读指南
- 混杂因素检查清单
- 效应量解读参考

report-writer subagent 的 report skill，包含：
- 论文 Results/Discussion 写作模板
- APA 统计报告格式
- 图表引用规范

### 8.3 YAML 知识库

```
knowledge/
├── paradigm_metrics.yaml    # 各范式标准指标定义
├── confounders.yaml         # 混杂因素检查清单
├── effect_size.yaml         # 效应量解读参考
└── statistics_rules.yaml    # 统计方法选择规则
```

---

## 9. 改动文件汇总

### 新增文件

| 文件 | 阶段 | 说明 |
|------|------|------|
| `ethoinsight/parse.py` | P1 | EthoVision 解析 |
| `ethoinsight/utils.py` | P1 | 通用工具（列名映射等） |
| `ethoinsight/metrics.py` | P2 | 指标计算 |
| `ethoinsight/charts.py` | P2 | 图表生成 |
| `ethoinsight/statistics.py` | P3 | 统计检验 |
| `ethoinsight/assess.py` | P3 | 结果评估 |
| `ethoinsight/templates/shoaling.py` | P2 | 斑马鱼模板（+ 其他 10 种范式） |
| `ethoinsight/tests/` | P1-P3 | 测试文件 |
| `deerflow/subagents/builtins/code_executor.py` | P3 | code-executor subagent |
| `deerflow/subagents/builtins/data_analyst.py` | P3 | data-analyst subagent |
| `deerflow/subagents/builtins/report_writer.py` | P3 | report-writer subagent |
| `deerflow/agents/middlewares/experiment_context.py` | P4 | 实验上下文中间件 |
| `skills/custom/ethoinsight/SKILL.md` | P5 | 领域 skills |
| `knowledge/*.yaml` | P5 | YAML 知识库 |

### 修改文件

| 文件 | 阶段 | 改动 |
|------|------|------|
| `config.yaml` | P0 | GLM-5.1 配置（已完成） |
| `app/gateway/routers/uploads.py` | P1 | EthoVision 检测，跳过 markitdown |
| `deerflow/subagents/builtins/__init__.py` | P3 | 注册 3 个新 subagent |
| `deerflow/agents/lead_agent/prompt.py` | P4 | 角色→EthoInsight, orchestration guide, 动态 subagent 列表, prompt 顺序 |
| `deerflow/agents/thread_state.py` | P4 | 扩展 experiment_context, analysis_results |
| `deerflow/agents/lead_agent/agent.py` | P4 | 插入 ExperimentContextMiddleware |
| `deerflow/agents/memory/prompt.py` | P5 | 行为学领域 fact 提取规则 |

---

## 10. 端到端验证

1. **上传**: 35 个斑马鱼轨迹文件
2. **Lead Agent 提问**: "检测到 Shoaling 实验数据，5 个 Subject。请确认分组？"
3. **用户回答**: "Subject 1-2 对照组，3-5 实验组"
4. **Step 1**: Lead Agent → code-executor（含范式、分组、文件路径）
5. **code-executor**: 读 shoaling.py 模板 → 修改参数 → 执行 → 写 handoff + output 文件
6. **Step 2**: Lead Agent 读 handoff → data-analyst（含 output 文件路径）
7. **data-analyst**: 读 metrics/statistics → 用领域知识洞察 → 查 noldus-kb → 写 handoff + 分析文档
8. **Step 3**: Lead Agent 读 handoff → report-writer（含 output + 分析文档路径）
9. **report-writer**: 读所有数据 → 写论文级报告 → 写 handoff
10. **Lead Agent**: 读报告 → 整合 → 返回用户完整回复
11. **记忆**: 保存实验参数和统计发现
12. **追问**: "上次 IID 差异显著吗？" → 从记忆回答
