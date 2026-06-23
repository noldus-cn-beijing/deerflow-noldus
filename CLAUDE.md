# CLAUDE.md

本文档为 Claude Code 在 `noldus-insight` 仓库工作时提供上下文。

## 项目定位

**EthoInsight** — 面向行为学研究员的 AI 分析助手。研究员上传 EthoVision XT 导出的轨迹数据，Agent 自动完成统计分析、专业解读、APA 格式报告生成。

- **当前状态**：端到端流水线可用，v0.1 已支持 6 个哺乳动物焦虑/抑郁范式（EPM/OFT/LDB/FST/Zero Maze/TST）；其余范式（鱼类如 shoaling/aquatic_open_field/cross_maze_fish/3d_swimming、学习记忆类如 MWM/Barnes/Y/T maze、社会/新物体、PhenoTyper 居家、昆虫旷场等）**暂未支持** — 关键词识别后 agent 会明示用户「v0.1 未实现」并反问。**EV19 模板识别地基设计已完成、实施计划已就绪**（详见 [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) 和配套 plan）。**用户自定义分析区列的 HITL 列语义对齐 Sprint 1 已合 dev**（真实 OFT 数据中 `中心区`/`边缘区` 等用户命名列经反问对齐后可正确算指标；结构聚合 Sprint 2 等行为学专家，见 [milestone](docs/milestone/column-semantics-alignment.md)）
- **愿景**：从"数据分析工具"演进为"全生命周期行为学研究助手"（实验指导 → 数据分析 → 追问 → 知识问答 → 跨范式证据链）
- **关键里程碑**：2026 年 9 月 v0.1 可用版本
- **路线图 / 全局进展**：见 [docs/milestone/README.md](docs/milestone/README.md)（**milestone 索引即本项目的 roadmap** — 每个 feature track 的当前状态、阻塞、最新 handoff 都在这里；不存在独立的 `roadmap.md`）
- **当前 v0.1 推进的真实阻塞**：两条路都卡在**行为学同事的范式方法论待产出**（不是工程卡点）——① **结构聚合**（自定义分区粒度按范式聚合，[Issue #98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)，需逐范式确认聚合语义）② **Golden Cases**（微调 benchmark + 回归种子，[Issue #90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90)）。详见 [docs/milestone/blocked-on-expert-methodology.md](docs/milestone/blocked-on-expert-methodology.md)。harness/基础设施层均无此阻塞，可独立推进。

## 仓库结构

```
noldus-insight/
├── packages/
│   ├── agent/              # DeerFlow fork（LangGraph agent 框架，作为 subtree 引入）
│   │   ├── backend/        # Python 后端（LangGraph + Gateway）
│   │   ├── frontend/       # Next.js 前端
│   │   ├── skills/
│   │   │   ├── public/     # 上游公共 skill（已提交）
│   │   │   └── custom/     # 项目定制 skill（**在 git 中**，ethoinsight* 等）
│   │   ├── config.yaml     # 主配置（模型、工具、sandbox 等）
│   │   └── extensions_config.json  # MCP 服务器 + skill 启用状态
│   └── ethoinsight/        # 行为学数据分析库（Python）
│       ├── ethoinsight/
│       │   ├── parse/             # EthoVision XT 文件解析（支持 TXT/CSV/XLSX/XLS）
│       │   ├── metrics.py        # 行为指标计算
│       │   ├── statistics.py     # 统计决策树（Shapiro-Wilk → 自动选择参数/非参数）
│       │   ├── charts.py         # 发表级图表生成
│       │   ├── assess.py         # 领域阈值判断（正常/异常）
│       │   └── templates/        # 范式模板
│       └── tests/
├── docs/
│   ├── milestone/          # 里程碑索引（**即本项目的 roadmap / 全局进展地图**，README.md 是入口）
│   ├── architecture-diagram.md  # 架构图（上层价值 + 下层技术）
│   ├── plans/              # 设计文档
│   ├── specs/              # 技术规格
│   ├── handoffs/           # 会话交接文档（按月份分目录：2026-04/ 2026-05/ …）
│   ├── problems/           # 问题记录
│   ├── sop/                # 操作手册
│   └── EthoInsight-技术文档/
├── demo-data/              # 测试用的 EthoVision 导出数据
├── golden-cases/           # 行为学专家标注的"黄金标准" case（SCHEMA.md 定义结构）
└── scripts/
    ├── sync-deerflow.sh        # DeerFlow 上游选择性同步工具
    └── validate_golden_case.py # golden-case schema 校验脚本
```

## 架构核心

### Agent 分析流水线

```
Lead Agent（deepseek-v4-pro，路由判断：有数据→分析，无数据→知识）
    ↓
code-executor（按 lead 生成的 metric_plan.json 逐条 bash 调用对应脚本：python -m ethoinsight.scripts.<paradigm>.<script_name>。指标清单来自 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml（single source of truth），由 lead 通过 catalog.resolve CLI 在派遣 code-executor 之前生成 plan.json。中间状态经 /mnt/user-data/workspace/ 文件传递。详见 docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md）
    ↓
data-analyst（审核统计方法、排查混杂因素、发现洞察）
    ↓
report-writer（结构化简单报告 + 文献引用）
```

专家思维分三层：

1. **自动统计决策树**（`ethoinsight/statistics.py`）— Shapiro-Wilk 正态性检验 → 自动选择参数/非参数检验
2. **领域知识驱动解读**（Skills + `ethoinsight/assess.py`）— 表型推断、混杂因素排查、效应量判断
3. **质量审核关卡**（data-analyst + 两层指标验证）— 统计方法适配性检查、异常检测
   - **两层指标验证（catalog-driven，2026-06-06 落地）**：L-A（`ethoinsight/validate.py`）进程内 `emit_result` 只查 NaN/Inf（name-agnostic 安全网，拿不到 paradigm）；L-B（`ethoinsight/validate_catalog.py`）在 code-executor 层调 `python -m ethoinsight.validate_catalog --plan`，按 catalog `output_unit` 做范围校验（ratio→[0,1]/count→≥0整数/物理单位→≥0），**按 subject 逐条验证**、**直接用 plan 自带的 output_unit**。两层都把 `VALIDATION_ERROR` 行汇入 data_quality_warnings（`code=METRIC_VALIDATION`）供 data-analyst fast-fail。详见 [docs/design/2026-06-06-data-processing-methodology-design.md](docs/design/2026-06-06-data-processing-methodology-design.md)。

### 知识注入三层

1. **System prompt 注入**（静态）— skill reference 文件直接在 context 中
2. **Knowledge-assistant 专用 prompt** — 优先用 skill 知识，其次调 MCP
3. **noldus-kb MCP**（**当前禁用**）— 6200+ 论文，深度知识来源

### DeerFlow fork 策略

`packages/agent/` 是 DeerFlow 上游的 subtree fork，在其基础上做了定制：

- **受保护文件**（有 Noldus 定制）：`agents/lead_agent/prompt.py`、`subagents/builtins/__init__.py`、`mcp/tools.py`、`sandbox/tools.py` 等
- **同步方式**：用 `scripts/sync-deerflow.sh` 全量跟随上游改动，受保护文件做 surgical 守护定制（deerflow 是 infra 底座，默认全合；详见下方「同步核心规则」）
- **上游架构详情**：见 [packages/agent/backend/CLAUDE.md](packages/agent/backend/CLAUDE.md)

## 常用命令

### 启动完整应用（项目根）

```bash
cd packages/agent
make dev           # 启动所有服务（Gateway-embedded 运行时 + Frontend + Nginx，3 进程），访问 localhost:2026
make stop          # 停止所有服务
```

> **运行时模式（2026-06 起跟随上游）**：根目录 `make dev`（经 `scripts/serve.sh --dev`）与生产 compose 默认走 **Gateway-embedded 模式** —— agent runtime 内嵌在 Gateway（`uvicorn app.gateway.app:app`），**没有独立的 LangGraph server 进程**。nginx 把公网 `/api/langgraph/*` 改写后转发给 Gateway 内嵌 runtime。Standard 模式（独立 `langgraph dev` on :2024）仍保留用于后端单独调试（见下「后端开发」的 `make dev`），但**不是默认部署形态**。

### 后端开发（backend 目录）

```bash
cd packages/agent/backend
source .venv/bin/activate
make dev           # 仅 LangGraph server (port 2024)
make gateway       # 仅 Gateway API (port 8001)
make test          # 运行所有后端测试
make lint          # ruff 检查
```

### ethoinsight 库

```bash
cd packages/ethoinsight
pytest tests/      # 运行分析库测试
```

### DeerFlow 上游同步

```bash
./scripts/sync-deerflow.sh --dry-run       # 预览
./scripts/sync-deerflow.sh                 # 交互式合入
```

#### ⚠️ 同步时的核心规则:全量跟随上游 + surgical 守护 Noldus 定制

**deerflow 是我们的 infra 底座,不是外部参考库**(2026-06-02 策略锁定)。我们整个 agent harness 站在它肩上,它的 bug fix / 性能改进 / 新能力**默认全要**——它修的坑就是我们迟早会踩的坑(event-loop 阻塞、SSE、checkpointer、中间件全是地基)。

**默认全量合,例外是受保护文件做 surgical**(不是"默认跳过、挑着合")。理由:挑着合 = 主动让底座和上游分叉,分叉越久未来 sync 越痛,且会错过"还没意识到需要"的修复。即使某 commit 当前用不上(如 postgres pool / 非 deepseek provider),只要落在**非受保护文件**,合进来零害(不走那条路就不触发)且消除下次 sync 的冲突点——为"用不上"主动跳过等于人为制造分叉。

**唯一必须 surgical(逐处对比、绝不整文件覆盖)的,是含 Noldus 独特改动的受保护文件**(清单 `scripts/sync-deerflow.sh:51` 的 22 个)。这些文件上游也改时,合上游 fix **保留所有 Noldus 定制**。真正可跳过的只有纯 docs。

Noldus 独特改动包括(但不限于):

- **Prompt 与提示词**:`agents/lead_agent/prompt.py`(中文调度规则、subagent 描述、Gate 反问机制)、subagent 系统 prompt、tool description 中文化
- **Subagent 名字与注册**:`subagents/builtins/__init__.py`(注册的 4 个 ethoinsight 子代理:`code-executor`、`data-analyst`、`report-writer`、`knowledge-assistant` 等)
- **自定义中间件**:`ArchivingSummarizationMiddleware`、`ThinkTagMiddleware`、`TrainingDataMiddleware`、`GateEnforcementMiddleware`、`SealGateMiddleware`、`DegradationCircuitBreakerMiddleware`(数据降级熔断器，挂 lead 链，检测 code-executor handoff 的 `gate_signals.statistics_status=="crashed"` → 自救一次 → 转 HITL `ask_clarification`；spec 2026-06-17 P2+P7)等(它们出现在 `lead_agent/agent.py` 的中间件链里,上游没有)
- **Sandbox 接口扩展**:`sandbox/sandbox.py` 的 `extra_env` 参数、`local_sandbox.py` 的 venv PATH + `DEERFLOW_PATH_*` 环境变量、`sandbox/tools.py` 的 `{{shared://}}` 占位符
- **Shared workspace 路径**:`config/paths.py` 的 `/mnt/shared`、`shared_dir()`、`thread_state.py` / `thread_data_middleware.py` 的 `shared_path` 字段
- **错误处理增强**:`llm_error_handling_middleware.py` 的总超时上限 + 多种 timeout 关键字识别
- **Skill 系统**:`skills/custom/` 下 5 个 ethoinsight 定制 skill 的注册和加载逻辑（`ethoinsight`、`ethoinsight-code`、`ethoinsight-charts`、`ethoinsight-planning`、**新增 `ethovision-paradigm-knowledge` — EV19 模板识别 + 学术范式映射的渐进披露知识库**）
- **MCP / 工具截断**:`mcp/tools.py` 的 4096 字符截断
- **Subagent executor 修复**:`subagents/executor.py` 的 `recursion_limit` 修复 + `max_turns` 硬限制 + seal-resume 失败后的确定性 auto-seal 兜底（`_attempt_auto_seal_from_artifacts`，2026-06-08）

**正确做法**(取长补短):

1. 先 `diff <(git show deerflow/main:<上游路径>) <本地路径>` 看具体差异
2. 识别上游的「真正修复」(常是几行 try/except、几行 import、几行边界检查)
3. **手工编辑本地文件**,只把上游的修复点合入,**保留所有 Noldus 定制代码原样不动**
4. 改完跑 `make test` 验证;如果上游修复带了配套 test,把那个 test 拿过来一起加
5. **上游改动涉及 Tier 4 体系**(`runtime.user_context` / `persistence.*` / `runtime.checkpointer` / `runtime.events` 等):**本仓库已吃下 Tier 4**(persistence 模块齐全,多用户研究助手,见第 13 条),这些**可以跟随上游**,不再"整 PR 跳过"。只需确认上游改动没碰到我们的受保护定制(如 thread_state 的 shared_path 字段)。**例外**:若上游改动依赖了我们**确实没有**的新子系统(unified auth + better-auth 等我们没接的),那一处仍 surgical 隔离或跳过,在交接文档记录。

**错误做法**(永远禁止):

- ❌ `git show deerflow/main:<file> > <local_file>` 直接覆盖含 Noldus 定制的**受保护**文件(非受保护文件全量覆盖正是我们要的)
- ❌ 使用 `./scripts/sync-deerflow.sh --auto-apply` 而不对受保护文件做人工 surgical(脚本「安全文件」分类用于提示"哪些要 surgical",非受保护文件可在交互里全 Y)
- ❌ 接受上游 `lead_agent/agent.py` 整文件 — 它包含中间件链顺序,直接覆盖会丢失 Noldus 的定制中间件
- ❌ 接受上游 `prompt.py` 整文件 — 你会立刻丢掉所有中文调度规则和 ethoinsight subagent 描述

**血泪教训**(2026-05-06 同步实测,⚠️ **部分前提已随 Tier4 合入而变**,见下):

- 上游脚本把 `runtime/user_context.py`、`runtime/runs/manager.py`、`agents/memory/storage.py`、`tools/builtins/setup_agent_tool.py` 等标为「安全文件」——**2026-05-06 时**它们引 `persistence/*`(当时 Noldus 没有)会 ImportError;**但 2026-05-07/08 Tier234 合入后我们已有 persistence**,此风险大幅消解,这些文件现在多可跟随(仍 grep 确认无受保护定制)
- 上游 `view_image_tool.py` 引用 sandbox/tools.py 中 Noldus 还没合入的新函数,直接拉取会运行时炸 → **教训仍有效**:全量合后跑 `make test`,缺函数会立刻暴露
- 上游 `local_sandbox.py` 不接受 Noldus 定制的 `extra_env` 参数,直接覆盖会让 bash 工具立刻报 TypeError(test_client_live 立刻发现)→ **教训仍有效**:`local_sandbox.py` 含 extra_env 定制,属受保护,surgical

**判断「需 surgical 隔离」的简易方法**:

如果上游文件 import 了下列模块**且我们没有对应实现**,该处 surgical(我们已有的 persistence/checkpointer 不在此列,可跟随):

```python
from deerflow.runtime.user_context import ...     # per-user filesystem isolation
from deerflow.persistence import ...              # SQLAlchemy 持久化层
from deerflow.runtime.events import ...           # event store
from deerflow.runtime.checkpointer import ...     # 新 checkpointer 抽象
from deerflow.runtime.journal import ...          # auth-related journal
from deerflow.utils.time import ...               # 与 persistence 的 ISO8601 配套
from deerflow.config.database_config import ...   # 数据库配置
from deerflow.config.run_events_config import ... # event store 配置
from deerflow.skills.storage import ...           # Tier 4 重构的 skill storage
```

#### ⚠️ harness 模块顶层 import 闭环风险（2026-06-08 实证）

harness 模块图存在**已知导入环**（证据：`backend/tests/conftest.py` 为破 `task_tool → subagents` 环而在 `sys.modules` mock 了 `deerflow.subagents.executor`）。**任何在 `subagents/`、`tools/builtins/`、`agents/` 等核心模块顶层新增 `from deerflow...import` 都可能闭环**。一旦闭环：

- 生产启动（uvicorn `import app.gateway` / langgraph `make_lead_agent`）抛 `ImportError: cannot import name ... partially initialized module`，**Gateway 起不来**，`make dev` 卡在 `Waiting for Gateway on port 8001`。
- **但 pytest 全绿是假绿**——conftest 的 mock 把 executor 短路了，测试里那条环根本不触发。

**铁律**：
1. 改完 `subagents/executor.py`、`tools/builtins/*`、`agents/**` 等核心后，**除 `make test` 外必须裸导入两生产入口验证**（backend/ 下，无 conftest）：
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 0 退出才算过。
2. 新 helper 即使抽成纯函数，**import 它的那行别放模块顶层**——放函数体内**惰性 import**（同「所有 ethoinsight import 必须惰性放函数体」的处方）。
3. 永久 guard 已就位：`tests/test_gateway_import_no_cycle.py`（subprocess 裸导入两入口、绕开 conftest mock）。CI 会抓这类回归。

## 开发规范

### Python

- **Python 3.10+**（ethoinsight）/ **3.12+**（agent backend）
- ruff 格式化，line length 240
- 双引号、空格缩进、类型注解

### 测试

- **TDD 强制**：每个新功能/bug 修复都必须带单测，放在 `packages/agent/backend/tests/` 或 `packages/ethoinsight/tests/`
- 运行：`make test`（agent backend）或 `pytest tests/`（ethoinsight）
- 现在测试全绿（test_client.py 的两个 pre-existing 失败已在 2026-04-17 修复）

### Git

- 分支模型：
  - `main` — 生产分支。PR merge 到 main 触发 GitHub Actions 自动 build & push 镜像到 ACR（`.github/workflows/build-push-acr.yml`）
  - `dev` — 日常开发分支。所有 commit 先进 dev
  - Sprint/feature 完成后从 dev 提 PR 到 main
- 提交前跑 `make test` 和 `make lint`
- commit message 用中文，简洁描述改动意图

### CI/CD

**当前实际部署方式（ACR 到位前）**: 本地 build → 镜像 tar 推送 ECS → docker compose up

- **触发**: 开发者在本地跑 `cd packages/agent && make deploy-tar`
- **构建**: 本地 `docker compose build`（linux/amd64），导出两个镜像（frontend + backend）为 gzip tar
- **传输**: rsync 镜像 tar + docker-compose.yaml + nginx.conf + skills + config 到 ECS `/opt/ethoinsight/`
- **部署**: 远程 `docker load` + `docker compose up -d`（frontend / gateway / nginx **三个**服务 — agent runtime 内嵌在 gateway，**无独立 langgraph 容器**，跟随上游 Gateway 化；provisioner 仅 K8s 模式用，部署链跳过）
- **反代**: ECS 上 1Panel + OpenResty，80/443 → 内部 127.0.0.1:2026
- **配置文件**: `config.yaml` / `extensions_config.json` / `.env` 存在开发者本地 `~/ethoinsight-prod/`，不进 git，部署时 rsync 到 ECS
- **SOP**: 见 [docs/sop/deploy-via-tar-sop.md](docs/sop/deploy-via-tar-sop.md)
- **所需 env vars**: `DEPLOY_HOST` / `DEPLOY_PATH` / `DEPLOY_CONFIG` / `DEPLOY_EXTENSIONS` / `DEPLOY_AGENT_ENV`

**未来 ACR pipeline（暂未启用）**: 待 ACR 权限到位后启用 `.github/workflows/build-push-acr.yml`：
- PR merge 到 `main` → GitHub Actions build & push 到 ACR `registry.cn-beijing.aliyuncs.com/ethoinsight/`
- ECS watchtower poll 检测新镜像 → `docker compose pull && up -d`
- 届时 `make deploy-tar` 退役，`deploy-via-tar.sh` 中 `docker load` 换 `docker compose pull`

### 文档

- 代码改动后同步更新相关 `docs/` 下文档
- 重要会话结束后在 `docs/handoffs/` 写交接文档（文件名 `YYYY-MM-DD-<topic>-handoff.md`）
- **写 handoff 时判断**：本次会话是否让某个 feature track 到达了 checkpoint（完成/阶段切换/阻塞解除）？如果是，在 handoff 末尾加 `## milestone 建议` 段，列出应创建/更新的 milestone 和关键摘要。下一 agent 读到后执行。用户也可随时说"为 XX 写 milestone"手动触发

### ⚠️ 改 harness / prompt / 控制流前的三大病理自检

来自 HarnessX 报告（arXiv 2606.14249，把 harness 改动形式化成符号空间的 RL，三个 RL 病理在改 harness 时复现）。**每次改 prompt / middleware / guardrail / 控制流前，过一遍这三问**——它们廉价、命中我们反复踩的坑：

1. **Reward hacking（奖励黑客）**：这个改动会不会让 LLM 学会"糊弄验收"而非真完成？
   - 我们踩过：chart-maker 曾**伪造 failed reason 抄旧 handoff**（产出一个能通过校验的失败叙述，而非真画图）。
   - 防法：验收看**真产物**不看 LLM 自述；兜底机制配**触发率可观测**（`sealed_by` 标记等），别让"学会依赖兜底"藏在全绿下。

2. **Catastrophic forgetting（灾难性遗忘）**：修 A 会不会悄悄回归 B？
   - 我们踩过：sync 时改一处 prompt **削弱了别处约束伞句**；`385f8989` 改 SSOT 一处**漏同步三处镜像文案**（PR#175 补的 drift）。
   - 防法：改共享组件（prompt 段 / Step dataclass / PATHS）先 **grep 所有消费者**，全量跑回归（seesaw：不回归已通过的），别只跑新测试。

3. **Under-exploration（探索不足）**：我是不是又在**只改 prompt 逃避结构改动**？
   - 我们踩过：seal 漏调**改了 4 次 prompt 打地鼠**才终于上 SealGateMiddleware（结构改动）。
   - **关键警告（HarnessX §6.6 Telecom 实证）**：累加 reminder/提醒规则会**亚阈值耦合致崩**（连加 5 条 reminder→第 6 条把合规率 100% 打回 80%，且 per-task 回归检测发现不了）。**别再用"加一条提醒 prompt"修反复漏调的问题——改终止条件/加确定性门（结构），不加规则。**
   - 反向自检：但**不是所有问题都该上结构门**——若结构层已正确（如工具已全量扫描），缺的只是 prompt 指引硬度，那 prompt 修法是对的（ETHO-5 即此类）。判据：**结构缺失→上门；结构已对、只缺指引→改 prompt**。

> 配套原则：**用确定性结构约束行为，不用 prompt 规则**（"LLM 提议，确定性门定生死"）。我们的确定性门 = DeerFlow 的 `GuardrailProvider` / `SealGateMiddleware` / `path_registry.PATHS` / Pydantic schema。详见 memory `reference_harnessx_report_and_etho_spec_application`。

## 重要注意事项

1. **skills/custom/ 是项目定制 skill 的目录** — ethoinsight 系列定制 skill **在 git 中**（上一任交接文档误标为 gitignored，实际并非如此）。**权威清单以 `packages/agent/skills/custom/` 目录 + `extensions_config.json` 的 `skills` 段为准**（不在此处手工枚举以免漂移）。当前含 `ethoinsight` / `ethoinsight-code` / `ethoinsight-charts` / `ethoinsight-chart-maker` / `ethoinsight-grouping` / `ethoinsight-metric-catalog` / `ethoinsight-lead-interaction` / `ethovision-paradigm-knowledge` / **`ethoinsight-column-confirmation`（新增 — EV19 自定义分析区列的 HITL 列语义对齐，见 [milestone](docs/milestone/column-semantics-alignment.md)）** 等。新增 skill 须三件一起：① 建文件 ② extensions_config 注册 ③ lead prompt 加触发/read 指引。
2. **noldus-kb 当前禁用** — `extensions_config.json` 里 `"enabled": false`，等 `180.184.84.124:7001` 恢复后再启用。禁用状态不要提交为 true
3. **受保护文件修改后同步要小心** — `scripts/sync-deerflow.sh` 会把它们标为"需人工判断"
4. **v0.1 是 9 月硬指标** — Phase 0（当前阶段）要完成 EPM + OFT 范式 + 鲁棒性验证 + 基础设施修复
5. **微调方案已锁定** — Qwen3-8B Dense + Fireworks.ai（SFT 先行，DPO 推迟到 v0.1 后）。**2026-05-13 提出升级提议**：基座改 Qwen3-30B-A3B-Instruct-2507 MoE + 客户硬件锁定 RTX 5090 32GB（放弃 5060 Ti 兜底）+ 后训练候选火山引擎 verl / Fireworks，详见 [docs/plans/2026-05-13-base-model-decision-memo.md](docs/plans/2026-05-13-base-model-decision-memo.md)，**待团队对齐前仍按原锁定方案执行**
6. **deepseek 的正面提示** — 不要用"禁止 X""不要 X"，会反向激活；必须用正面指令描述想要的行为
7. **训练数据飞轮已启动** — 每次 agent 会话自动录制到 `packages/agent/backend/.deer-flow/training-data/auto-collected/`；专家反馈走 `/api/threads/{tid}/runs/{rid}/feedback` API（**SQLite 后端，verdict 三分类 + revised_text**）+ 前端三按钮。查看累计进度：`cd packages/agent/backend && make training-stats`。详见 [docs/sop/training-data-flywheel-sop.md](docs/sop/training-data-flywheel-sop.md)。
8. **Golden-case 是专家知识注入的正式途径** — 行为学同事对一份数据标注"期望的分析结论"，同时承担**领域知识源 + 回归测试 + SFT 种子数据**三重角色。结构由 [golden-cases/SCHEMA.md](golden-cases/SCHEMA.md) 定义，模板在 `golden-cases/TEMPLATE/`，校验用 `python3 scripts/validate_golden_case.py`。**不要为范式知识另建文档系统（如 `docs/domain/`），所有专家领域知识统一走 golden-cases/**。详见 [docs/sop/golden-case-sop.md](docs/sop/golden-case-sop.md)。
9. **判读哲学：组间比较，不用绝对阈值** — 行为学同事确认：EPM/OFT 等焦虑范式的解读**只看 control vs treatment 是否有显著差异**，不按"正常范围 15-25%，小于 10% 就是高焦虑"这种绝对阈值判断。`ethoinsight/assess.py` 里的 `_DEFAULT_THRESHOLDS` 保留作为"批次质检参考"，**不作为判读依据**。agent 给出的结论必须基于统计检验 + 效应量。
10. **范式体系正在重构（2026-04-29 起）** — 旧体系是「7 大类 18 范式」学术分类（写在 `lead_agent/prompt.py` 里），与 EthoVision XT 19 真实模板（20 大类 62 变体）不对应，导致 Gate 1 反问机制不准。新体系采用**双层**：用户语言走 EV19 模板，内部走学术范式。**领域知识独立 skill**（`ethovision-paradigm-knowledge` 待建），by-template/by-experiment 双向索引由行为学同事 markdown 维护。
    - **设计文档（产品级）**：[docs/plans/2026-04-29-ev19-template-paradigm-design.md](docs/plans/2026-04-29-ev19-template-paradigm-design.md)
    - **设计文档（工程级，2026-05-08 完成）**：[docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) — 定义新 skill 架构 + GuardrailMiddleware 集成 + 软门 + 默认值降级
    - **实施计划（2026-05-08 完成）**：[docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md](docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md) — 14 个 task / 90 个 step / 完整代码 — 可由 agent 直接执行
    - **行为学同事 review 包**（已生成，等待补充）：[docs/review-packages/2026-04-29-ev19-templates/](docs/review-packages/2026-04-29-ev19-templates/) — 同事 PR 后会被搬入 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/`
    - **当前状态**：工程地基（spec + plan）已就绪，可启动实施；6 范式分析模板补全等同事 PR 后再做
    - **核心架构决策**：
      - 新 skill `ethovision-paradigm-knowledge` 作为 EV19 模板知识渐进披露入口（agent 主动 read_file）
      - 复用 deerflow 现成的 `LoopDetectionMiddleware`（防反复反问）+ `GuardrailMiddleware`（拦截 ev19_template=null 时的 code-executor 派遣）
      - 不删除现有 `GateEnforcementMiddleware`（管 paradigm 字段）；新 GuardrailMiddleware 与之职责正交（管 ev19_template 字段 + 锁定）
      - **agent 交互流程不变**：lead 通过 prompt + skill 决策 → 派遣 subagent → subagent 间 handoff JSON 文件传 hard fact
    - **影响范围**：`agents/lead_agent/prompt.py` Gate 1 段（删除旧 18 范式表）、`experiment_context.py`（set_experiment_paradigm 加 ev19_template 必填）、`packages/ethoinsight/ethoinsight/ev19_facts.py`（新增）、`packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py`（新增）、`packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md`
    - **不影响**：`ethoinsight/templates/*.py`（按学术范式组织的分析模板保留）、code-executor 流水线、Gate 2 数据质量检查、5 个 ethoinsight subagent 的注册和职责
11. **Memory event-loop 修复（已完成 2026-04-29）** — `RuntimeError: Event loop is closed` 已通过 sync 上游 `82731aeb` 彻底修复（memory 更新改 sync `model.invoke()`，不再创建短命 event loop）。详见 [docs/handoffs/2026-04/2026-04-29-event-loop-fix-v2-completed-handoff.md](docs/handoffs/2026-04/2026-04-29-event-loop-fix-v2-completed-handoff.md)。本地 fork 现在比上游更接近最新版。
12. **复用 deerflow 现成功能优先于自造轮子** — 实施新 agent 行为时，先调研 deerflow harness 已有的中间件 / 工具 / provider 协议，能复用就复用，不要重新发明。已知现成可用的关键能力：`ask_clarification` + `ClarificationMiddleware`（反问中断）、`LoopDetectionMiddleware`（防 tool call 死循环，已默认启用）、`GuardrailMiddleware` + `GuardrailProvider` 协议（pre-tool-call 授权决策）、`ToolErrorHandlingMiddleware`（tool 抛错自动转 error ToolMessage）、Skill 渐进披露（agent 主动 read_file SKILL.md + references/）、`update_agent` / `setup_agent` 工具（custom agent 自我修改 SOUL.md，v0.1 后启用）、`Skill Evolution`（agent 自建/改 skill，v0.1 后启用）、`/api/threads/{id}/runs/{rid}/feedback` API（替代手写飞轮反馈通道）。**自写中间件之前先看 `packages/agent/backend/packages/harness/deerflow/agents/middlewares/` 和 `tools/builtins/` 目录有没有现成的**。
13. **项目状态修正（2026-05-12）** — 本仓库已经吃下 Tier 4 体系（unified persistence、`@require_permission`、`get_effective_user_id`、`UserRow` 等），是**多用户**研究助手。CLAUDE.md 第 11 条之前提到的"v0.1 单用户故意不要 Tier 4"在 2026-05-07/08 Tier234 round1-3 合入后已过时——这些指导仍适用于评估上游 sync 风险，但**本仓库现状**是建立在 Tier 4 之上。
14. **Skill 优化 → SFT 路线（2026-06-04 启动）** — 采用微软 SkillOpt 方法论：行为学专家产出 Golden Cases（eval benchmark）→ 改造 SkillOpt 代码（自定义 EnvAdapter）→ 跑优化循环产出 best_skill.md → 用优化后的 skill 驱动 agent 生成高质量 SFT 轨迹 → 微调 Qwen3-30B。详见 [docs/plans/2026-06-04-skillopt-skill-optimization-plan.md](docs/plans/2026-06-04-skillopt-skill-optimization-plan.md)。**当前阻塞项：行为学专家 Golden Cases 待产出。**
15. **Subagent handoff 鲁棒性 + n=1 路由 + 判读语言对齐（2026-06-08 一批合入 dev）** — EPM dogfood 多轮诊断后的一组正交修复，全部已在 dev：
    - **seal 黑洞 harness 兜底**：subagent 完成产出（report.md / plot_*.png）却漏调 seal 工具时，harness 用已有产出**确定性 auto-seal** 构造 handoff，不再判 FAILED 白等重派。**仅 report-writer / chart-maker**（核心字段可从文件机械推导）；code-executor / data-analyst 的认知产物**永不 auto-seal**。实现见 `subagents/executor.py:_attempt_auto_seal_from_artifacts` + `seal_handoff_tools.py:_seal_handoff_to_workspace`（纯函数变体）。
    - **n=1 单样本路径**：每组 n<2 时 lead fast-path 跳过 data-analyst（无统计基础），但**用户主动要洞察仍派 data-analyst 走 partial 描述性路径**；`path_sequence` guardrail 感知 n=1（单 subject 时 data-analyst 非必需）；四个 handoff schema 的 `status` 补齐 `partial` 三态对齐。
    - **初次数据判读归 data-analyst**：knowledge-assistant 场景 A 收窄为"只解释**已有** data-analyst 结论"，不从零产完整判读（它不受输出宪法约束会产违禁词）；lead 按 workspace 有无 `handoff_data_analyst.json` 区分路由。
    - **判读语言宪法术语松绑**：松绑定性行为术语（焦虑样行为/趋近-回避），**保留**绝对阈值（正常范围 X-Y%）+ 绝对程度（高/低焦虑）禁令（守第 9 条铁律）。SSOT 在 `skills/custom/ethoinsight/references/output-constitution.md`。
    - **路径/展示一致性**：`validate_catalog` 用 `resolve_sandbox_path`（`scripts/_cli.py`，按 `DEERFLOW_PATH_*` env 解析）读 plan 内 `/mnt` 虚拟路径，修 `result_file_unreadable` 误报；`present_files` 工具拒绝磁盘上不存在的文件，防 LLM 幻影文件名（如 `epm_bar_{metric}.png`）污染 artifacts 致前端 404。
    - **⚠️ 教训**：seal auto-seal 抽的 helper 当初在 `executor.py` **顶层** import `seal_handoff_tools` 闭成导入环，生产启动（uvicorn `app.gateway.app`）崩 `partially initialized module`、`make dev` 卡 Waiting for Gateway；测试因 `conftest.py` mock 了 executor 而假绿没抓到。已改惰性 import + 加 `tests/test_gateway_import_no_cycle.py`（subprocess 裸导入两生产入口）。见下「同步核心规则」末尾。

## 快速上手

新开会话建议的读取顺序：

1. 本文档 — 了解全貌
2. [docs/milestone/README.md](docs/milestone/README.md) — **项目地图 / roadmap（即全局进展）**，每个 feature 当前状态 + 阻塞一目了然（2 分钟）
3. [docs/milestone/blocked-on-expert-methodology.md](docs/milestone/blocked-on-expert-methodology.md) — 当前等行为学同事方法论的两条阻塞（结构聚合 + golden case）的精确待办
4. 具体 feature 的 milestone — 深入了解某个 track 的全貌（5 分钟）
5. `docs/handoffs/` 下具体 handoff — 最细粒度的操作细节（按需）

## 相关文档

- [docs/milestone/README.md](docs/milestone/README.md) — **里程碑索引 = 项目 roadmap / 全局进展地图**（无独立 roadmap.md）
- [docs/milestone/blocked-on-expert-methodology.md](docs/milestone/blocked-on-expert-methodology.md) — 等同事方法论的阻塞清单（结构聚合 Issue #98 + golden case Issue #90）
- [docs/architecture-diagram.md](docs/architecture-diagram.md) — 架构图
- [docs/plans/2026-04-29-ev19-template-paradigm-design.md](docs/plans/2026-04-29-ev19-template-paradigm-design.md) — EV19 模板范式重定位设计（产品级）
- [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) — **EV19 模板识别地基设计（工程级，2026-05-08）**
- [docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md](docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md) — **EV19 模板识别地基实施计划（14 task / 90 step，可由 agent 直接执行）**
- [docs/review-packages/2026-04-29-ev19-templates/README.md](docs/review-packages/2026-04-29-ev19-templates/README.md) — 行为学同事 review 包入口
- [docs/specs/paradigm-analysis-tools-spec.md](docs/specs/paradigm-analysis-tools-spec.md) — 范式分析工具规格
- [golden-cases/SCHEMA.md](golden-cases/SCHEMA.md) — Golden-case 标注结构字典
- [docs/sop/golden-case-sop.md](docs/sop/golden-case-sop.md) — Golden-case 协作流程 SOP
- [docs/specs/llm-finetuning-strategy.md](docs/specs/llm-finetuning-strategy.md) — 微调策略
- [docs/plans/2026-04-13-fine-tuning-small-model-design.md](docs/plans/2026-04-13-fine-tuning-small-model-design.md) — 微调设计
- [docs/plans/2026-06-04-skillopt-skill-optimization-plan.md](docs/plans/2026-06-04-skillopt-skill-optimization-plan.md) — **SkillOpt 方法论：Golden Cases → Skill 优化 → SFT 数据 → 微调（5 阶段实施计划）**
- [docs/sop/deerflow-sync-sop.md](docs/sop/deerflow-sync-sop.md) — DeerFlow 同步 SOP
- [docs/refs/2026-05-22-mousegpt-paper-review.md](docs/refs/2026-05-22-mousegpt-paper-review.md) — MouseGPT 论文借鉴分析（2026-05-22）
- [packages/agent/backend/CLAUDE.md](packages/agent/backend/CLAUDE.md) — DeerFlow 后端架构细节
