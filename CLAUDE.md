# CLAUDE.md

本文档为 Claude Code 在 `noldus-insight` 仓库工作时提供上下文。

## 项目定位

**EthoInsight** — 面向行为学研究员的 AI 分析助手。研究员上传 EthoVision XT 导出的轨迹数据，Agent 自动完成统计分析、专业解读、APA 格式报告生成。

- **当前状态**：端到端流水线可用，v0.1 已支持 6 个哺乳动物焦虑/抑郁范式（EPM/OFT/LDB/FST/Zero Maze/TST）；其余范式（鱼类如 shoaling/aquatic_open_field/cross_maze_fish/3d_swimming、学习记忆类如 MWM/Barnes/Y/T maze、社会/新物体、PhenoTyper 居家、昆虫旷场等）**暂未支持** — 关键词识别后 agent 会明示用户「v0.1 未实现」并反问。**EV19 模板识别地基设计已完成、实施计划已就绪**（详见 [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) 和配套 plan）
- **愿景**：从"数据分析工具"演进为"全生命周期行为学研究助手"（实验指导 → 数据分析 → 追问 → 知识问答 → 跨范式证据链）
- **关键里程碑**：2026 年 9 月 v0.1 可用版本
- **路线图**：见 [docs/roadmap.md](docs/roadmap.md)

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
│   ├── roadmap.md          # 12 个月产品路线图
│   ├── prd.md              # 产品需求文档
│   ├── architecture-diagram.md  # 架构图（上层价值 + 下层技术）
│   ├── plans/              # 设计文档
│   ├── specs/              # 技术规格
│   ├── handoffs/           # 会话交接文档（按月份分目录：2026-04/ 2026-05/ …）
│   ├── milestone/          # 里程碑总结
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
3. **质量审核关卡**（data-analyst）— 统计方法适配性检查、异常检测

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
make dev           # 启动所有服务（LangGraph + Gateway + Frontend + Nginx），访问 localhost:2026
make stop          # 停止所有服务
```

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
- **自定义中间件**:`ArchivingSummarizationMiddleware`、`ThinkTagMiddleware`、`TrainingDataMiddleware`、`GateEnforcementMiddleware` 等(它们出现在 `lead_agent/agent.py` 的中间件链里,上游没有)
- **Sandbox 接口扩展**:`sandbox/sandbox.py` 的 `extra_env` 参数、`local_sandbox.py` 的 venv PATH + `DEERFLOW_PATH_*` 环境变量、`sandbox/tools.py` 的 `{{shared://}}` 占位符
- **Shared workspace 路径**:`config/paths.py` 的 `/mnt/shared`、`shared_dir()`、`thread_state.py` / `thread_data_middleware.py` 的 `shared_path` 字段
- **错误处理增强**:`llm_error_handling_middleware.py` 的总超时上限 + 多种 timeout 关键字识别
- **Skill 系统**:`skills/custom/` 下 5 个 ethoinsight 定制 skill 的注册和加载逻辑（`ethoinsight`、`ethoinsight-code`、`ethoinsight-charts`、`ethoinsight-planning`、**新增 `ethovision-paradigm-knowledge` — EV19 模板识别 + 学术范式映射的渐进披露知识库**）
- **MCP / 工具截断**:`mcp/tools.py` 的 4096 字符截断
- **Subagent executor 修复**:`subagents/executor.py` 的 `recursion_limit` 修复 + `max_turns` 硬限制

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
- **部署**: 远程 `docker load` + `docker compose up -d`（frontend / gateway / langgraph / nginx 四个服务）
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

## 重要注意事项

1. **skills/custom/ 是项目定制 skill 的目录** — `ethoinsight`、`ethoinsight-code`、`ethoinsight-charts`、`ethoinsight-planning`、`ethovision-paradigm-knowledge`（实施完成后）共 5 个定制 skill **在 git 中**（上一任交接文档误标为 gitignored，实际并非如此）
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

## 快速上手

新开会话建议的读取顺序：

1. 本文档 — 了解全貌
2. [docs/roadmap.md](docs/roadmap.md) — 了解 12 个月规划和 v0.1 里程碑
3. [docs/milestone/README.md](docs/milestone/README.md) — 项目地图，每个 feature 当前状态一目了然（2 分钟）
4. 具体 feature 的 milestone — 深入了解某个 track 的全貌（5 分钟）
5. `docs/handoffs/` 下具体 handoff — 最细粒度的操作细节（按需）

## 相关文档

- [docs/roadmap.md](docs/roadmap.md) — 产品路线图
- [docs/prd.md](docs/prd.md) — 产品需求文档
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
- [docs/sop/deerflow-sync-sop.md](docs/sop/deerflow-sync-sop.md) — DeerFlow 同步 SOP
- [docs/refs/2026-05-22-mousegpt-paper-review.md](docs/refs/2026-05-22-mousegpt-paper-review.md) — MouseGPT 论文借鉴分析（2026-05-22）
- [packages/agent/backend/CLAUDE.md](packages/agent/backend/CLAUDE.md) — DeerFlow 后端架构细节
