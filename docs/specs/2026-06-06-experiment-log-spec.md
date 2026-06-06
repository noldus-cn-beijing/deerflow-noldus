# AutoResearch 启发 — v0.1 实施 Spec

**日期**：2026-06-06（2026-06-06 Opus spec review 修正）
**基于**：[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 分析 + Opus 设计 review + DeerFlow 代码库 audit + Opus spec review
**状态**：经两轮 Opus review 修正，可指导实施

---

## 前置结论：DeerFlow 已有的能力（不需要做）

Audit 确认以下三个提议项 DeerFlow 已经完全实现，无需任何新代码：

| 提议 | 实际情况 | 证据 |
|------|---------|------|
| 差异化 max_turns | **已存在**。每个 subagent 的 `SubagentConfig.max_turns` 已分别配置：code-executor=40, data-analyst=12, report-writer=15, chart-maker=15, knowledge-assistant=6, general-purpose=100, bash=60 | `subagents/builtins/code_executor.py:177` 等 |
| ethoinsight/ 写保护 | **已存在，两层防护**：(A) sandbox `validate_local_tool_path` 拒绝 `/mnt/user-data/` 外的任何 `write_file`（`sandbox/tools.py:645-700`）；(B) `ScriptInvocationOnlyProvider._is_path_safe` 对 code-executor/bash 额外拒绝 `.venv`/`site-packages` 路径（`script_invocation_only_provider.py:161`） | 两层独立验证 |
| max_turns 配置机制 | **`SubagentConfig` 字段 + `config.yaml` 覆盖**。`subagents/config.py:36` 定义字段，`executor.py:933-936` 按 AI message 计数强制执行，`registry.py:92-99` 支持 config.yaml 三级覆盖 | 机制完整 |

---

## 实际需要实施的 Feature（共 5 项）

```
P0 (现在就能做，< 半天):
  S1. Subagent 级 LoopDetectionMiddleware        — 注意 thread_id 隔离
  S2. 代码级 metric validator                      — ~30 行 Python
  S3. data-analyst SKILL.md fast-fail 规则         — ~20 行 markdown
  S4. ethoinsight/ 写保护 red anchor test          — 3 个测试函数

P1 (v0.1 核心交付):
  S5. Experiment Log — ExperimentRow + seal + API + 前端
```

---

## S1: Subagent 级 LoopDetectionMiddleware

### 关键发现

`LoopDetectionMiddleware` 按 `thread_id` 跟踪历史（`loop_detection_middleware.py:264-280`）。Subagent 与 lead **共享** `thread_id`（`executor.py:877-879`）。如果共享同一个 middleware 实例，lead 和 subagent 的 tool call 计数会互相污染。

**解法**：在 `executor.py:_build_middlewares`（`executor.py:613`）中为每次 subagent 运行**实例化一个新的 `LoopDetectionMiddleware()`**。因为 middleware 的 `self._history` 是实例级 `OrderedDict`，新实例 = 干净的计数空间。Lead 保有自己的长生命周期实例（`agent.py:285`），两者隔离。

### 实施

```python
# subagents/executor.py, _build_middlewares() (~line 651)
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware

# 在 return 之前添加（在现有 GuardrailMiddleware 之后）
middlewares.append(LoopDetectionMiddleware())
```

### 验证

- 确认 subagent 内重复 tool call 被检测
- 确认 lead 的 tool call 历史不污染 subagent，反之亦然（thread_id 隔离测试）
- 确认 `recursion_limit` 自动调整（`calculate_subagent_recursion_limit` 统计 middleware hooks）

---

## S2: 代码级 Metric Validator

（与上一版 spec 相同，此处省略重复内容——见上一版 §S2）

### 实施

新建 `packages/ethoinsight/ethoinsight/validate.py`，在各 compute script 末尾调用 `validate_metrics(results)`。

---

## S3: data-analyst SKILL.md Fast-Fail 规则

（与上一版 spec 相同——见上一版 §S3）

---

## S4: ethoinsight/ 写保护 Red Anchor Test

（与上一版 spec 相同——见上一版 §S4）

---

## S5: Experiment Log（核心 Feature）

### 5.1 架构概览

```
                        experiment-context.json
                        (set_experiment_paradigm_tool 写入)
                              │
                              ├── paradigm, ev19_template
                              ├── analysis_config_id
                              └── column_semantics

                        handoff_code_executor.json
                        (code-executor seal 工具写入)
                              │
                              ├── metrics_summary: dict[str, dict[str, MetricStat]]
                              ├── groups: dict[str, str]  (e.g. {"Arena 1": "Treatment"})
                              └── data_quality_warnings

                        handoff_data_analyst.json
                        (data-analyst seal 工具写入)
                              │
                              ├── key_findings
                              └── quality_warnings

                        handoff_report_writer.json
                        (report-writer seal 工具写入)
                              │
                              └── status, report_path

                    ── seal 时刻 ──→  ExperimentRow (SQLite)
                                      用户可见
```

**核心原则**：ExperimentRow 是已有文件的**持久化投影**，不是第二条写入路径。从 experiment-context.json + handoff JSON 一次性读取、校验、写入 DB。

**文件读取复用已有函数**（不重新实现）：
- `experiment_context.read_context(workspace_dir)` → 返回 `experiment-context.json` 内容（`experiment_context.py:84-94`）
- `experiment_context.read_handoff(workspace_dir, thread_data)` → 读取 + schema 校验 `handoff_code_executor.json`（`experiment_context.py:113-171`）
- `experiment_context.resolve_workspace_from_state(state)` → 获取 workspace 路径（`experiment_context.py:102`）

### 5.2 Trigger：何时 Seal

**Trigger 1（主路径）— Middleware 轮询**：`ExperimentSealMiddleware.aafter_agent()` 在每次 lead agent turn 结束后检查 `handoff_report_writer.json` 是否存在。存在且 `status == "completed"` → seal。这是轮询模式（不是事件驱动），与 `TrainingDataMiddleware` 模式一致：每次 turn 都检查，但只有条件满足时才写。

**Trigger 2（反馈路径）— Feedback Router 内嵌**：用户提交 feedback（`POST /api/threads/{tid}/runs/{rid}/feedback`）后，在 `routers/feedback.py` 的 `submit_feedback` handler 内，调用一个同步的 seal 函数。此时已有 thread_id 和 user_id（来自 `get_current_user(request)`），可以直接读取 workspace 文件并写入 ExperimentRow。

**Trigger 3 明确不做**：thread 关闭 / N 分钟无交互——需要 scheduler/cron 基础设施，v0.1 不实施。如果分析未完成就关闭 thread，该 experiment 不会出现在列表中（没有 report-writer handoff = 未 seal）。后续可按需加兜底逻辑。

**幂等性**：同一个 `experiment_id` 多次触发 seal → upsert。首次写入所有字段；后续只更新 `updated_at` 和 `user_feedback`，不覆盖分析数据。

### 5.3 ExperimentRow Schema

```python
# persistence/experiment/model.py

from datetime import datetime, UTC
from sqlalchemy import JSON, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column
from deerflow.persistence.base import Base


class ExperimentRow(Base):
    __tablename__ = "experiments"

    # 标识
    experiment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # = f"{thread_id}:{analysis_config_id}"
    # analysis_config_id 由 set_experiment_paradigm_tool 写入 experiment-context.json
    # seal 前必须确认 analysis_config_id 存在（否则不 seal，防止 "thread_id:unknown" 孤儿行）
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # 范式 — 来自 experiment-context.json
    paradigm: Mapped[str | None] = mapped_column(String(64), index=True)
    ev19_template: Mapped[str | None] = mapped_column(String(128))

    # 数据指纹 — 来自 experiment-context.json
    analysis_config_id: Mapped[str | None] = mapped_column(String(16))

    # 分析结果 — 从 handoff JSON 直接投影（JSON 字段结构引用 handoff_schemas.py）
    metrics_summary: Mapped[dict | None] = mapped_column(JSON)
    # CodeExecutorHandoff.metrics_summary → {metric_name: {group_name: MetricStat}}
    key_findings: Mapped[list | None] = mapped_column(JSON)
    # DataAnalystHandoff.key_findings
    quality_warnings: Mapped[list | None] = mapped_column(JSON)
    # 聚合 code-executor + data-analyst 的 quality_warnings

    # 状态 — 从 report-writer handoff status 推导
    status: Mapped[str] = mapped_column(String(20), default="completed")
    # "completed" | "partial" | "failed"
    # 推导规则（_determine_status）:
    #   report-writer handoff status="completed" → "completed"
    #   data-analyst handoff status="partial" 或无 report-writer → "partial"
    #   任何 handoff status="failed" → "failed"
    user_feedback: Mapped[str | None] = mapped_column(String(20))
    # "accepted" | "revised" | "rejected" | null

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index("ix_experiments_user_paradigm", "user_id", "paradigm"),
        Index("ix_experiments_thread", "thread_id"),
    )
```

**关键设计决策**：
- `experiment_id = f"{thread_id}:{analysis_config_id}"` — analysis_config_id 来自 experiment-context.json，由 `set_experiment_paradigm_tool` 在 Gate 1 完成后写入。**seal 前必须有 analysis_config_id，否则不 seal**（防止产生 `thread_id:unknown` 孤儿行）
- **不设 `groups` 列**：experiment-context.json 没有 group role/label/n 的标准化字段。Group 信息分散在 `CodeExecutorInputs.groups`（raw mapping）和 `metrics_summary[*][*].n` 中，缺乏统一的 role+label+n 结构。v0.1 从 `metrics_summary` 的 key 中隐式获取 group 名称，不做独立列
- **不设 `run_id` FK**：一个 experiment 可关联多个 run（用户追问触发新 run），1:N 关系不能由单一 FK 表达
- JSON 字段引用已有 Pydantic schema 结构，不发明第三套字段名
- `user_id`: `nullable=False`，通过 `get_effective_user_id()` 获取。**注意**：`get_effective_user_id()` 在 contextvar 未设置时返回 `DEFAULT_USER_ID`（不抛异常）。middleware 中 contextvar 由 `make_lead_agent` 在 `agent.py:499` 设置，正常情况下不会 fallback。记录此假设，并在 seal 时添加非 default 断言

### 5.4 实现架构：两个执行上下文

**关键约束**：LangGraph Server 和 Gateway API 是**不同进程**。Middleware 运行在 LangGraph 进程，Router 运行在 Gateway 进程。两者不能共享 `app.state`。

因此 S5 有两套独立的数据库访问路径：

```
LangGraph 进程                          Gateway 进程
─────────────                          ────────────
ExperimentSealMiddleware               routers/experiments.py
  │                                      │
  ├─ get_session_factory()               ├─ get_experiment_repo(request)
  │  (from deerflow.persistence          │  (FastAPI DI, from app.state)
  │   .engine)                           │
  │                                      │
  ├─ 直接构造 ExperimentRepository(sf)    ├─ 使用注入的 repo
  │                                      │
  └─ aafter_agent (async!)               └─ async handler

routers/feedback.py (Trigger 2)
  │
  └─ submit_feedback 内调用 seal 函数
     (同一 Gateway 进程，共用 app.state.experiment_repo)
```

#### 5.4a Middleware 侧（Trigger 1）

```python
# agents/middlewares/experiment_seal_middleware.py

from deerflow.agents.middleware import AgentMiddleware
from deerflow.agents.middlewares.experiment_context import (
    read_context,
    read_handoff,
    resolve_workspace_from_state,
)
from deerflow.persistence.engine import get_session_factory
from deerflow.persistence.experiment.sql import ExperimentRepository
from deerflow.runtime.user_context import get_effective_user_id


class ExperimentSealMiddleware(AgentMiddleware[AgentState]):
    """每次 lead turn 后检查是否应 seal experiment record。
    
    轮询模式（非事件驱动）：检查 handoff_report_writer.json 是否存在。
    与 TrainingDataMiddleware 相同的 aafter_agent + swallow-exception 模式。
    """

    state_schema = AgentState  # 不需要扩展 ThreadState

    async def aafter_agent(self, state: AgentState, runtime) -> dict | None:
        try:
            return await self._maybe_seal(state, runtime)
        except Exception:
            logger.exception("ExperimentSealMiddleware: seal failed, continuing")
            return None  # 绝不抛异常

    async def _maybe_seal(self, state, runtime) -> dict | None:
        # 获取 workspace 路径（复用已有 helper）
        workspace = resolve_workspace_from_state(state)
        if not workspace:
            return None

        # 检查 report-writer handoff
        report_path = workspace / "handoff_report_writer.json"
        if not report_path.exists():
            return None

        # 读取 report handoff
        try:
            report_data = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        if report_data.get("status") != "completed":
            return None  # 只有 completed 才 seal

        # 读取 experiment-context.json（复用已有函数）
        context = read_context(str(workspace))
        analysis_config_id = context.get("analysis_config_id")
        if not analysis_config_id:
            logger.warning("ExperimentSealMiddleware: no analysis_config_id, skipping seal")
            return None  # 防止 "thread_id:unknown" 孤儿行

        # 读取 code-executor + data-analyst handoff（复用已有函数）
        # read_handoff 需要 thread_data；从 state 获取
        thread_data = state.get("thread_data", {})
        code_handoff = read_handoff(str(workspace), thread_data)
        analyst_handoff = _read_json_if_exists(workspace, "handoff_data_analyst.json")

        # 获取 user_id（注意：get_effective_user_id 在 contextvar 未设置时返回 DEFAULT_USER_ID）
        user_id = get_effective_user_id()
        if not user_id or user_id == "default":
            logger.warning("ExperimentSealMiddleware: no effective user_id, skipping seal")
            return None

        thread_id = runtime.context.get("thread_id")

        # 组装 ExperimentRow
        row_data = {
            "experiment_id": f"{thread_id}:{analysis_config_id}",
            "thread_id": thread_id,
            "user_id": user_id,
            "paradigm": context.get("paradigm"),
            "ev19_template": context.get("ev19_template"),
            "analysis_config_id": analysis_config_id,
            "metrics_summary": code_handoff.get("metrics_summary") if code_handoff else None,
            "key_findings": analyst_handoff.get("key_findings") if analyst_handoff else None,
            "quality_warnings": _aggregate_warnings(code_handoff, analyst_handoff),
            "status": _determine_status(report_data, code_handoff, analyst_handoff),
        }

        # 写入 DB
        sf = get_session_factory()
        if sf is None:
            # memory backend — 无持久化，跳过
            logger.debug("ExperimentSealMiddleware: no session_factory (memory backend), skipping")
            return None

        repo = ExperimentRepository(sf)
        await repo.upsert(row_data)

        return None


def _read_json_if_exists(workspace, filename):
    """读取 workspace 中的 JSON 文件，不存在或损坏返回 None。"""
    path = workspace / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _aggregate_warnings(code_handoff, analyst_handoff):
    """聚合 code-executor + data-analyst 的 quality_warnings。"""
    warnings = []
    if code_handoff:
        warnings.extend(code_handoff.get("data_quality_warnings", []))
    if analyst_handoff:
        warnings.extend(analyst_handoff.get("quality_warnings", []))
    return warnings if warnings else None


def _determine_status(report_handoff, code_handoff, analyst_handoff):
    """综合多个 handoff 推导 experiment status。"""
    # 任何 handoff 的 status == "failed" → failed
    for h in [report_handoff, code_handoff, analyst_handoff]:
        if h and h.get("status") == "failed":
            return "failed"
    # report-writer completed → completed
    if report_handoff.get("status") == "completed":
        return "completed"
    # 其他 → partial
    return "partial"
```

**Middleware 注入**：在 `_build_middlewares`（`agent.py:270`）中，紧挨 `TrainingDataMiddleware` 之后添加：

```python
# agent.py, _build_middlewares()
middlewares.append(ExperimentSealMiddleware())  # 紧挨 TrainingDataMiddleware 之后
```

注意：`make_lead_agent` 不暴露 `custom_middlewares` 参数——默认路径在 `agent.py:637` 直接调用 `_build_middlewares(config, model_name=model_name, agent_name=agent_name)`。因此 middleware 必须在 `_build_middlewares` 内部直接添加，不能通过外部参数注入。

#### 5.4b Feedback Router 侧（Trigger 2）

在 `routers/feedback.py` 的 `submit_feedback` handler 中，upsert feedback 之后调用 seal：

```python
# routers/feedback.py, submit_feedback()

# ... existing feedback upsert logic ...

# Trigger experiment seal（如果尚未 seal）
experiment_repo = get_experiment_repo(request)
await _seal_experiment_from_feedback(
    thread_id=thread_id,
    user_id=user_id,
    user_feedback=verdict,  # "accepted" | "revised" | "rejected"
    experiment_repo=experiment_repo,
)
```

`_seal_experiment_from_feedback` 的逻辑与 middleware 的 `_maybe_seal` 相同，但：
- 不需要等 report-writer handoff（用户反馈本身就是 seal 信号——即使分析 partial/failed，用户主动反馈意味着"这个 experiment 值得记录"）
- 设置 `user_feedback` 字段
- 如果 ExperimentRow 不存在（Trigger 1 从未触发，比如分析 partial 但用户仍给了反馈），新建一行

### 5.5 Repository

```python
# persistence/experiment/sql.py

class ExperimentRepository:
    """参考: persistence/feedback/sql.py FeedbackRepository"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def upsert(self, data: dict) -> ExperimentRow:
        """幂等 upsert。
        - 不存在 → INSERT
        - 已存在 → 只更新 updated_at 和 user_feedback（不覆盖分析数据）
        """

    async def list_by_user(
        self, user_id: str, paradigm: str | None = None,
        status: str | None = None, limit: int = 20
    ) -> list[ExperimentRow]:
        """参考 FeedbackRepository.list_prior_corrections 的过滤模式"""

    async def get_by_id(self, experiment_id: str, user_id: str) -> ExperimentRow | None:
        """单条查询，校验 user_id 所有权。参考 FeedbackRepository.get"""
```

### 5.6 Gateway 依赖注入

```python
# app/gateway/deps.py

# 在 langgraph_runtime() (line ~104) 中新增：
app.state.experiment_repo = ExperimentRepository(sf)

# 新增 getter（参考 get_feedback_repo, line ~162）：
def get_experiment_repo(request: Request) -> ExperimentRepository:
    return request.app.state.experiment_repo
```

### 5.7 API

```python
# app/gateway/routers/experiments.py

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

@router.get("/")
@require_permission("threads", "read")
async def list_experiments(
    request: Request,
    paradigm: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
) -> ExperimentListResponse:
    """列出当前用户的所有实验"""
    user_id = await get_current_user(request)
    repo = get_experiment_repo(request)
    rows = await repo.list_by_user(user_id, paradigm=paradigm, status=status, limit=limit)
    return ExperimentListResponse(experiments=[_to_item(r) for r in rows])


@router.get("/{experiment_id}")
@require_permission("threads", "read")
async def get_experiment(
    request: Request,
    experiment_id: str,
) -> ExperimentDetailResponse:
    """单个实验详情"""
    user_id = await get_current_user(request)
    repo = get_experiment_repo(request)
    row = await repo.get_by_id(experiment_id, user_id=user_id)
    if not row:
        raise HTTPException(status_code=404)
    return _to_detail(row)
```

**v0.1 MVP 只有 list + detail**。compare、export、notes 推迟。

### 5.8 Migration

```
# persistence/migrations/versions/20260606_experiments_table.py
revision = "20260606_experiments"
down_revision = "20260601_1500"  # 当前 head

def upgrade():
    op.create_table("experiments", ...)

def downgrade():
    op.drop_table("experiments")
```

**注意**：`make deploy-tar` 不跑 alembic。ECS 部署后需手动执行 `alembic upgrade head`。

### 5.9 前端 MVP

"Analysis History" 列表页 + 详情页（与上一版 spec 相同）。

### 5.10 测试策略

```
□ ExperimentRepository 单测:
  □ upsert 幂等（2nd seal 不覆盖 metrics_summary/key_findings）
  □ upsert 更新 user_feedback（不覆盖分析数据）
  □ list_by_user 过滤 paradigm/status
  □ get_by_id 校验 user_id 所有权（跨用户拒绝访问）

□ ExperimentSealMiddleware 单测:
  □ report-writer completed → seal 触发
  □ report-writer 不存在 → 跳过
  □ report-writer status=failed → 跳过
  □ analysis_config_id 缺失 → 跳过（防止 "thread_id:unknown" 孤儿行）
  □ get_session_factory() 返回 None（memory backend）→ 跳过，不抛异常
  □ middleware 抛异常 → swallow，agent turn 不受影响（red anchor）
  □ user_id 为 "default" → 跳过（防止 DEFAULT_USER_ID 泄漏）
  □ async upsert 实际 await 并 commit（不是 fire-and-forget coroutine）

□ 集成测试:
  □ 完整流水线（上传 EPM → 分析完成）后 GET /api/experiments 返回一条记录
  □ 提交 feedback 后 GET /api/experiments 返回带 user_feedback 的记录

□ 全量测试:
  ⚠ ExperimentRow 加入 persistence/models/__init__.py 改变 Base.metadata
  ⚠ _build_middlewares 加入新 middleware 改变链
  ⚠ deps.py 加入 app.state.experiment_repo
  → 必须跑全量测试（MEMORY feedback_pr_merge_must_run_full_suite_on_shared_logic）
  → grep 所有引用 Base.metadata 和 middleware 链的 fixture
```

---

## 实施检查清单

```
□ S1: LoopDetectionMiddleware 加入 executor.py:_build_middlewares（每次新实例）
□ S1: thread_id 隔离测试
□ S2: ethoinsight/validate.py + compute scripts 调用
□ S3: data-analyst SKILL.md fast-fail 段
□ S4: test_ethoinsight_write_protection.py (red anchor ×3)
□ S5.1: ExperimentRow model (persistence/experiment/model.py)
□ S5.2: persistence/models/__init__.py 注册 ExperimentRow
□ S5.3: ExperimentRepository (persistence/experiment/sql.py)
□ S5.4: ExperimentSealMiddleware (agents/middlewares/experiment_seal_middleware.py)
□ S5.5: Middleware 注入 (_build_middlewares, 紧挨 TrainingDataMiddleware 之后)
□ S5.6: deps.py — app.state.experiment_repo + get_experiment_repo
□ S5.7: Feedback router Trigger 2 (_seal_experiment_from_feedback)
□ S5.8: API router (app/gateway/routers/experiments.py) + gateway app 注册
□ S5.9: migration script (versions/20260606_experiments_table.py)
□ S5.10: 前端 Analysis History 列表页 + 详情页
□ S5.11: 测试（repo 单测 + middleware 单测 + 集成测试 + 全量测试）
```

---

## 附录：两轮 Opus Review 修正的关键问题

| 轮次 | 发现 | 修正 |
|------|------|------|
| 设计 review | `experiment-context.json` 已有 paradigm/fingerprint | ExperimentRow 改为投影而非新写入路径 |
| 设计 review | Lead 不读 handoff（relay 拓扑） | 取消 `record_experiment` 工具，改为 seal-time middleware |
| 设计 review | `max_turns` 在 `SubagentConfig` 不在 middleware | 确认已有，S1-S4 中移除 max_turns 工作项 |
| 设计 review | `write_file` 参数名是 `path` 不是 `file_path` | 修正，且 audit 确认已有两层防护 |
| Spec review | `ExperimentRepository()` 无参构造不可用 | 改为 `ExperimentRepository(session_factory)` |
| Spec review | Middleware 在 LangGraph 进程，无法访问 Gateway `app.state` | 两套 DB 访问路径：middleware 用 `get_session_factory()`，router 用 DI |
| Spec review | 3-trigger 模型 2/3 不可行（middleware 无法感知 HTTP 事件） | Trigger 2 移至 feedback router，Trigger 3 推迟 |
| Spec review | `groups` 列无数据源（experiment-context.json 无此字段） | 移除 `groups` 列 |
| Spec review | `custom_middlewares` slot 对默认 agent 是死代码 | 直接在 `_build_middlewares` 内部添加 |
| Spec review | S1 thread_id 共享导致 lead/subagent 计数污染 | 每次 subagent 运行新实例 `LoopDetectionMiddleware()` |
| Spec review | `get_effective_user_id()` 静默 fallback 到 DEFAULT_USER_ID | seal 前断言非 default |
| Spec review | `after_agent` 同步不能调用异步 `repo.upsert` | 改为 `aafter_agent` |
