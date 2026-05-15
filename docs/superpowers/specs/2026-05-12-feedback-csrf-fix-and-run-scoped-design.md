# 反馈按钮 CSRF 修复 + 接入上游 run-scoped 反馈架构 — 设计文档

**日期**：2026-05-12
**状态**：设计已审，待落 plan
**前置交接**：`docs/handoffs/2026-05/2026-05-12-feedback-button-csrf-and-upstream-sync-handoff.md`

## 1. 目标与范围

### 1.1 要解决的问题

聊天页面 assistant 消息下方的反馈按钮（✅ 正确 / ⚠️ 需修正 / ❌ 错误）点击后无任何反应。直接成因是前端 `submitFeedback()` 用裸 `fetch` 而未注入 `X-CSRF-Token`，被 `CSRFMiddleware` 静默拦截为 403，前端 `catch{}` 又把错误吞掉。

### 1.2 在审查过程中发现的更深问题

| # | 问题 | 严重性 |
|---|---|---|
| A | 前端 silently-ignore，任何 5xx/4xx 都让用户看到"按钮无反应" | 高 |
| B | nginx 在非标端口（2026）下 `Host $host` 丢端口号，CSRF cookie 域不匹配 | 高 |
| C | 当前 `routers/feedback.py` 无 `@require_permission`，**任何登录用户可向任意 thread 提交反馈**（auth bypass） | 严重 |
| D | Noldus 自定义 JSONL 反馈与上游已存在的 `FeedbackRepository` SQLite 设施平行，造成两套不联通的反馈存储 | 中 |
| E | 反馈不带 `run_id`，无法对应到具体的对话回合，未来 SFT 训练数据无法精确定位 | 高 |

### 1.3 范围

**包含**：CSRF 修复 + auth bypass 闭洞 + 反馈架构对齐上游（带 run_id、走 `FeedbackRepository`、扩展 model 装 Noldus verdict 语义）+ 拉取上游 nginx CSRF 单条修复。

**不包含**：
- 其他上游 sync（沙箱安全 6 条、bug fix 32 条、feat/refactor 12 条 — 单独 PR-B/C/D）
- 旧 JSONL 数据迁移（无线上数据可丢）
- 前端测试框架引入（保持零前端测试现状）
- 训练数据飞轮 SOP 全量重写（只更新反馈相关章节）

## 2. 项目状态修正

本设计基于以下确认的事实，**与 CLAUDE.md 中部分描述不一致**（CLAUDE.md 落后于代码实际状态）：

| CLAUDE.md 描述 | 实际状态（2026-05-12） |
|---|---|
| 「v0.1 单用户研究助手」 | 已落地多用户（`@require_permission` + `get_effective_user_id` + `UserRow` 等已挂在 threads 等多个 router） |
| 「Tier 4 体系故意不要」 | 2026-05-07/08 Tier234 round1-3 已合入，本仓库现状即建立在 `deerflow.persistence`、`runtime.user_context`、unified auth 之上 |
| 「反馈走 JSONL」 | 平行存在两套：Noldus 自定义 JSONL（前端用）+ 上游 `FeedbackRepository` SQLite（`thread_runs.py:298` 已在用 `list_by_thread_grouped`） |

实施时按本设计为准。完成后需同步更新 CLAUDE.md。

## 3. 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│ 前端 feedback-buttons.tsx                                          │
│   ✅ 正确 / ⚠️ 需修正 / ❌ 错误（+ revised_text 文本框）              │
│   Props 增加 runId                                                │
│       │                                                          │
│       │ csrfFetch (自动注入 X-CSRF-Token + credentials)            │
│       ▼                                                          │
│   submitFeedback(threadId, runId, body) → POST                   │
│                                                                  │
│   失败 → 按钮下方临时红字「提交失败，请重试」(3s) + console.error      │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼ HTTP through nginx (port 2026, Host $http_host)
┌──────────────────────────────────────────────────────────────────┐
│ Gateway · CSRFMiddleware ✓ AuthMiddleware ✓                       │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ Router: POST /api/threads/{tid}/runs/{rid}/feedback               │
│   @require_permission("threads", "write", owner_check=True,      │
│                       require_existing=True)                     │
│   1. 校验 rid 存在 + run.thread_id == tid                          │
│   2. 校验 verdict ∈ {correct, needs_fix, wrong}                   │
│   3. verdict→rating 映射：correct→+1, wrong→-1, needs_fix→null   │
│   4. feedback_repo.upsert(...) 落地 SQLite                        │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ FeedbackRow（扩展后）                                                │
│   feedback_id, run_id, thread_id, user_id, message_id            │
│   rating: int? (nullable, 兼容上游 aggregate_by_run)               │
│   comment: text? (Noldus 用作可选 note)                            │
│   verdict: str(16)? (correct/needs_fix/wrong) ← Noldus 新增        │
│   revised_text: text?                          ← Noldus 新增       │
│   created_at                                                     │
│   UniqueConstraint(thread_id, run_id, user_id, message_id)        │
│   ← 原 (tid,rid,uid)，扩展 message_id 才能支持同 run 多 message     │
└──────────────────────────────────────────────────────────────────┘
```

**核心思路**：
- 复用上游已挂在 `thread_runs.py` 上跑着的 `FeedbackRepository`
- 上游 model 加 2 个 nullable 字段 + 1 个 unique constraint 扩展
- 删除 Noldus 自定义 `feedback.py` 的 JSONL 落盘代码
- 拉 1 条 nginx 修复（`Host $http_host`）

## 4. 后端详细设计

### 4.1 ORM 扩展（`persistence/feedback/model.py`）

```python
class FeedbackRow(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint(
            "thread_id", "run_id", "user_id", "message_id",
            name="uq_feedback_thread_run_user_message",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    message_id: Mapped[str | None] = mapped_column(String(64))

    # 改 nullable：Noldus verdict 路径不强制写 rating
    rating: Mapped[int | None] = mapped_column(nullable=True)
    comment: Mapped[str | None] = mapped_column(Text)

    # Noldus 扩展
    verdict: Mapped[str | None] = mapped_column(String(16))      # correct / needs_fix / wrong
    revised_text: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC),
    )
```

变更总览：

| 字段 | 之前 | 之后 |
|---|---|---|
| `rating` | `int, NOT NULL` | `int, NULL` |
| `verdict` | — | `String(16), NULL` |
| `revised_text` | — | `Text, NULL` |
| UniqueConstraint | `(tid, rid, uid)` | `(tid, rid, uid, message_id)` |

**为什么 verdict 用 `String(16)` 不用 SQL Enum**：SQL Enum 在 SQLite 表现为 CHECK 约束，跨数据库 migration 处理麻烦。前端/后端用 Pydantic `Literal["correct","needs_fix","wrong"]` 做校验已足够。

### 4.2 Repository 扩展（`persistence/feedback/sql.py`）

`upsert` 加 `verdict`、`revised_text`、`message_id` 参数。`rating` 入参改 optional，向后兼容上游既有调用：

```python
async def upsert(
    self,
    *,
    run_id: str,
    thread_id: str,
    user_id: str | None | _AutoSentinel = AUTO,
    message_id: str | None = None,
    rating: int | None = None,           # 改 optional
    comment: str | None = None,
    verdict: str | None = None,           # 新增
    revised_text: str | None = None,      # 新增
) -> dict:
    if rating is not None and rating not in (1, -1):
        raise ValueError(f"rating must be +1 or -1, got {rating}")
    if verdict is not None and verdict not in {"correct", "needs_fix", "wrong"}:
        raise ValueError(f"verdict must be one of correct/needs_fix/wrong, got {verdict}")
    # 查询时 unique key 改为 (tid, rid, uid, message_id) 复合
    ...
```

`list_by_run` 返回字段中需附带 `verdict` 与 `revised_text`（`_row_to_dict` 自动随 ORM to_dict 输出）。

`list_by_thread_grouped` 签名保持不变（`thread_runs.py:300` 在用），它返回 `{run_id → record}` 仅取每 run 一条。**实施时必须 grep 该方法所有调用方**（已知至少 `thread_runs.py`，可能还有 client.py / 测试）**确认它们对返回 dict 字段集是宽容的**——如果有任何调用方对 dict keys 做 schema 校验，新增 `verdict` / `revised_text` 字段可能破坏。若发现严格 schema 校验，则改为**新增** `list_by_thread_grouped_v2` 而非扩展现有方法。

如未来需要 `(run_id, message_id)` 复合维度的聚合，新增 `list_by_thread_messages` 方法（**本次不实施**）。

### 4.3 Alembic migration

文件：`packages/agent/backend/packages/harness/deerflow/persistence/alembic/versions/<yyyymmdd_hhmm>_feedback_add_verdict_revised_text.py`

**SQLite 限制**：ALTER COLUMN 不被原生支持，必须用 `op.batch_alter_table`。

```python
def upgrade() -> None:
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.add_column(sa.Column("verdict", sa.String(16), nullable=True))
        batch.add_column(sa.Column("revised_text", sa.Text(), nullable=True))
        batch.alter_column("rating", existing_type=sa.Integer(), nullable=True)
        batch.drop_constraint("uq_feedback_thread_run_user", type_="unique")
        batch.create_unique_constraint(
            "uq_feedback_thread_run_user_message",
            ["thread_id", "run_id", "user_id", "message_id"],
        )

def downgrade() -> None:
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.drop_constraint("uq_feedback_thread_run_user_message", type_="unique")
        batch.create_unique_constraint(
            "uq_feedback_thread_run_user",
            ["thread_id", "run_id", "user_id"],
        )
        batch.alter_column("rating", existing_type=sa.Integer(), nullable=False)
        batch.drop_column("revised_text")
        batch.drop_column("verdict")
```

Downgrade 中 `rating` 改回 `NOT NULL` 在已有 verdict=needs_fix 写入（即 rating IS NULL）的环境下会失败。downgrade 仅用于：(a) migration 测试 M1 验证 upgrade 可逆性（测试在空表或新增空行上运行），(b) 紧急回滚——若回滚时 dev 环境已写过 needs_fix，需手动先跑 `UPDATE feedback SET rating=0 WHERE rating IS NULL` 清理再 downgrade。本次无线上数据，dev 环境清理可接受。

### 4.4 路由重写（`app/gateway/routers/feedback.py`）

**完全删除**现有 JSONL 实现，重写为：

```python
"""Feedback router — Noldus verdict-based feedback for SFT 训练数据飞轮.

Backed by FeedbackRepository (SQLite). URL aligned with upstream:
POST/GET /api/threads/{thread_id}/runs/{run_id}/feedback
"""
from __future__ import annotations
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_current_user, get_feedback_repo, get_run_store

router = APIRouter(prefix="/api/threads", tags=["feedback"])

VerdictT = Literal["correct", "needs_fix", "wrong"]
_VERDICT_TO_RATING: dict[str, int | None] = {
    "correct": 1, "wrong": -1, "needs_fix": None,
}


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    verdict: VerdictT
    revised_text: str | None = None
    note: str | None = None


class FeedbackItem(BaseModel):
    feedback_id: str
    thread_id: str
    run_id: str
    user_id: str | None
    message_id: str | None
    verdict: VerdictT | None
    revised_text: str | None
    note: str | None
    created_at: str


def _to_item(record: dict[str, Any]) -> FeedbackItem:
    return FeedbackItem(
        feedback_id=record["feedback_id"],
        thread_id=record["thread_id"],
        run_id=record["run_id"],
        user_id=record.get("user_id"),
        message_id=record.get("message_id"),
        verdict=record.get("verdict"),
        revised_text=record.get("revised_text"),
        note=record.get("comment"),
        created_at=record["created_at"],
    )


@router.post(
    "/{thread_id}/runs/{run_id}/feedback",
    response_model=FeedbackItem,
)
@require_permission("threads", "write", owner_check=True, require_existing=True)
async def submit_feedback(
    thread_id: str,
    run_id: str,
    body: FeedbackRequest,
    request: Request,
) -> dict[str, Any]:
    run_store = get_run_store(request)
    run = await run_store.get(run_id)
    if run is None or run.get("thread_id") != thread_id:
        raise HTTPException(404, f"Run {run_id} not found in thread {thread_id}")

    user_id = await get_current_user(request)
    feedback_repo = get_feedback_repo(request)
    record = await feedback_repo.upsert(
        thread_id=thread_id,
        run_id=run_id,
        user_id=user_id,
        message_id=body.message_id,
        verdict=body.verdict,
        rating=_VERDICT_TO_RATING[body.verdict],
        revised_text=body.revised_text,
        comment=body.note,
    )
    return _to_item(record).model_dump()


@router.get(
    "/{thread_id}/runs/{run_id}/feedback",
    response_model=list[FeedbackItem],
)
@require_permission("threads", "read", owner_check=True)
async def list_run_feedback(
    thread_id: str,
    run_id: str,
    request: Request,
) -> list[dict[str, Any]]:
    feedback_repo = get_feedback_repo(request)
    user_id = await get_current_user(request)
    rows = await feedback_repo.list_by_run(thread_id, run_id, user_id=user_id)
    return [_to_item(r).model_dump() for r in rows]
```

**关键点**：
- `@require_permission` 闭 auth bypass（问题 C）
- 校验 `run.thread_id == thread_id` 防跨 thread 注入
- `verdict→rating` 自动映射，使上游 `aggregate_by_run` thumbs-up/down 统计仍可用
- 前端 `note` 字段走上游 `comment` 列
- **删除**老版 `_base_dir()` 和 JSONL `with path.open("a") as f` 等代码

`app/gateway/app.py:367` 注释更新为：`# Feedback API mounted at /api/threads/{thread_id}/runs/{run_id}/feedback`

### 4.5 训练数据导出脚本（顺手，可选）

新增 `packages/agent/backend/scripts/export_feedback_jsonl.py`：

```python
"""Export SQLite feedback to Fireworks-style JSONL for SFT 训练流水线."""
# SELECT * FROM feedback ORDER BY created_at;
# 输出 .deer-flow/training-data/feedback-export.jsonl
# 每行 {thread_id, run_id, message_id, verdict, revised_text, note, user_id, created_at}
```

不挂 cron、不进 middleware——离线工具，训练时手动跑一次。若实施时间紧可推迟到下一 PR。

## 5. 前端详细设计

### 5.1 message → run_id 透传链路

LangGraph SDK 的 `Message` 类型本身不带 `run_id`，但 stream event metadata 里有。两种方案：

**方案 metadata（推荐）**：把 `run_id` 落到每条 message 的 `response_metadata` 上，FeedbackButtons 直接读。
**方案 parallel Map（备选）**：thread state 加 `messageRunIds: Map<string,string>`，逐层传递。

实施 plan 的 Step 4 必须以**侦察任务**开始，按以下顺序验证后再写代码：

1. 在 `packages/agent/frontend/src/core/threads/hooks.ts` 中定位 `useThreadStream` 或等价 stream 处理函数，找到处理每条 message 事件的回调（通常是 `onMessage`、`onValues` 或类似命名）
2. 检查 stream event 里 `run_id` 字段的存在位置（可能在 `event.metadata.run_id`、`event.run_id` 或外层闭包）
3. 验证 `Message` 类型（来自 `@langchain/langgraph-sdk`）是否允许任意字段写入 `response_metadata`——如果该类型严格，需另寻字段或退方案 parallel Map
4. 检查现有代码是否已经把 `run_id` 落到了 message 的某个位置（grep `run_id` / `runId`）

侦察完成后：
- 若方案 metadata 可行：在 message 事件回调里 `message.response_metadata = { ...message.response_metadata, run_id: <来源> }`
- 若不可行：在 `useThreadStream` 返回值新增 `messageRunIds: Map<string,string>`，逐层透传到 `MessageList` → `MessageListItem` → `FeedbackButtons`

**Step 4 的子任务在 plan 中拆为独立任务，第一个子任务即"侦察并选定方案"，再后续才是实施。**

### 5.2 `submitFeedback` 签名与错误处理

```typescript
// api-client.ts
import { fetch as csrfFetch } from "./fetcher";

export async function submitFeedback(
  threadId: string,
  runId: string,
  body: FeedbackRequest,
): Promise<FeedbackItem> {
  const res = await csrfFetch(
    `/api/threads/${threadId}/runs/${runId}/feedback`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) throw new Error(`submitFeedback failed: ${res.status}`);
  return res.json() as Promise<FeedbackItem>;
}

export async function listFeedback(
  threadId: string,
  runId: string,
): Promise<{ items: FeedbackItem[] }> {
  const res = await csrfFetch(
    `/api/threads/${threadId}/runs/${runId}/feedback`,
  );
  if (!res.ok) throw new Error(`listFeedback failed: ${res.status}`);
  const items = (await res.json()) as FeedbackItem[];
  return { items };
}
```

### 5.3 FeedbackButtons 签名与错误提示

```typescript
interface Props {
  threadId: string;
  runId: string;          // 新增，必传
  messageId: string;
  existingVerdict?: FeedbackVerdict;
  className?: string;
}

const [error, setError] = useState<string | null>(null);

const handleCorrect = async () => {
  setSubmitting(true);
  setError(null);
  try {
    await submitFeedback(threadId, runId, { message_id: messageId, verdict: "correct" });
    setVerdict("correct");
  } catch (e) {
    console.error("Feedback submission failed:", e);
    setError("提交失败，请重试");
    setTimeout(() => setError(null), 3000);
  } finally {
    setSubmitting(false);
  }
};
```

UI 渲染时若 `error`，在按钮组下方显示一行 `text-xs text-destructive`。无 toast、无 sonner，零新依赖。

**`handleSubmitRevision`** 同样处理。

### 5.4 消费层更新

| 文件 | 改动 |
|---|---|
| `message-list-item.tsx:60` | `<FeedbackButtons runId={message.response_metadata?.run_id ?? ...} ...>`，**runId 缺失时不渲染整个组件** |
| `subtask-card.tsx:172` | 同上 |

## 6. 基础设施

### 6.1 cherry-pick 上游 nginx 修复

`70737af7 fix(nignx): resolve CSRF auth failure on non-standard ports (#2796)`：

```bash
git cherry-pick 70737af7
```

冲突可能性极低（diff 仅 `Host $host` → `$http_host`，多处重复）。若冲突，手动改 `packages/agent/docker/nginx/nginx.conf` 所有 `proxy_set_header Host $host;` → `proxy_set_header Host $http_host;`。

### 6.2 与本次拉取无关的其他上游 commits

参见同日 handoff 中"上游 sync 分类清单"——分 PR-B（剩 6 条安全）/ PR-C（32 条 bug fix）/ PR-D（12 条 feat）单独立项。**本次仅拉 nginx 一条**。

## 7. 测试策略

### 7.1 后端测试（pytest）

文件：`packages/agent/backend/tests/test_feedback_api.py`（新建）

#### 7.1.1 ORM / Repository 层（6 条）

| # | 名称 | 验证 |
|---|---|---|
| U1 | `test_upsert_creates_with_verdict` | verdict="correct" 落地，rating=1，revised_text 持久化 |
| U2 | `test_upsert_with_needs_fix_rating_null` | verdict="needs_fix" 时 rating=NULL，aggregate_by_run 不计入 |
| U3 | `test_upsert_replaces_existing` | 同 (tid,rid,uid,mid) 二次 upsert 覆盖 |
| U4 | `test_upsert_different_message_separate_rows` | 同 (tid,rid,uid) 不同 message_id 产生两行 |
| U5 | `test_upsert_invalid_verdict_raises` | verdict 非三选一抛 ValueError |
| U6 | `test_list_by_run_returns_verdict_fields` | list 返回包含 verdict、revised_text |

#### 7.1.2 路由层（FastAPI TestClient，8 条）

| # | 名称 | 验证 |
|---|---|---|
| R1 | `test_post_without_csrf_returns_403` | POST 不带 X-CSRF-Token → 403 |
| R2 | `test_post_with_csrf_returns_200_and_persists` | 200 且 SELECT 验证字段 |
| R3 | `test_post_unauthorized_user_returns_403_or_404` | 用户 A 提交到用户 B 的 thread → 拦截 |
| R4 | `test_post_run_not_in_thread_returns_404` | run_id 存在但属于另一 thread |
| R5 | `test_post_nonexistent_run_returns_404` | run_id 不存在 |
| R6 | `test_post_invalid_verdict_returns_422` | verdict 非三选一 → Pydantic 422 |
| R7 | `test_get_returns_feedback_list` | GET 返回当前用户反馈，他人反馈不出现 |
| R8 | `test_post_two_messages_in_same_run` | 同 run 不同 message 各一条 → DB 两行 |

#### 7.1.3 兼容性回归（2 条）

| # | 名称 | 验证 |
|---|---|---|
| C1 | `test_thread_runs_messages_endpoint_still_attaches_feedback` | `GET /api/threads/{tid}/messages` 仍能拿到 feedback |
| C2 | `test_upstream_rating_only_path_still_works` | repo.upsert(rating=+1, 不传 verdict) 仍工作 |

#### 7.1.4 Migration 测试（1 条）

| # | 名称 | 验证 |
|---|---|---|
| M1 | `test_alembic_upgrade_then_downgrade` | upgrade head 后含新列 + 新 unique；downgrade -1 后回归 |

### 7.2 前端手动 QA 清单

PR 评审时跑：

1. `make dev` 起服务，登录 `localhost:2026/workspace/chats/<thread-id>`
2. 等 assistant 出消息，点 ✅ → 按钮变 "已反馈"
3. F12 Network 看 `POST /api/threads/{tid}/runs/{rid}/feedback` 200，请求头含 `X-CSRF-Token` + Cookie
4. 点 ⚠️ → 出现修订文本框 → 写一段 → 提交 → `sqlite3 backend/.deer-flow/db.sqlite "SELECT verdict, revised_text FROM feedback;"` 字段非空
5. **故障注入**：浏览器 devtools 删 csrf cookie，再点 ✅ → 按钮下方红字「提交失败，请重试」3 秒后消失
6. **故障注入**：后端 feedback router 临时改 raise 500，重试 → 同样红字
7. 同一 message 二次反馈 → DB 行 upsert 不增行

### 7.3 nginx 修复手动验证（1 条）

1. `git cherry-pick 70737af7` 后看 diff 仅 `Host $host` → `$http_host`
2. `make dev` 启动，`curl -v http://localhost:2026/api/...` 看 Gateway 收到 Host header 含端口
3. 跑一遍 §7.2 QA 清单

## 8. 改动文件总览

### 8.1 后端

| 文件 | 类型 | 说明 |
|---|---|---|
| `packages/harness/deerflow/persistence/feedback/model.py` | 修改 | 加 verdict / revised_text；rating→nullable；unique constraint 扩展 message_id |
| `packages/harness/deerflow/persistence/feedback/sql.py` | 修改 | upsert 新参数；rating optional；verdict 校验 |
| `packages/harness/deerflow/persistence/alembic/versions/<new>.py` | 新建 | batch_alter_table migration |
| `app/gateway/routers/feedback.py` | 重写 | 删 JSONL；改走 FeedbackRepository；URL 加 run_id；加 require_permission |
| `app/gateway/app.py` | 修改 | 路由挂载注释更新 |
| `tests/test_feedback_api.py` | 新建 | §7.1 全部 17 条测试 |
| `tests/conftest.py` | 可能微调 | 需要新 fixture 时 |
| `scripts/export_feedback_jsonl.py` | 新建（可选） | 离线导 SQLite → JSONL |

### 8.2 前端

| 文件 | 类型 | 说明 |
|---|---|---|
| `src/core/api/api-client.ts` | 修改 | submitFeedback / listFeedback 签名加 runId；URL 加 /runs/{rid}；改 csrfFetch |
| `src/core/threads/hooks.ts`（或 stream 事件处理处） | 修改 | stream event run_id 写入 message metadata |
| `src/components/feedback/feedback-buttons.tsx` | 修改 | 加 runId 入参；error state；3s 红字提示 |
| `src/components/workspace/messages/message-list-item.tsx` | 修改 | 取 run_id 传 FeedbackButtons；缺则不渲染 |
| `src/components/workspace/messages/subtask-card.tsx` | 修改 | 同上 |

### 8.3 基础设施

| 文件 | 类型 | 说明 |
|---|---|---|
| `packages/agent/docker/nginx/nginx.conf` | 修改 | cherry-pick 70737af7：`Host $host` → `$http_host` |

### 8.4 文档（CLAUDE.md 强制）

| 文件 | 改动 |
|---|---|
| `packages/agent/backend/CLAUDE.md` | feedback router 段说明 SQLite + run_id 新路径 |
| `CLAUDE.md`（项目根） | 第 7 条训练数据飞轮更新为「反馈走 SQLite，路由 `/api/threads/{tid}/runs/{rid}/feedback`」；第 11 条之后追加多用户 / Tier 4 已合状态的修正 |
| `docs/sop/training-data-flywheel-sop.md` | "反馈从 JSONL 读" → "反馈从 SQLite 读 + `scripts/export_feedback_jsonl.py` 导出脚本" |
| `docs/handoffs/2026-05/2026-05-12-feedback-csrf-fix-completion-handoff.md` | 新建完成 handoff |

合计 ~16 个文件改动 + ~4 个新建。

## 9. 实施顺序

| Step | 内容 | 验证 |
|---|---|---|
| 1 | ORM model + migration + Repository 扩展 | U1-U6 全绿 |
| 2 | 重写 feedback router | R1-R8 + C1-C2 全绿 |
| 3 | cherry-pick 70737af7 nginx | `make dev` 启动 + curl 验 Host header |
| 4 | 前端 run_id 透传 + csrfFetch + error state + FeedbackButtons 签名 | 编译通过 + `pnpm check` 全绿 |
| 5 | 集成验证：`make dev` → §7.2 手动 QA 清单 | 7 项全通 |
| 6 | 文档：CLAUDE.md / SOP / handoff | 同步完成 |
| 7 | commit（中文 message，按 CLAUDE.md 规范） | — |

每 step 独立可 `git revert`。前 3 步与前端解耦，可并行/分批 commit。

## 10. 风险与回滚

| 风险 | 缓解 |
|---|---|
| SQLite ALTER COLUMN 限制 | `batch_alter_table`；M1 兜底 |
| `list_by_thread_grouped` 改造影响 `/api/threads/{tid}/messages` | C1 兜底；保持签名不变 |
| 前端 message metadata.run_id 拿不到（SDK 限制） | Step 4 第一步验证；备选 parallel Map |
| nginx `Host $http_host` 行为变化 | 上游已验证；§7.3 手动 QA 兜底 |
| auth bypass 修复后旧客户端 401/403 | 项目未上线，无外部客户端，硬切可接受 |
| `upsert` rating=optional 破坏 thread_runs.py 已有调用 | C2 兜底；rating 默认 None 时不传也兼容 |

回滚：每 step 独立 commit，`git revert`。Migration 有 downgrade（注意 §4.3 中 rating NULL 行的清理需求）。

## 11. 不在本设计范围

明确**不**做：

- ❌ 拉 PR-B 安全批 6 条（沙箱安全修复，单独 PR）
- ❌ 拉 PR-C bug fix 32 条（单独 PR）
- ❌ 拉 PR-D feat / refactor 12 条（单独 PR，可能 v0.1 后）
- ❌ 训练数据飞轮 SOP 全量重写（只更新反馈段）
- ❌ 前端引入 vitest / 测试框架
- ❌ 旧 JSONL → SQLite 数据迁移（无线上数据可丢）
- ❌ 新增 `list_by_thread_messages` 复合维度聚合接口（后续需要时再加）

## 12. 后续可优化方向

- 把 `verdict` SQL 校验从应用层下沉到 SQL CHECK 约束（跨 SQLite/Postgres 兼容性较繁，留待 Postgres 化时一并处理）
- `aggregate_by_run` 扩展返回 `needs_fix` 计数（目前只返回 positive/negative）
- 训练数据导出脚本接入 cron / make target
- 把 PR-B 安全批拉进来（与本次拉的 nginx 同源逻辑）
