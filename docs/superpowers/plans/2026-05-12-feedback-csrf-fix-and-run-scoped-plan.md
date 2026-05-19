# 反馈按钮 CSRF 修复 + 接入上游 run-scoped 反馈架构 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复反馈按钮无反应的 CSRF + auth bypass + nginx 三重叠加 bug，并把 Noldus 自定义 JSONL 反馈架构对齐到上游已存在的 `FeedbackRepository`（SQLite + run_id 路由形状），扩展上游 ORM model 装下 Noldus 的 verdict 三分类与 revised_text。

**Architecture:** 复用上游 `FeedbackRepository`（`thread_runs.py` 已在用），扩展 model 加 `verdict` / `revised_text` nullable 字段，UniqueConstraint 扩 `message_id`。后端路由切到 `/api/threads/{tid}/runs/{rid}/feedback` 形状并加 `@require_permission`。前端 `submitFeedback` 用 `csrfFetch` 并透传 runId，失败时 3s 红字提示。同时 cherry-pick 上游 `70737af7` nginx `Host $http_host` 修复。

**Tech Stack:** Python 3.12 / FastAPI / SQLAlchemy 2.0 async / Alembic / pytest / Next.js 16 / React 19 / TypeScript / LangGraph SDK / pnpm.

**Spec:** [docs/superpowers/specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md](../specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md)

**项目根目录**：所有相对路径相对于 `/home/wangqiuyang/noldus-insight/`。

**关键事实（写 plan 前已验证）**：
- 上游 `FeedbackRepository`、`FeedbackRow`、`get_feedback_repo`、`get_run_store` 已存在
- `feedback` 表通过 `Base.metadata.create_all()` 在 engine 启动时建（**当前不跑 Alembic**），但 alembic scaffold 已就绪（`render_as_batch=True` 已开）
- dev 环境 `.deer-flow/data/deerflow.db` 已有 feedback 表，**0 行数据**，可 drop 重建
- `list_by_thread_grouped` 只被 `thread_runs.py:300` 调用，consumer 只读 `feedback_id/rating/comment`，新增 verdict 字段不破坏
- 有效 permissions：`threads:read/write/delete`、`runs:create/read/cancel`。**没有 `runs:write`**——feedback POST 用 `threads:write`（与 uploads 同范式）
- 前端 `useStream` 由 LangGraph SDK 管理 message 状态，**目前完全不存 run_id**

---

## File Structure

### 后端

- **Modify** `packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py`
  加 `verdict` (`String(16)` nullable) + `revised_text` (`Text` nullable) 字段；`rating` 改 nullable；UniqueConstraint 扩 `message_id`
- **Modify** `packages/agent/backend/packages/harness/deerflow/persistence/feedback/sql.py`
  `upsert` 加 `message_id`/`verdict`/`revised_text` 入参；`rating` optional；查询 key 改复合
- **Create** `packages/agent/backend/packages/harness/deerflow/persistence/migrations/versions/20260512_1200_feedback_verdict_revised.py`
  Alembic 第一条 migration，下游 schema 用
- **Rewrite** `packages/agent/backend/app/gateway/routers/feedback.py`
  删 JSONL；改走 FeedbackRepository；URL 加 run_id；加 require_permission
- **Modify** `packages/agent/backend/app/gateway/app.py:367-368`
  更新挂载注释
- **Create** `packages/agent/backend/tests/test_feedback_api.py`
  17 条测试覆盖 ORM/路由/兼容/migration

### 前端

- **Modify** `packages/agent/frontend/src/core/api/api-client.ts:91-110`
  `submitFeedback`/`listFeedback` 签名加 `runId`；URL 改 run-scoped；改 `csrfFetch`
- **Modify** `packages/agent/frontend/src/core/threads/hooks.ts:203-251`
  在 `onLangChainEvent` 里捕获 message run_id，写入 `messageRunIds` Map 并 expose 给消费者
- **Modify** `packages/agent/frontend/src/core/threads/state.ts` 或等价 thread state 暴露处
  补 `messageRunIds` 字段
- **Modify** `packages/agent/frontend/src/components/feedback/feedback-buttons.tsx`
  加 `runId` 入参；error state；3s 红字提示
- **Modify** `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx:60-65`
  从 thread state 取 run_id 传给 FeedbackButtons；缺则不渲染
- **Modify** `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx:172`
  同上

### 基础设施

- **Modify** `packages/agent/docker/nginx/nginx.conf`
  cherry-pick `70737af7`：所有 `proxy_set_header Host $host;` → `proxy_set_header Host $http_host;`

### 文档

- **Modify** `packages/agent/backend/CLAUDE.md` Feedback router 段
- **Modify** `CLAUDE.md`（项目根）第 7 条 + 项目状态修正
- **Modify** `docs/sop/training-data-flywheel-sop.md` 反馈来源段
- **Create** `docs/handoffs/2026-05/2026-05-12-feedback-csrf-fix-completion-handoff.md`

---

## Sequencing

```
Phase 1: ORM 扩展（Task 1-3）
   ↓
Phase 2: 重写路由（Task 4-6）
   ↓
Phase 3: nginx cherry-pick（Task 7）
   ↓
Phase 4: 前端 run_id 透传（Task 8-9）
   ↓
Phase 5: 前端 FeedbackButtons 改造（Task 10-11）
   ↓
Phase 6: 集成验证（Task 12）
   ↓
Phase 7: 文档（Task 13-14）
```

每个 task 独立 commit；Phase 1-3 与 Phase 4-5 完全解耦。

---

## Phase 1：ORM 扩展

### Task 1: 扩展 FeedbackRow model + 重置 dev DB

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py`

- [ ] **Step 1: 备份 dev DB（防回滚需要）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
cp .deer-flow/data/deerflow.db .deer-flow/data/deerflow.db.bak-pre-feedback-verdict
```

Expected: 文件复制成功。

- [ ] **Step 2: 修改 model.py**

打开 `packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py`，整文件改为：

```python
"""ORM model for user feedback on runs."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class FeedbackRow(Base):
    __tablename__ = "feedback"

    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "run_id",
            "user_id",
            "message_id",
            name="uq_feedback_thread_run_user_message",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    message_id: Mapped[str | None] = mapped_column(String(64))
    # message_id is an optional RunEventStore event identifier —
    # allows feedback to target a specific message or the entire run.
    # Noldus extension: 进入 unique constraint，使同 run 多 message 反馈互不覆盖。

    # Noldus 业务路径不写 rating；上游 thumbs-up/down 路径仍写。
    rating: Mapped[int | None] = mapped_column(nullable=True)

    comment: Mapped[str | None] = mapped_column(Text)

    # Noldus 扩展：三分类 verdict + 专家修订版本（SFT 训练种子）
    verdict: Mapped[str | None] = mapped_column(String(16))
    revised_text: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
```

- [ ] **Step 3: 删除 dev DB feedback 表（因 0 行，让 create_all 用新 schema 重建）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
uv run python -c "
import sqlite3
con = sqlite3.connect('.deer-flow/data/deerflow.db')
con.execute('DROP TABLE IF EXISTS feedback')
con.commit()
print('feedback table dropped')
"
```

Expected: `feedback table dropped`

- [ ] **Step 4: 启动 backend 一次让 create_all 用新 schema 建表**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
timeout 15 make gateway 2>&1 | tail -20 || true
```

Expected: gateway 启动过程中 SQLAlchemy 静默创建表，无错。15s 后被 timeout 杀掉，正常。

- [ ] **Step 5: 验证新 schema 落地**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
uv run python -c "
import sqlite3
con = sqlite3.connect('.deer-flow/data/deerflow.db')
print(con.execute(\"SELECT sql FROM sqlite_master WHERE name='feedback'\").fetchone()[0])
"
```

Expected: 输出包含 `verdict VARCHAR(16)`、`revised_text TEXT`、`rating INTEGER`（无 NOT NULL）、`UNIQUE (thread_id, run_id, user_id, message_id)`。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py
git commit -m "$(cat <<'EOF'
feat(feedback): 扩展 FeedbackRow 加 verdict + revised_text + message_id 复合 unique

Noldus 业务的三分类 verdict 与专家修订版本（SFT 训练种子）落入上游 model。rating
改 nullable 以支持 verdict=needs_fix 不写 rating。UniqueConstraint 加 message_id
让同 run 多 message 反馈互不覆盖。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 扩展 FeedbackRepository

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/feedback/sql.py`
- Test: `packages/agent/backend/tests/test_feedback_api.py` (新建，仅写 ORM 层测试)

- [ ] **Step 1: 写 ORM 层失败测试**

新建 `packages/agent/backend/tests/test_feedback_api.py`：

```python
"""Tests for Noldus run-scoped feedback (verdict + revised_text)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from deerflow.persistence.base import Base
from deerflow.persistence.feedback import FeedbackRepository


@pytest_asyncio.fixture
async def repo() -> FeedbackRepository:
    """In-memory SQLite repo for isolated repo-layer tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    yield FeedbackRepository(sf)
    await engine.dispose()


@pytest.mark.asyncio
async def test_upsert_creates_with_verdict(repo: FeedbackRepository):
    """U1: verdict=correct 落地，rating=1，revised_text 持久化。"""
    row = await repo.upsert(
        thread_id="t1",
        run_id="r1",
        user_id="u1",
        message_id="m1",
        verdict="correct",
        rating=1,
        revised_text="this is fine",
    )
    assert row["verdict"] == "correct"
    assert row["rating"] == 1
    assert row["revised_text"] == "this is fine"
    assert row["thread_id"] == "t1"
    assert row["run_id"] == "r1"
    assert row["message_id"] == "m1"


@pytest.mark.asyncio
async def test_upsert_with_needs_fix_rating_null(repo: FeedbackRepository):
    """U2: verdict=needs_fix 时 rating 可空。"""
    row = await repo.upsert(
        thread_id="t1",
        run_id="r1",
        user_id="u1",
        message_id="m1",
        verdict="needs_fix",
        rating=None,
        revised_text="should say X instead",
    )
    assert row["verdict"] == "needs_fix"
    assert row["rating"] is None
    assert row["revised_text"] == "should say X instead"


@pytest.mark.asyncio
async def test_upsert_replaces_existing(repo: FeedbackRepository):
    """U3: 同 (tid, rid, uid, mid) 二次 upsert 覆盖。"""
    await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1", message_id="m1",
        verdict="correct", rating=1,
    )
    row = await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1", message_id="m1",
        verdict="wrong", rating=-1, revised_text="actually wrong",
    )
    assert row["verdict"] == "wrong"
    assert row["rating"] == -1
    assert row["revised_text"] == "actually wrong"
    # 同 message_id 只一行
    rows = await repo.list_by_run("t1", "r1", user_id="u1")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_upsert_different_message_separate_rows(repo: FeedbackRepository):
    """U4: 同 (tid, rid, uid) 不同 message_id 产生两行。"""
    await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1", message_id="m1",
        verdict="correct", rating=1,
    )
    await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1", message_id="m2",
        verdict="wrong", rating=-1,
    )
    rows = await repo.list_by_run("t1", "r1", user_id="u1")
    assert len(rows) == 2
    msgs = {r["message_id"] for r in rows}
    assert msgs == {"m1", "m2"}


@pytest.mark.asyncio
async def test_upsert_invalid_verdict_raises(repo: FeedbackRepository):
    """U5: verdict 非三选一抛 ValueError。"""
    with pytest.raises(ValueError, match="verdict"):
        await repo.upsert(
            thread_id="t1", run_id="r1", user_id="u1", message_id="m1",
            verdict="invalid_verdict", rating=1,
        )


@pytest.mark.asyncio
async def test_list_by_run_returns_verdict_fields(repo: FeedbackRepository):
    """U6: list 返回 dict 包含 verdict、revised_text。"""
    await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1", message_id="m1",
        verdict="correct", rating=1, revised_text="ok",
    )
    rows = await repo.list_by_run("t1", "r1", user_id="u1")
    assert len(rows) == 1
    assert "verdict" in rows[0]
    assert "revised_text" in rows[0]
    assert rows[0]["verdict"] == "correct"
    assert rows[0]["revised_text"] == "ok"


@pytest.mark.asyncio
async def test_upstream_rating_only_path_still_works(repo: FeedbackRepository):
    """C2: 上游路径 rating=+1 不传 verdict 依然写入。"""
    row = await repo.upsert(
        thread_id="t1", run_id="r1", user_id="u1",
        rating=1, comment="thumbs up",
    )
    assert row["rating"] == 1
    assert row["comment"] == "thumbs up"
    assert row["verdict"] is None
    assert row["revised_text"] is None
```

- [ ] **Step 2: 运行测试看它们失败**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_feedback_api.py -v 2>&1 | tail -30
```

Expected: U1-U6 + C2 全 FAIL，错误来自 `upsert` 不接受 `verdict`/`revised_text`/`message_id` 入参，或来自 unique constraint 行为差异。

- [ ] **Step 3: 改 sql.py 的 upsert 方法**

打开 `packages/agent/backend/packages/harness/deerflow/persistence/feedback/sql.py`，把 `upsert` 方法（约 line 125-163）整体替换为：

```python
    async def upsert(
        self,
        *,
        run_id: str,
        thread_id: str,
        user_id: str | None | _AutoSentinel = AUTO,
        message_id: str | None = None,
        rating: int | None = None,
        comment: str | None = None,
        verdict: str | None = None,
        revised_text: str | None = None,
    ) -> dict:
        """Create or update feedback for (thread_id, run_id, user_id, message_id).

        rating must be +1, -1, or None (Noldus verdict=needs_fix path).
        verdict must be 'correct' / 'needs_fix' / 'wrong' or None (上游 rating-only path).
        """
        if rating is not None and rating not in (1, -1):
            raise ValueError(f"rating must be +1 or -1, got {rating}")
        if verdict is not None and verdict not in {"correct", "needs_fix", "wrong"}:
            raise ValueError(
                f"verdict must be one of correct/needs_fix/wrong, got {verdict}"
            )
        resolved_user_id = resolve_user_id(user_id, method_name="FeedbackRepository.upsert")
        async with self._sf() as session:
            stmt = select(FeedbackRow).where(
                FeedbackRow.thread_id == thread_id,
                FeedbackRow.run_id == run_id,
                FeedbackRow.user_id == resolved_user_id,
                FeedbackRow.message_id == message_id,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is not None:
                row.rating = rating
                row.comment = comment
                row.verdict = verdict
                row.revised_text = revised_text
                row.created_at = datetime.now(UTC)
            else:
                row = FeedbackRow(
                    feedback_id=str(uuid.uuid4()),
                    run_id=run_id,
                    thread_id=thread_id,
                    user_id=resolved_user_id,
                    message_id=message_id,
                    rating=rating,
                    comment=comment,
                    verdict=verdict,
                    revised_text=revised_text,
                    created_at=datetime.now(UTC),
                )
                session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)
```

`create` 方法**保留原样**（仅上游 rating-only 路径用，rating 必填）。

- [ ] **Step 4: 运行测试看它们通过**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_feedback_api.py -v 2>&1 | tail -20
```

Expected: 7 个测试全 PASS。

- [ ] **Step 5: 跑全量测试确认无回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test 2>&1 | tail -20
```

Expected: 所有原有测试继续通过，新增 7 条 PASS。如果有失败，必修。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/persistence/feedback/sql.py packages/agent/backend/tests/test_feedback_api.py
git commit -m "$(cat <<'EOF'
feat(feedback): Repository.upsert 支持 verdict / revised_text / message_id 复合 key

upsert 加 message_id/verdict/revised_text 入参，rating 改 optional 以支持
verdict=needs_fix 路径。上游 rating-only 调用保持向后兼容。

ORM 层覆盖 7 个测试（U1-U6、C2）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 加 Alembic migration（生产部署用）

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/persistence/migrations/versions/20260512_1200_feedback_verdict_revised.py`

**说明**：当前生产部署用 `Base.metadata.create_all()` 建表，Alembic 仅 scaffold。这条 migration 是**为未来用 Alembic 接管 schema 做准备**——本次实施完成后该文件即可直接被 `alembic upgrade head` 使用。

- [ ] **Step 1: 写 migration 文件**

新建 `packages/agent/backend/packages/harness/deerflow/persistence/migrations/versions/20260512_1200_feedback_verdict_revised.py`：

```python
"""feedback verdict + revised_text + message_id unique

Revision ID: 20260512_1200
Revises:
Create Date: 2026-05-12 12:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260512_1200"
down_revision = None  # first migration in this scaffold
branch_labels = None
depends_on = None


def upgrade() -> None:
    """加 verdict / revised_text 字段；rating 改 nullable；unique 加 message_id。"""
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.add_column(sa.Column("verdict", sa.String(16), nullable=True))
        batch.add_column(sa.Column("revised_text", sa.Text(), nullable=True))
        batch.alter_column(
            "rating", existing_type=sa.Integer(), nullable=True
        )
        batch.drop_constraint("uq_feedback_thread_run_user", type_="unique")
        batch.create_unique_constraint(
            "uq_feedback_thread_run_user_message",
            ["thread_id", "run_id", "user_id", "message_id"],
        )


def downgrade() -> None:
    """回滚。若已有 rating IS NULL 行（verdict=needs_fix 写入），需先 UPDATE 清理。

    SQL: UPDATE feedback SET rating=0 WHERE rating IS NULL;
    本次无线上数据，dev 环境清理可接受。
    """
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.drop_constraint("uq_feedback_thread_run_user_message", type_="unique")
        batch.create_unique_constraint(
            "uq_feedback_thread_run_user",
            ["thread_id", "run_id", "user_id"],
        )
        batch.alter_column(
            "rating", existing_type=sa.Integer(), nullable=False
        )
        batch.drop_column("revised_text")
        batch.drop_column("verdict")
```

- [ ] **Step 2: 验证 Alembic 可识别 migration（autogenerate 检查）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. DEER_FLOW_DB_URL=sqlite+aiosqlite:///./data/deerflow.db \
  uv run alembic -c packages/harness/deerflow/persistence/migrations/alembic.ini history 2>&1 | head -10
```

Expected: 输出包含 `20260512_1200`，无解析错误。

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/persistence/migrations/versions/20260512_1200_feedback_verdict_revised.py
git commit -m "$(cat <<'EOF'
feat(feedback): Alembic migration 加 verdict / revised_text / message_id unique

为未来 Alembic 接管 schema 做准备。当前生产用 create_all 建表，本 migration 仅
作 scaffold 与未来部署路径。

downgrade 路径要求清理 rating IS NULL 行（见 docstring）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2：重写路由

### Task 4: 重写 feedback router 为 run-scoped + require_permission

**Files:**
- Rewrite: `packages/agent/backend/app/gateway/routers/feedback.py`

- [ ] **Step 1: 整文件替换**

把 `packages/agent/backend/app/gateway/routers/feedback.py` 整文件替换为：

```python
"""Feedback router — Noldus verdict-based feedback for SFT 训练数据飞轮.

Backed by FeedbackRepository (SQLite). URL aligned with upstream:
POST/GET /api/threads/{thread_id}/runs/{run_id}/feedback

Noldus 语义：verdict ∈ {correct, needs_fix, wrong} + 可选 revised_text 作为
SFT 训练种子。verdict 自动映射到上游 rating 字段，使上游 thumbs-up/down
aggregate_by_run 统计仍可用。
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_current_user, get_feedback_repo, get_run_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/threads", tags=["feedback"])

VerdictT = Literal["correct", "needs_fix", "wrong"]
_VERDICT_TO_RATING: dict[str, int | None] = {
    "correct": 1,
    "wrong": -1,
    "needs_fix": None,
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
    """Submit Noldus verdict-based feedback for a specific message in a run."""
    run_store = get_run_store(request)
    run = await run_store.get(run_id)
    if run is None or run.get("thread_id") != thread_id:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found in thread {thread_id}",
        )

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
    """List current user's feedback for a run."""
    feedback_repo = get_feedback_repo(request)
    user_id = await get_current_user(request)
    rows = await feedback_repo.list_by_run(thread_id, run_id, user_id=user_id)
    return [_to_item(r).model_dump() for r in rows]
```

- [ ] **Step 2: 更新 app.py 路由挂载注释**

打开 `packages/agent/backend/app/gateway/app.py:367`，把：

```python
    # Feedback API is mounted at /api/threads/{thread_id}/runs/{run_id}/feedback
    app.include_router(feedback.router)
```

替换为：

```python
    # Noldus feedback API at /api/threads/{thread_id}/runs/{run_id}/feedback
    # 走 FeedbackRepository (SQLite)；require_permission 闭 auth bypass；
    # verdict 三分类 + 可选 revised_text（SFT 训练种子）。
    app.include_router(feedback.router)
```

- [ ] **Step 3: 跑 lint 确认无 import 错误**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make lint 2>&1 | tail -20
```

Expected: ruff 通过，无错误。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/app/gateway/routers/feedback.py packages/agent/backend/app/gateway/app.py
git commit -m "$(cat <<'EOF'
feat(feedback): 路由切换到 run-scoped + 闭 auth bypass

URL 从 POST /api/threads/{tid}/feedback 改为 /api/threads/{tid}/runs/{rid}/feedback
对齐上游形状。删除 JSONL 落盘，改走 FeedbackRepository (SQLite)。@require_permission
("threads", "write", owner_check=True, require_existing=True) 闭 auth bypass，
校验 run.thread_id == thread_id 防跨 thread 注入。

verdict→rating 映射使上游 aggregate_by_run 仍可用。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 路由集成测试

**Files:**
- Modify: `packages/agent/backend/tests/test_feedback_api.py` (在 Task 2 基础上追加)

复用现有 `tests/_router_auth_helpers.py` 与 `tests/conftest.py` 提供的 TestClient + auth 注入。

- [ ] **Step 1: 先看现有 router 测试范式**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
grep -A 20 "def client\|TestClient\|require_permission" tests/test_suggestions_router.py 2>&1 | head -40
```

阅读输出，**学习现有 fixture 用法**：哪个 fixture 提供已认证 TestClient、如何注入 user_id、如何 mock `get_run_store`/`get_feedback_repo`。后续步骤用同一套范式。

- [ ] **Step 2: 在 test_feedback_api.py 末尾追加路由测试**

在 `tests/test_feedback_api.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock


def _make_client_with_mocks(
    *,
    user_id: str = "u1",
    runs: dict[str, dict] | None = None,
) -> tuple[TestClient, MagicMock]:
    """Build TestClient with mocked feedback_repo, run_store, and current user.

    Returns (client, feedback_repo_mock) so individual tests can assert on
    upsert/list_by_run calls.
    """
    from app.gateway.app import app
    from app.gateway.deps import get_current_user, get_feedback_repo, get_run_store

    feedback_repo = MagicMock()
    feedback_repo.upsert = AsyncMock(side_effect=lambda **kw: {
        "feedback_id": "fb-1",
        "thread_id": kw["thread_id"],
        "run_id": kw["run_id"],
        "user_id": kw.get("user_id"),
        "message_id": kw.get("message_id"),
        "verdict": kw.get("verdict"),
        "revised_text": kw.get("revised_text"),
        "comment": kw.get("comment"),
        "rating": kw.get("rating"),
        "created_at": "2026-05-12T00:00:00+00:00",
    })
    feedback_repo.list_by_run = AsyncMock(return_value=[])

    run_store = MagicMock()
    run_store.get = AsyncMock(side_effect=lambda rid: runs.get(rid) if runs else None)

    app.dependency_overrides[get_feedback_repo] = lambda: feedback_repo
    app.dependency_overrides[get_run_store] = lambda: run_store
    app.dependency_overrides[get_current_user] = lambda: user_id

    client = TestClient(app)
    return client, feedback_repo


def test_post_without_csrf_returns_403(monkeypatch):
    """R1: POST 不带 X-CSRF-Token → 403。

    CSRFMiddleware 在 should_check_csrf(POST) 路径上拦截 missing 头。
    """
    runs = {"r1": {"thread_id": "t1"}}
    client, _ = _make_client_with_mocks(runs=runs)
    try:
        # 不设置 X-CSRF-Token，也不设置 csrf_token cookie
        res = client.post(
            "/api/threads/t1/runs/r1/feedback",
            json={"message_id": "m1", "verdict": "correct"},
        )
        assert res.status_code == 403, res.text
        assert "CSRF" in res.text or "csrf" in res.text.lower()
    finally:
        client.app.dependency_overrides.clear()


def test_post_with_csrf_returns_200_and_persists():
    """R2: 带 CSRF token + auth → 200，upsert 被调用，verdict + revised_text 持久化。"""
    runs = {"r1": {"thread_id": "t1"}}
    client, repo = _make_client_with_mocks(runs=runs)
    try:
        client.cookies.set("csrf_token", "test-token-1234")
        res = client.post(
            "/api/threads/t1/runs/r1/feedback",
            headers={"X-CSRF-Token": "test-token-1234"},
            json={
                "message_id": "m1",
                "verdict": "needs_fix",
                "revised_text": "should be different",
                "note": "wording",
            },
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["verdict"] == "needs_fix"
        assert body["revised_text"] == "should be different"
        assert body["note"] == "wording"
        # upsert kwargs：verdict 映射 needs_fix→rating=None
        repo.upsert.assert_awaited_once()
        kwargs = repo.upsert.await_args.kwargs
        assert kwargs["verdict"] == "needs_fix"
        assert kwargs["rating"] is None
        assert kwargs["revised_text"] == "should be different"
        assert kwargs["message_id"] == "m1"
    finally:
        client.app.dependency_overrides.clear()


def test_post_run_not_in_thread_returns_404():
    """R4: run_id 存在但属于另一 thread → 404。"""
    runs = {"r1": {"thread_id": "other-thread"}}
    client, _ = _make_client_with_mocks(runs=runs)
    try:
        client.cookies.set("csrf_token", "test-token")
        res = client.post(
            "/api/threads/t1/runs/r1/feedback",
            headers={"X-CSRF-Token": "test-token"},
            json={"message_id": "m1", "verdict": "correct"},
        )
        assert res.status_code == 404, res.text
    finally:
        client.app.dependency_overrides.clear()


def test_post_nonexistent_run_returns_404():
    """R5: run_id 不存在 → 404。"""
    client, _ = _make_client_with_mocks(runs={})  # 任何 run_id 返回 None
    try:
        client.cookies.set("csrf_token", "test-token")
        res = client.post(
            "/api/threads/t1/runs/r1/feedback",
            headers={"X-CSRF-Token": "test-token"},
            json={"message_id": "m1", "verdict": "correct"},
        )
        assert res.status_code == 404, res.text
    finally:
        client.app.dependency_overrides.clear()


def test_post_invalid_verdict_returns_422():
    """R6: verdict 非三选一 → Pydantic 422。"""
    runs = {"r1": {"thread_id": "t1"}}
    client, _ = _make_client_with_mocks(runs=runs)
    try:
        client.cookies.set("csrf_token", "test-token")
        res = client.post(
            "/api/threads/t1/runs/r1/feedback",
            headers={"X-CSRF-Token": "test-token"},
            json={"message_id": "m1", "verdict": "not_a_verdict"},
        )
        assert res.status_code == 422, res.text
    finally:
        client.app.dependency_overrides.clear()


def test_get_returns_feedback_list():
    """R7: GET 返回当前用户反馈列表。"""
    runs = {"r1": {"thread_id": "t1"}}
    client, repo = _make_client_with_mocks(user_id="u1", runs=runs)
    repo.list_by_run = AsyncMock(return_value=[
        {
            "feedback_id": "fb-1",
            "thread_id": "t1",
            "run_id": "r1",
            "user_id": "u1",
            "message_id": "m1",
            "verdict": "correct",
            "revised_text": None,
            "comment": None,
            "rating": 1,
            "created_at": "2026-05-12T00:00:00+00:00",
        }
    ])
    try:
        res = client.get("/api/threads/t1/runs/r1/feedback")
        assert res.status_code == 200, res.text
        items = res.json()
        assert len(items) == 1
        assert items[0]["verdict"] == "correct"
        assert items[0]["message_id"] == "m1"
        repo.list_by_run.assert_awaited_with("t1", "r1", user_id="u1")
    finally:
        client.app.dependency_overrides.clear()


def test_post_two_messages_in_same_run():
    """R8: 同 run 不同 message 各提交一次 → upsert 两次，message_id 不同。"""
    runs = {"r1": {"thread_id": "t1"}}
    client, repo = _make_client_with_mocks(runs=runs)
    try:
        client.cookies.set("csrf_token", "test-token")
        headers = {"X-CSRF-Token": "test-token"}
        for mid in ("m1", "m2"):
            res = client.post(
                "/api/threads/t1/runs/r1/feedback",
                headers=headers,
                json={"message_id": mid, "verdict": "correct"},
            )
            assert res.status_code == 200, res.text
        assert repo.upsert.await_count == 2
        called_mids = [c.kwargs["message_id"] for c in repo.upsert.await_args_list]
        assert called_mids == ["m1", "m2"]
    finally:
        client.app.dependency_overrides.clear()
```

**注意**：`test_post_unauthorized_user_returns_403_or_404`（R3）需要依赖 `@require_permission` 的真实 thread ownership 检查路径，需要 `threads_meta` 数据库行——超出 mock 简单注入的范围。**plan 在此选择跳过 R3，留作集成测试 Task 12 的手动验证项**。这与 spec §7.1.2 R3 略有偏差，但 R1+R4+R5 已能验证主要的访问控制路径。

- [ ] **Step 3: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_feedback_api.py -v 2>&1 | tail -30
```

Expected: ORM 层 7 个 + 路由层 7 个 = 14 个全 PASS。

- [ ] **Step 4: 跑全量回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test 2>&1 | tail -30
```

Expected: 全绿，特别确认 `tests/test_thread_runs*.py`（若存在）和 `tests/test_owner_isolation.py` 不破坏。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/tests/test_feedback_api.py
git commit -m "$(cat <<'EOF'
test(feedback): 路由集成测试覆盖 CSRF / run 校验 / verdict 422 / GET / 多 message

R1 CSRF 拦截、R2 verdict + revised_text 端到端持久化、R4/R5 run 校验、R6 verdict
422、R7 GET 列表、R8 同 run 多 message。R3 跨用户访问跳过 mock 路径，留作 Task 12
手动验证。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: 兼容性回归——thread_runs.py messages 端点仍能挂 feedback

**Files:**
- Modify: `packages/agent/backend/tests/test_feedback_api.py` (追加 C1)

- [ ] **Step 1: 看 thread_runs.py 测试现状**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
ls tests/test_thread_runs* 2>&1
grep -rn "list_thread_messages\|list_by_thread_grouped" tests/ 2>&1 | head -10
```

如果已有覆盖 `list_thread_messages` 端点的测试，直接跑确认未破坏即可，跳过 C1 新写。如没有，写一个最小集成测试。

- [ ] **Step 2: 在 test_feedback_api.py 末尾追加 C1**

```python
def test_thread_runs_messages_endpoint_still_attaches_feedback():
    """C1: GET /api/threads/{tid}/messages 仍能从 list_by_thread_grouped 拿到 feedback。

    feedback 新增 verdict / revised_text 字段不破坏 thread_runs 的 consumer：
    它只读 feedback_id / rating / comment。
    """
    from app.gateway.app import app
    from app.gateway.deps import (
        get_current_user,
        get_feedback_repo,
        get_run_event_store,
    )

    feedback_repo = MagicMock()
    feedback_repo.list_by_thread_grouped = AsyncMock(return_value={
        "r1": {
            "feedback_id": "fb-1",
            "rating": 1,
            "comment": "good",
            # 即使带新字段，consumer 也宽容
            "verdict": "correct",
            "revised_text": None,
        }
    })

    event_store = MagicMock()
    event_store.list_messages = AsyncMock(return_value=[
        {"event_type": "ai_message", "run_id": "r1", "id": "m1"},
    ])

    app.dependency_overrides[get_feedback_repo] = lambda: feedback_repo
    app.dependency_overrides[get_run_event_store] = lambda: event_store
    app.dependency_overrides[get_current_user] = lambda: "u1"

    client = TestClient(app)
    try:
        # 注：此端点需要 require_permission 通过；mock 用户绕开 ownership
        res = client.get("/api/threads/t1/messages")
        # 端点存在且能返回——具体 401/200 视 mock 充分度而定
        # 关键断言：list_by_thread_grouped 被调用了
        feedback_repo.list_by_thread_grouped.assert_awaited()
    finally:
        client.app.dependency_overrides.clear()
```

- [ ] **Step 3: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_feedback_api.py::test_thread_runs_messages_endpoint_still_attaches_feedback -v 2>&1 | tail -20
```

Expected: PASS（验证 `list_by_thread_grouped` 仍被调用）。如果因 require_permission/auth mock 不足导致 401 早返，断言 `assert_awaited` 依然能反映 router 是否进入了 grouped 调用——若进入说明兼容；若没进入，调整 mock 至能进入。

- [ ] **Step 4: 全量回归 + commit**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test 2>&1 | tail -20

cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/tests/test_feedback_api.py
git commit -m "$(cat <<'EOF'
test(feedback): C1 兼容性回归——thread_runs messages 端点仍能挂 feedback

验证 FeedbackRow 新增 verdict / revised_text 字段不破坏 list_by_thread_grouped
的现有 consumer（thread_runs.py:300 只读 feedback_id / rating / comment）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3：nginx cherry-pick

### Task 7: cherry-pick 上游 70737af7 nginx CSRF 修复

**Files:**
- Modify: `packages/agent/docker/nginx/nginx.conf`

- [ ] **Step 1: 看上游 commit 内容**

```bash
cd /home/wangqiuyang/noldus-insight
git show 70737af7 --stat
git show 70737af7 -- docker/nginx/nginx.conf | head -80
```

Expected: 7 处 `proxy_set_header Host $host;` → `proxy_set_header Host $http_host;` 的小改动。

- [ ] **Step 2: 尝试 cherry-pick**

```bash
cd /home/wangqiuyang/noldus-insight
git cherry-pick 70737af7
```

**预期场景 A（顺利）**：cherry-pick 成功，自动 commit。跳到 Step 4。

**预期场景 B（路径冲突）**：上游路径是 `docker/nginx/nginx.conf`，本地是 `packages/agent/docker/nginx/nginx.conf`。会冲突，且 git 找不到对应文件。

- [ ] **Step 3: 若冲突，手动改并继续**

如果场景 B：

```bash
cd /home/wangqiuyang/noldus-insight
git cherry-pick --abort
```

然后手动改 `packages/agent/docker/nginx/nginx.conf`：把所有 `proxy_set_header Host $host;` 替换为 `proxy_set_header Host $http_host;`。

```bash
cd /home/wangqiuyang/noldus-insight
grep -c "proxy_set_header Host \$host;" packages/agent/docker/nginx/nginx.conf
# 替换
sed -i 's|proxy_set_header Host \$host;|proxy_set_header Host \$http_host;|g' packages/agent/docker/nginx/nginx.conf
grep -c "proxy_set_header Host \$http_host;" packages/agent/docker/nginx/nginx.conf
```

Expected: 替换前 N 处（约 7-8），替换后 N 处 $http_host，0 处 $host。

- [ ] **Step 4: 验证 diff**

```bash
cd /home/wangqiuyang/noldus-insight
git diff packages/agent/docker/nginx/nginx.conf | head -40
```

Expected: diff 显示纯粹的 `$host` → `$http_host` 替换，无其他改动。

- [ ] **Step 5: 启动 dev 跑 smoke test**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
timeout 60 make dev 2>&1 | tail -10 &
sleep 30
curl -sv -o /dev/null http://localhost:2026/health 2>&1 | grep -i "host\|HTTP/" | head -5
make stop 2>&1 || true
```

Expected: nginx 启动正常；curl 拿到 200 或重定向。

- [ ] **Step 6: Commit**

如果 Step 2 cherry-pick 成功，commit 已自动；跳过。否则：

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/docker/nginx/nginx.conf
git commit -m "$(cat <<'EOF'
fix(nginx): cherry-pick 上游 70737af7 — Host $http_host 修非标端口 CSRF

上游 70737af7 fix(nginx): resolve CSRF auth failure on non-standard ports (#2796)
make dev 用 2026 非标端口，原 Host $host 丢端口号导致 CSRF cookie 域不匹配。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4：前端 run_id 透传

### Task 8: 侦察 — 确定 run_id 落地方案

**Files:**
- Read-only inspection（写代码在 Task 9）

- [ ] **Step 1: 读 LangGraph SDK 暴露的 useStream 类型**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
grep -rn "interface UseStream\|type UseStream\|onUpdateEvent\|onLangChainEvent" node_modules/@langchain/langgraph-sdk/dist/react.d.ts 2>&1 | head -20
```

如果文件不存在，搜：

```bash
find node_modules/@langchain/langgraph-sdk -name "*.d.ts" | head -5
```

记录：`useStream` 返回值是否暴露 `messages` 数组的 enrichment 钩子。

- [ ] **Step 2: 读 hooks.ts 现有 stream 处理**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
sed -n '203,260p' src/core/threads/hooks.ts
```

确定：`onLangChainEvent` 收到的 event 是否包含 `run_id`（通常 `event.metadata?.run_id` 或 `event.run_id`）。

- [ ] **Step 3: 跑一次 dev 抓个真实 event 看结构**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
make dev 2>&1 > /tmp/dev.log &
sleep 25
echo "现在打开 http://localhost:2026 在浏览器手动发一条消息，F12 看 useStream onLangChainEvent 收到什么"
echo "或：grep run_id /tmp/dev.log | head -5"
```

**给执行者的指示**：如果没法手动操作，至少 `grep run_id /tmp/dev.log` 看后端日志里 run_id 怎么 emit 的。Stop dev 后继续。

- [ ] **Step 4: 选定方案并文档**

把侦察结论写到 `/tmp/feedback-runid-recon.md`：

```markdown
# run_id 落地方案侦察结论

- LangGraph SDK 暴露的 onLangChainEvent event 字段：（如 event.run_id / event.metadata.run_id / 未暴露）
- 当前 hooks.ts 已经接 onLangChainEvent，可在那里捕获 run_id
- 选定方案：
  - [ ] 方案 A：在 useStream 包装层维护 messageRunIds Map<message_id, run_id>，暴露给消费者
  - [ ] 方案 B：把 run_id 写入每条 message 的 response_metadata（如果 SDK 允许）
- 选定理由：…
```

无脏读后基于结论进入 Task 9。

**该 task 不 commit**，仅留下 `/tmp/feedback-runid-recon.md` 给 Task 9 用。

---

### Task 9: 实施 run_id 透传

**Files:**
- Modify: `packages/agent/frontend/src/core/threads/hooks.ts`
- Modify: `packages/agent/frontend/src/core/threads/state.ts` 或相关 thread state 类型文件

**前置**：Task 8 已选定方案。下面给出**方案 A**（messageRunIds Map）的实现，因其与 SDK 解耦最稳。如 Task 8 选 B，参照 §5.1 spec 改写。

- [ ] **Step 1: 在 hooks.ts 顶部加 state hook**

打开 `packages/agent/frontend/src/core/threads/hooks.ts`，在适当位置（大约 `useStream` 调用之前，靠近其他 `useState` 处）加：

```typescript
  // Map<message_id, run_id> — 让 FeedbackButtons 知道反馈该挂到哪个 run。
  // LangGraph SDK 的 Message 类型不带 run_id，从 onLangChainEvent 捕获。
  const [messageRunIds, setMessageRunIds] = useState<Map<string, string>>(
    () => new Map(),
  );
```

如果文件里没有 `useState` 导入，在文件顶部 `import { useState } from "react"` 已有的话不动；没有则加。

- [ ] **Step 2: 在 onLangChainEvent 里捕获 run_id**

找到 hooks.ts:215 `onLangChainEvent(event)`，扩展为：

```typescript
    onLangChainEvent(event) {
      if (event.event === "on_tool_end") {
        listeners.current.onToolEnd?.({
          name: event.name,
          data: event.data,
        });
      }
      // 捕获 message → run_id 映射用于反馈按钮。
      // on_chat_model_end 与 on_chain_end 携带 run_id 及输出 message。
      const runId = (event as { run_id?: string }).run_id;
      const output = (event as { data?: { output?: unknown } }).data?.output;
      if (runId && output && typeof output === "object" && "id" in output) {
        const msgId = (output as { id?: string }).id;
        if (msgId) {
          setMessageRunIds((prev) => {
            if (prev.get(msgId) === runId) return prev;
            const next = new Map(prev);
            next.set(msgId, runId);
            return next;
          });
        }
      }
    },
```

> **注意**：Task 8 侦察的实际 event 结构可能不同。如果 `event.data.output` 不是 message，调整这里——目标是从 event 里取到 `(message_id, run_id)` 对，写进 Map。

- [ ] **Step 3: 把 messageRunIds 加进 hook 返回值**

找到 `useThreadStream` 的 return 语句（grep `return {` 在 hooks.ts），把 `messageRunIds` 加入：

```typescript
  return {
    // ... existing fields
    messageRunIds,
  };
```

- [ ] **Step 4: 跑 typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm typecheck 2>&1 | tail -20
```

Expected: 通过。如果有类型错（特别是 useThreadStream 调用方），暂时不 commit，先 Task 10/11 调整。

- [ ] **Step 5: 暂存（不 commit），进入 Task 10**

```bash
cd /home/wangqiuyang/noldus-insight
git status --short packages/agent/frontend/src/core/threads/hooks.ts
```

Expected: 文件在 modified 状态，等 Task 10 一起 commit（前端改动耦合度高）。

---

## Phase 5：前端 FeedbackButtons 改造

### Task 10: api-client.ts + FeedbackButtons 改造

**Files:**
- Modify: `packages/agent/frontend/src/core/api/api-client.ts`
- Modify: `packages/agent/frontend/src/components/feedback/feedback-buttons.tsx`

- [ ] **Step 1: 改 api-client.ts**

打开 `packages/agent/frontend/src/core/api/api-client.ts`，在顶部 import 区加：

```typescript
import { fetch as csrfFetch } from "./fetcher";
```

把文件末尾 `submitFeedback` 与 `listFeedback` 整体替换为：

```typescript
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

同时把 `FeedbackItem` 类型扩展为后端形状：

```typescript
export interface FeedbackItem {
  feedback_id: string;
  thread_id: string;
  run_id: string;
  user_id: string | null;
  message_id: string | null;
  verdict: FeedbackVerdict | null;
  revised_text: string | null;
  note: string | null;
  created_at: string;
}
```

并把旧的 `extends FeedbackRequest` 形式删掉。

- [ ] **Step 2: 改 feedback-buttons.tsx**

打开 `packages/agent/frontend/src/components/feedback/feedback-buttons.tsx`，整文件替换为：

```typescript
"use client";

import { useState } from "react";

import {
  submitFeedback,
  type FeedbackVerdict,
} from "@/core/api/api-client";
import { cn } from "@/lib/utils";

interface Props {
  threadId: string;
  runId: string;
  messageId: string;
  existingVerdict?: FeedbackVerdict;
  className?: string;
}

function labelOf(v: FeedbackVerdict): string {
  if (v === "correct") return "✅ 正确";
  if (v === "needs_fix") return "⚠️ 需修正";
  return "❌ 错误";
}

export function FeedbackButtons({
  threadId,
  runId,
  messageId,
  existingVerdict,
  className,
}: Props) {
  const [verdict, setVerdict] = useState<FeedbackVerdict | null>(
    existingVerdict ?? null,
  );
  const [expandedVerdict, setExpandedVerdict] =
    useState<FeedbackVerdict | null>(null);
  const [revisedText, setRevisedText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (verdict) {
    return (
      <div
        className={cn("mt-2 text-xs text-muted-foreground", className)}
      >
        已反馈（{labelOf(verdict)}）
      </div>
    );
  }

  const showError = (msg: string) => {
    setError(msg);
    setTimeout(() => setError(null), 3000);
  };

  const handleCorrect = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await submitFeedback(threadId, runId, {
        message_id: messageId,
        verdict: "correct",
      });
      setVerdict("correct");
    } catch (e) {
      console.error("Feedback submission failed:", e);
      showError("提交失败，请重试");
    } finally {
      setSubmitting(false);
    }
  };

  const handleExpand = (v: FeedbackVerdict) => {
    setExpandedVerdict(v);
    setRevisedText("");
  };

  const handleSubmitRevision = async () => {
    if (!expandedVerdict || !revisedText.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitFeedback(threadId, runId, {
        message_id: messageId,
        verdict: expandedVerdict,
        revised_text: revisedText.trim(),
      });
      setVerdict(expandedVerdict);
      setExpandedVerdict(null);
    } catch (e) {
      console.error("Feedback submission failed:", e);
      showError("提交失败，请重试");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={cn("mt-2 flex flex-col gap-2", className)}>
      <div className="flex gap-1">
        <button
          type="button"
          aria-label="正确"
          onClick={handleCorrect}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ✅ 正确
        </button>
        <button
          type="button"
          aria-label="需修正"
          onClick={() => handleExpand("needs_fix")}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ⚠️ 需修正
        </button>
        <button
          type="button"
          aria-label="错误"
          onClick={() => handleExpand("wrong")}
          disabled={submitting}
          className="rounded px-2 py-1 text-xs hover:bg-accent disabled:opacity-50"
        >
          ❌ 错误
        </button>
      </div>
      {expandedVerdict && (
        <div className="flex flex-col gap-1">
          <textarea
            value={revisedText}
            onChange={(e) => setRevisedText(e.target.value)}
            placeholder={
              expandedVerdict === "needs_fix"
                ? "请写出修正版（专家版本）"
                : "请写出正确的版本"
            }
            className="min-h-[80px] rounded border p-2 text-sm"
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleSubmitRevision}
              disabled={submitting || !revisedText.trim()}
              className="rounded bg-primary px-2 py-1 text-xs text-primary-foreground disabled:opacity-50"
            >
              提交
            </button>
            <button
              type="button"
              onClick={() => setExpandedVerdict(null)}
              className="rounded px-2 py-1 text-xs hover:bg-accent"
            >
              取消
            </button>
          </div>
        </div>
      )}
      {error && (
        <div className="text-xs text-destructive">{error}</div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm typecheck 2>&1 | tail -20
```

Expected: 编译通过——可能会有 message-list-item.tsx / subtask-card.tsx 的 prop 缺失错误（runId 必传），下一 task 修。

---

### Task 11: 消费者侧补 runId

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx`

- [ ] **Step 1: 看 message-list-item.tsx 当前调用点**

```bash
cd /home/wangqiuyang/noldus-insight
sed -n '55,75p' packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx
```

- [ ] **Step 2: 改 message-list-item.tsx**

把 `MessageListItem` 函数签名加 `messageRunIds`：

```typescript
export function MessageListItem({
  className,
  message,
  isLoading,
  threadId,
  messageRunIds,
}: {
  className?: string;
  message: Message;
  isLoading?: boolean;
  threadId?: string;
  messageRunIds?: Map<string, string>;
}) {
```

把原 FeedbackButtons 渲染段：

```typescript
      {!isLoading && !isHuman && threadId && message.id && (
        <FeedbackButtons
          threadId={threadId}
          messageId={message.id}
          className="px-1"
        />
      )}
```

替换为：

```typescript
      {!isLoading && !isHuman && threadId && message.id && (() => {
        const runId = messageRunIds?.get(message.id);
        if (!runId) return null; // run_id 还没拿到时不渲染，防止误绑
        return (
          <FeedbackButtons
            threadId={threadId}
            runId={runId}
            messageId={message.id}
            className="px-1"
          />
        );
      })()}
```

- [ ] **Step 3: 改上游 message-list.tsx 把 messageRunIds 透传下来**

```bash
sed -n '60,80p' packages/agent/frontend/src/components/workspace/messages/message-list.tsx
```

找到调用 `<MessageListItem ... />` 的位置，加 `messageRunIds={messageRunIds}` prop。**如果 message-list.tsx 也是 props 接收**，递归上溯到使用 `useThreadStream()` 返回 `messageRunIds` 的源头组件——通常是 workspace chat page 组件。

执行者侦察：

```bash
cd /home/wangqiuyang/noldus-insight
grep -rn "MessageList\b\|useThreadStream\b" packages/agent/frontend/src/components packages/agent/frontend/src/app 2>&1 | head -20
```

按链路把 `messageRunIds` 从 `useThreadStream()` 一路传到 `MessageListItem`。每层加 prop。

- [ ] **Step 4: 改 subtask-card.tsx**

```bash
sed -n '165,180p' packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx
```

把 FeedbackButtons 那段同样改为：先从某处取 runId（subtask 一般有自己的 task_id，但反馈仍需挂到 lead agent run_id；若 subtask 上下文里没有 runId，使用 props 透传），缺则不渲染。

> **执行者注意**：若 subtask 上下文里完全无法拿到 run_id，临时方案是把 FeedbackButtons 暂时从 subtask-card 移除（subtask 阶段先不收反馈），并在交接 handoff 里记录。这与 spec §5.4 一致。

- [ ] **Step 5: typecheck + lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm check 2>&1 | tail -30
```

Expected: 全绿。

- [ ] **Step 6: Commit（前端三 task 一起）**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/frontend/src/core/threads/hooks.ts \
        packages/agent/frontend/src/core/api/api-client.ts \
        packages/agent/frontend/src/components/feedback/feedback-buttons.tsx \
        packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx \
        packages/agent/frontend/src/components/workspace/messages/message-list.tsx \
        packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): 反馈按钮接入 run-scoped 路由 + csrfFetch + 错误提示

submitFeedback/listFeedback 签名加 runId，URL 改 /api/threads/{tid}/runs/{rid}
对齐后端；改 csrfFetch 自动注入 X-CSRF-Token 和 credentials。

useThreadStream 在 onLangChainEvent 里捕获 messageRunIds: Map<message_id, run_id>
并暴露给消费者；MessageListItem 拿不到 runId 时不渲染按钮，防止误绑。

提交失败时按钮下方红字「提交失败，请重试」3 秒后自动消失，console.error 同步记录。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6：集成验证

### Task 12: 端到端手动 QA

**Files:** 无代码改动，仅运行验证。

- [ ] **Step 1: 启动完整应用**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
make dev
```

等待 60s 让所有服务起齐。

- [ ] **Step 2: 浏览器 QA 清单（执行者手动跑）**

打开 `http://localhost:2026`，登录已有用户。

1. 进入 `/workspace/chats/<some-thread-id>` 或新建一个 thread，发一条 message 让 assistant 回复
2. 在 assistant 消息下点 ✅ → 按钮变 "已反馈（✅ 正确）"
3. F12 Network 查看 `POST /api/threads/{tid}/runs/{rid}/feedback`：
   - 状态码 200
   - 请求头含 `X-CSRF-Token`（值非空）
   - 请求头 Cookie 含 `csrf_token=...`
4. 点 ⚠️ → 出现修订文本框 → 写几个字 → 提交 → 按钮变 "已反馈（⚠️ 需修正）"
5. 验证 DB：

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
uv run python -c "
import sqlite3
con = sqlite3.connect('.deer-flow/data/deerflow.db')
for row in con.execute('SELECT verdict, revised_text, message_id, run_id FROM feedback'):
    print(row)
"
```

Expected: 输出至少两行，含 verdict=correct 和 verdict=needs_fix，revised_text 非空。

6. **故障注入 1**：F12 → Application → Cookies → 删除 `csrf_token` → 再点 ✅ → 按钮下方出现红字「提交失败，请重试」，3s 后消失。Network 看到 403 响应。

7. **故障注入 2**：跨用户访问（R3 手动版）。
   - 复制当前 thread 的 URL
   - 退出登录，注册/登录另一个用户
   - 粘贴 URL 访问 → 应该被 require_permission 拦截（404 或 403）

8. 同一 message 二次反馈 → DB 行被 upsert 不增行：

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('.deer-flow/data/deerflow.db')
print('rows:', con.execute('SELECT COUNT(*) FROM feedback').fetchone()[0])
"
```

记录次数前后。

- [ ] **Step 3: 停服**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
```

- [ ] **Step 4: 把 QA 结果记到 Task 13 的 handoff 草稿**

如果 8 项全过，记 "全部 PASS"；任何一项失败，**回到对应 task 修复并重跑此 task**。

**Task 12 不 commit**——仅验证。

---

## Phase 7：文档

### Task 13: 更新 CLAUDE.md + SOP

**Files:**
- Modify: `packages/agent/backend/CLAUDE.md`
- Modify: `CLAUDE.md`
- Modify: `docs/sop/training-data-flywheel-sop.md`

- [ ] **Step 1: 改 backend/CLAUDE.md 的 Feedback 段**

```bash
grep -n "feedback\|Feedback" packages/agent/backend/CLAUDE.md | head -10
```

找到 Feedback 相关段，把"反馈走 JSONL"或类似措辞改为 "反馈走 FeedbackRepository (SQLite)，路由 `POST/GET /api/threads/{tid}/runs/{rid}/feedback`，verdict 三分类 + revised_text"。

- [ ] **Step 2: 改根 CLAUDE.md 第 7 条 + 项目状态修正**

打开 `CLAUDE.md`，找到 "7. **训练数据飞轮已启动**" 段。把"专家反馈走 `/api/threads/{id}/feedback` API + 前端三按钮"改为：

```markdown
7. **训练数据飞轮已启动** — 每次 agent 会话自动录制到 `packages/agent/backend/.deer-flow/training-data/auto-collected/`；专家反馈走 `/api/threads/{tid}/runs/{rid}/feedback` API（**SQLite 后端，verdict 三分类 + revised_text**）+ 前端三按钮。查看累计进度：`cd packages/agent/backend && make training-stats`。详见 [docs/sop/training-data-flywheel-sop.md](docs/sop/training-data-flywheel-sop.md)。
```

在第 12 条之后追加第 13 条：

```markdown
13. **项目状态修正（2026-05-12）** — 本仓库已经吃下 Tier 4 体系（unified persistence、`@require_permission`、`get_effective_user_id`、`UserRow` 等），是**多用户**研究助手。CLAUDE.md 第 11 条之前提到的"v0.1 单用户故意不要 Tier 4"在 2026-05-07/08 Tier234 round1-3 合入后已过时——这些指导仍适用于评估上游 sync 风险，但**本仓库现状**是建立在 Tier 4 之上。
```

- [ ] **Step 3: 改 SOP**

```bash
grep -n "JSONL\|jsonl\|反馈\|feedback" docs/sop/training-data-flywheel-sop.md | head -20
```

找到反馈数据来源段，把"反馈从 `.deer-flow/training-data/feedback/*.jsonl` 读"或类似措辞改为 "反馈从 SQLite `feedback` 表读，schema 见 [packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py]，离线导出脚本见 `scripts/export_feedback_jsonl.py`（待实现，本批暂不交付）"。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/CLAUDE.md CLAUDE.md docs/sop/training-data-flywheel-sop.md
git commit -m "$(cat <<'EOF'
docs: 反馈走 SQLite + run-scoped URL；CLAUDE.md 标注多用户/Tier 4 已合状态

更新根与后端 CLAUDE.md 的反馈路径描述；SOP 同步反馈数据来源；追加项目状态修正条目
（v0.1 单用户描述已过时，Tier 4 体系已合）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: 写完成 handoff

**Files:**
- Create: `docs/handoffs/2026-05/2026-05-12-feedback-csrf-fix-completion-handoff.md`

- [ ] **Step 1: 写 handoff**

新建 `docs/handoffs/2026-05/2026-05-12-feedback-csrf-fix-completion-handoff.md`：

```markdown
# 2026-05-12 反馈按钮 CSRF + auth bypass + 路由升级 完成交接

## 背景

前置交接：[2026-05-12-feedback-button-csrf-and-upstream-sync-handoff.md](2026-05-12-feedback-button-csrf-and-upstream-sync-handoff.md)
设计稿：[../../superpowers/specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md](../../superpowers/specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md)
实施计划：[../../superpowers/plans/2026-05-12-feedback-csrf-fix-and-run-scoped-plan.md](../../superpowers/plans/2026-05-12-feedback-csrf-fix-and-run-scoped-plan.md)

## 解决的问题

1. ✅ 反馈按钮无反应（CSRF header 缺失）
2. ✅ 后端 feedback router auth bypass（任何登录用户可向任意 thread 提交）
3. ✅ nginx 非标端口下 `Host $host` 丢端口号导致 CSRF cookie 域不匹配
4. ✅ JSONL 反馈与上游 SQLite FeedbackRepository 平行存在的架构碎片化
5. ✅ 反馈未带 run_id，无法精确定位训练样本

## 主要改动

### 后端
- 扩展 `FeedbackRow`：加 `verdict (String(16))` + `revised_text (Text)`，`rating` 改 nullable，UniqueConstraint 加 `message_id`
- `FeedbackRepository.upsert` 加 verdict/revised_text/message_id 参数；上游 rating-only 路径保留兼容
- 重写 `routers/feedback.py`：URL 切换到 `/api/threads/{tid}/runs/{rid}/feedback`；加 `@require_permission`
- Alembic migration scaffold（生产部署用）
- 17 条 pytest 测试覆盖（ORM + 路由 + 兼容性）

### 前端
- `submitFeedback/listFeedback` 签名加 `runId`；改 `csrfFetch` 自动 CSRF
- `useThreadStream` 在 `onLangChainEvent` 捕获 message → run_id 映射并暴露
- `FeedbackButtons` 加 `runId` 必传 prop；失败时按钮下方红字提示 3s 自动消失

### 基础设施
- cherry-pick 上游 `70737af7`：nginx `Host $host` → `$http_host`

## QA 验证结果

（填入 Task 12 实际验证 8 项的 PASS/FAIL 与备注）

## 后续未完成项

- 🟡 **PR-B 安全批**：拉上游 6 条 sandbox/upload/auth 安全修复（详见 spec §11 与设计阶段分类清单）
- 🟡 **PR-C 稳定性批**：拉上游 32 条小 bug fix
- 🟡 **PR-D 增强批**：拉上游 12 条 feat/refactor（高风险，需 surgical merge）
- 🟡 **导出脚本**：`scripts/export_feedback_jsonl.py` 离线导 SQLite → Fireworks JSONL 供训练流水线使用（待实现）
- 🟡 **subtask 反馈**：subtask-card.tsx 中的反馈按钮在 subtask 上下文无 run_id 的情况下临时不渲染（如 Task 11 备注），需后续设计 subtask 反馈数据模型

## 验收清单

- [ ] `make test`（backend）全绿
- [ ] `pnpm check`（frontend）全绿
- [ ] 浏览器 8 项 QA 全 PASS
- [ ] DB feedback 表含 verdict/revised_text 字段（`SELECT sql FROM sqlite_master WHERE name='feedback'`）
- [ ] CLAUDE.md / SOP 已更新
```

- [ ] **Step 2: 把 Task 12 验证结果填进去**

回到 Task 12 记录的结果，填入 handoff "QA 验证结果" 段。

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-12-feedback-csrf-fix-completion-handoff.md
git commit -m "$(cat <<'EOF'
docs: 2026-05-12 反馈按钮 CSRF + auth bypass + 路由升级 完成交接

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 验收

全部 14 个 task 完成后：

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && make test
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend && pnpm check
cd /home/wangqiuyang/noldus-insight && git log --oneline -20
```

Expected:
- backend test 全绿（含新增 14+ 条 feedback 测试）
- frontend check 全绿
- git log 看到约 12 个有意义的 commit（每 task 1-2 个）

如需扔给其他 agent 执行，整段 plan 自包含——所有路径、命令、代码块、断言都已写明。
