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
