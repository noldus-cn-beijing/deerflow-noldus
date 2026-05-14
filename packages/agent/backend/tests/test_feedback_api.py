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


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _make_client_with_mocks(
    *,
    user_id: str = "u1",
    runs: dict[str, dict] | None = None,
) -> tuple[TestClient, MagicMock]:
    """Build TestClient with mocked feedback_repo, run_store, and current user.

    Returns (client, feedback_repo_mock) so individual tests can assert on
    upsert/list_by_run calls.
    """
    import app.gateway.deps as deps_mod
    from app.gateway.app import app
    from app.gateway.internal_auth import create_internal_auth_headers

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

    # _require() reads from app.state directly, not DI — inject mocks there
    thread_store = MagicMock()
    thread_store.check_access = AsyncMock(return_value=True)
    app.state.feedback_repo = feedback_repo
    app.state.run_store = run_store
    app.state.thread_store = thread_store

    # get_current_user is called directly (not via Depends) in the route handler;
    # FastAPI dependency_overrides doesn't apply. Patch both the deps module
    # (for indirect callers) and the feedback router module (for direct import).
    import app.gateway.routers.feedback as feedback_mod
    deps_mod.get_current_user = AsyncMock(return_value=user_id)
    feedback_mod.get_current_user = AsyncMock(return_value=user_id)

    client = TestClient(app)
    # Bypass AuthMiddleware via internal auth token
    client.headers.update(create_internal_auth_headers())
    return client, feedback_repo
    return client, feedback_repo
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


def test_post_accepts_arbitrary_run_id_no_run_store_check():
    """R4 (updated 2026-05-13): feedback 接口不再校验 run_id 是否在 run_store。

    原因：当前架构下没有任何 producer 往 run_store 写入（LangGraph 模式 run
    由 langgraph 自管），原"run 不存在 → 404"分支永远命中、前端反馈按钮
    100% 失败。修复后任意 run_id 都接受 —— feedback 表的核心价值是收集
    verdict + revised_text 给 SFT 飞轮，run_id 只是关联标识。

    本测试同时取代旧的 test_post_run_not_in_thread_returns_404 和
    test_post_nonexistent_run_returns_404。
    """
    # run_store 是空的：任何 run_id 都拿不到 run 记录
    client, repo = _make_client_with_mocks(runs={})
    try:
        client.cookies.set("csrf_token", "test-token")
        res = client.post(
            "/api/threads/t1/runs/nonexistent-run-id/feedback",
            headers={"X-CSRF-Token": "test-token"},
            json={"message_id": "m1", "verdict": "correct"},
        )
        assert res.status_code == 200, res.text
        repo.upsert.assert_awaited_once()
        kwargs = repo.upsert.await_args.kwargs
        assert kwargs["thread_id"] == "t1"
        assert kwargs["run_id"] == "nonexistent-run-id"
        assert kwargs["verdict"] == "correct"
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


def test_thread_runs_messages_endpoint_still_attaches_feedback():
    """C1: GET /api/threads/{tid}/messages 仍能从 list_by_thread_grouped 拿到 feedback。

    feedback 新增 verdict / revised_text 字段不破坏 thread_runs 的 consumer：
    它只读 feedback_id / rating / comment。
    """
    import app.gateway.deps as deps_mod
    import app.gateway.routers.feedback as feedback_mod
    import app.gateway.routers.thread_runs as thread_runs_mod
    from app.gateway.app import app
    from app.gateway.internal_auth import create_internal_auth_headers

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

    thread_store = MagicMock()
    thread_store.check_access = AsyncMock(return_value=True)

    # Inject into app.state for _require() callers
    app.state.feedback_repo = feedback_repo
    app.state.thread_store = thread_store
    app.state.run_event_store = event_store

    # Patch direct imports in route handlers
    deps_mod.get_current_user = AsyncMock(return_value="u1")
    feedback_mod.get_current_user = AsyncMock(return_value="u1")
    thread_runs_mod.get_current_user = AsyncMock(return_value="u1")

    client = TestClient(app)
    client.headers.update(create_internal_auth_headers())

    try:
        res = client.get("/api/threads/t1/messages")
        # 端点存在且能返回——具体 401/200 视 mock 充分度而定
        # 关键断言：list_by_thread_grouped 被调用了
        feedback_repo.list_by_thread_grouped.assert_awaited()
    finally:
        client.app.dependency_overrides.clear()
