"""Task 1 (P0): DB↔磁盘一致性 — 外键级联 + 删除事务化.

spec 2026-06-26-file-path-reliability-loadbearing-convergence-spec §任务1.

验证三件事：
1. 删 threads_meta 行 → runs / run_events 行被 FK ON DELETE CASCADE 自动归零
   （SQLite 需 PRAGMA foreign_keys=ON，engine.py 已在每条连接上设置）。
2. 删除路径事务化：DB 事务失败时磁盘目录不动（不产生「DB 在、磁盘没」的反向孤儿
   是另一方向；这里防的是「meta 没删成但磁盘删了 → search 还列着、点进去 404」）。
3. 结构化观测日志：删除完成 emit thread_id / user_id / runs_deleted_count / dir_existed。
"""

import pytest

from deerflow.persistence.run import RunRepository
from deerflow.persistence.thread_meta import ThreadMetaRepository


@pytest.fixture
async def stores(tmp_path):
    """初始化 sqlite 引擎 + 两个 SQL repo（共用同一 session factory）。"""
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine

    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    await init_engine("sqlite", url=url, sqlite_dir=str(tmp_path))
    yield (
        ThreadMetaRepository(get_session_factory()),
        RunRepository(get_session_factory()),
    )
    await close_engine()


class TestDeleteCascade:
    """runs.thread_id / run_events.run_id 外键级联。"""

    @pytest.mark.anyio
    async def test_delete_thread_cascades_runs(self, stores):
        """建 thread + 2 runs，删 thread，断言 runs 表对应行归零。"""
        thread_repo, run_repo = stores
        await thread_repo.create("thread-A", user_id=None)
        await run_repo.put("run-1", thread_id="thread-A", user_id=None)
        await run_repo.put("run-2", thread_id="thread-A", user_id=None)

        # 前置：两行在
        runs = await run_repo.list_by_thread("thread-A", user_id=None)
        assert len(runs) == 2

        # 删 thread_meta → FK CASCADE 应连带删除 runs 行
        await thread_repo.delete("thread-A", user_id=None)

        remaining = await run_repo.list_by_thread("thread-A", user_id=None)
        assert remaining == []

    @pytest.mark.anyio
    async def test_delete_thread_cascades_run_events(self, stores):
        """run_events.run_id → runs.run_id 级联：删 thread 连带清事件。"""
        from sqlalchemy import select, text

        from deerflow.persistence.engine import get_engine
        from deerflow.persistence.models.run_event import RunEventRow

        thread_repo, run_repo = stores
        await thread_repo.create("thread-E", user_id=None)
        await run_repo.put("run-E1", thread_id="thread-E", user_id=None)

        engine = get_engine()
        # 直接插一条 run_events 行（run_events 无独立 repo 写入入口）。走 ORM
        # 让 Python 端 default 填全 NOT NULL 列（content/event_metadata 等）。
        from deerflow.persistence.models.run_event import RunEventRow

        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(engine) as session:
            session.add(RunEventRow(thread_id="thread-E", run_id="run-E1", event_type="lifecycle", category="lifecycle", seq=1))
            await session.commit()

        async with engine.connect() as conn:
            rows = (await conn.execute(select(RunEventRow).where(RunEventRow.run_id == "run-E1"))).all()
            assert len(rows) == 1

        # 删 thread → cascade runs → cascade run_events
        await thread_repo.delete("thread-E", user_id=None)

        async with engine.connect() as conn:
            rows = (await conn.execute(select(RunEventRow).where(RunEventRow.run_id == "run-E1"))).all()
            assert rows == []

    @pytest.mark.anyio
    async def test_foreign_key_enforced_rejects_orphan_run(self, stores):
        """PRAGMA foreign_keys=ON + FK 建成时，往 runs 插一个不存在的 thread_id 应被拒。

        这是级联生效的前提证据：如果 FK 没建成或 pragma 没开，这里不会抛
        IntegrityError（孤儿静默写入）。通过 ORM put（自动填全 NOT NULL 字段）
        确保唯一可能的失败来源就是外键约束本身。
        """
        from sqlalchemy.exc import IntegrityError

        thread_repo, run_repo = stores  # noqa: F841 — 不建 thread，故意造孤儿
        with pytest.raises(IntegrityError):
            await run_repo.put("orphan-run", thread_id="no-such-thread", user_id=None)


class TestTransactionalDeleteOrdering:
    """删除路径事务化：DB 失败 → 磁盘不动。"""

    @pytest.mark.anyio
    async def test_db_fail_keeps_disk(self, tmp_path):
        """mock thread_store.delete 抛错 → 磁盘目录未被删 + checkpointer 不动。"""
        from unittest.mock import AsyncMock

        from app.gateway.routers import threads as threads_router
        from deerflow.config.paths import Paths

        paths = Paths(tmp_path)
        thread_id = "thread-disk"
        # 预建磁盘目录
        workspace = paths.sandbox_work_dir(thread_id)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "notes.txt").write_text("hello", encoding="utf-8")
        thread_dir = paths.thread_dir(thread_id)
        assert thread_dir.exists()

        # thread_store.delete 抛错（模拟 DB 事务失败）
        failing_store = AsyncMock()
        failing_store.delete = AsyncMock(side_effect=RuntimeError("db transaction failed"))

        checkpointer = AsyncMock()
        checkpointer.adelete_thread = AsyncMock()

        # DB 失败应冒泡（调用方决定如何回应），但磁盘与 checkpointer 不动
        with pytest.raises(RuntimeError):
            await threads_router._delete_thread_cascaded(
                thread_id,
                paths=paths,
                thread_store=failing_store,
                checkpointer=checkpointer,
            )

        # 关键断言：DB 失败 → 磁盘目录仍在（未产生「meta 没删、磁盘删了」孤儿）
        assert thread_dir.exists(), "DB 删除失败时磁盘目录不应被删除"
        # checkpointer 也不应在 DB 失败时被调（DB-first 顺序）
        checkpointer.adelete_thread.assert_not_called()

    @pytest.mark.anyio
    async def test_successful_delete_emits_observation_log(self, tmp_path, caplog):
        """删除成功 → emit 结构化日志（thread_id / runs_deleted_count / dir_existed）。"""
        import logging
        from unittest.mock import AsyncMock

        from app.gateway.routers import threads as threads_router
        from deerflow.config.paths import Paths

        paths = Paths(tmp_path)
        thread_id = "thread-ok"
        workspace = paths.sandbox_work_dir(thread_id)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "notes.txt").write_text("hello", encoding="utf-8")

        thread_store = AsyncMock()
        thread_store.delete = AsyncMock()

        checkpointer = AsyncMock()
        checkpointer.adelete_thread = AsyncMock()

        with caplog.at_level(logging.INFO, logger="app.gateway.routers.threads"):
            await threads_router._delete_thread_cascaded(
                thread_id,
                paths=paths,
                thread_store=thread_store,
                checkpointer=checkpointer,
                runs_deleted_count=3,
            )

        # 结构化观测日志：含 thread_id + runs_deleted_count + dir_existed
        structured = [r for r in caplog.records if "thread_deleted" in r.getMessage()]
        assert structured, "删除完成应 emit 一条含 thread_deleted 的结构化日志"
        msg = structured[0].getMessage()
        assert thread_id in msg
        assert "runs_deleted_count" in msg
        assert "dir_existed" in msg
