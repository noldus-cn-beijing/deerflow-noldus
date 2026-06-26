"""Task 5 (P2) legacy thread 观测（spec 2026-06-26 §任务5）。

调研一度判 legacy 路径「无审计」，实读发现 ``user_dir`` 已有迁移 + 审计日志
（paths.py:181-203）。故本任务不动路径逻辑，仅补一条启动观测：统计
``threads_meta.user_id IS NULL`` 行数并日志告警，便于运维判断是否还有老部署遗留。
"""

import logging

import pytest

from deerflow.persistence.thread_meta import ThreadMetaRepository


@pytest.fixture
async def repo(tmp_path):
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine

    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    await init_engine("sqlite", url=url, sqlite_dir=str(tmp_path))
    yield ThreadMetaRepository(get_session_factory())
    await close_engine()


class TestLegacyThreadObservation:
    """count_legacy_orphans：统计 user_id IS NULL 的 thread_meta 行数。"""

    @pytest.mark.anyio
    async def test_count_zero_when_no_legacy(self, repo):
        assert await repo.count_legacy_orphans() == 0

    @pytest.mark.anyio
    async def test_count_legacy_null_user_rows(self, repo):
        """三条无 user_id 行 + 两条有 user_id 行 → 计数 3。"""
        await repo.create("legacy-1", user_id=None)
        await repo.create("legacy-2", user_id=None)
        await repo.create("legacy-3", user_id=None)
        await repo.create("owned-1", user_id="user1")
        await repo.create("owned-2", user_id="user2")
        assert await repo.count_legacy_orphans() == 3

    @pytest.mark.anyio
    async def test_observe_logs_warning_when_legacy_present(self, repo, caplog):
        """有 legacy 行 → emit 含 legacy_thread_count 的 warning 日志。"""
        from app.gateway.legacy_observer import observe_legacy_threads

        await repo.create("legacy-1", user_id=None)
        # 模拟 gateway 启动：session factory 在 app.state 上可用时即观测
        from deerflow.persistence.engine import get_session_factory

        with caplog.at_level(logging.WARNING, logger="app.gateway.legacy_observer"):
            await observe_legacy_threads(session_factory=get_session_factory())

        warnings = [r for r in caplog.records if "legacy_thread_count" in r.getMessage()]
        assert warnings, "有 legacy 行时应 emit 含 legacy_thread_count 的 warning"
        assert "legacy_thread_count=1" in warnings[0].getMessage()

    @pytest.mark.anyio
    async def test_observe_silent_when_no_legacy(self, repo, caplog):
        """无 legacy 行 → 不 emit warning（不污染日志）。"""
        from app.gateway.legacy_observer import observe_legacy_threads
        from deerflow.persistence.engine import get_session_factory

        with caplog.at_level(logging.WARNING, logger="app.gateway.legacy_observer"):
            await observe_legacy_threads(session_factory=get_session_factory())

        warnings = [r for r in caplog.records if "legacy_thread_count" in r.getMessage()]
        assert warnings == []

    @pytest.mark.anyio
    async def test_observe_no_session_factory_is_noop(self, caplog):
        """memory backend（无 session factory）→ 静默跳过（观测是 best-effort）。"""
        from app.gateway.legacy_observer import observe_legacy_threads

        # 不应抛异常
        await observe_legacy_threads(session_factory=None)
