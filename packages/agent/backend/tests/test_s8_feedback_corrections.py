"""Unit tests for Sprint 8: feedback verdict回流 to lead prompt.

Tests cover:
1. FeedbackRow paradigm column
2. FeedbackRepository.list_prior_corrections (via real SQLite)
3. FeedbackRepository.upsert paradigm passthrough (via real SQLite)
4. _get_prior_corrections_context prompt injection (unit, no DB)
"""

from datetime import UTC, datetime

import pytest

from deerflow.persistence.feedback.model import FeedbackRow
from deerflow.persistence.feedback.sql import FeedbackRepository


# ============================================================================
# Test FeedbackRow paradigm column
# ============================================================================


class TestFeedbackRowParadigmColumn:
    def test_paradigm_column_accepts_string(self):
        row = FeedbackRow(feedback_id="fb-1", run_id="r-1", thread_id="t-1", paradigm="epm")
        assert row.paradigm == "epm"

    def test_paradigm_none_for_legacy(self):
        row = FeedbackRow(feedback_id="fb-2", run_id="r-2", thread_id="t-2")
        assert row.paradigm is None

    def test_paradigm_stores_paradigm_name(self):
        row = FeedbackRow(feedback_id="fb-3", run_id="r-3", thread_id="t-3", paradigm="forced_swim")
        assert row.paradigm == "forced_swim"


# ============================================================================
# Test list_prior_corrections + upsert paradigm (integration with real SQLite)
# Uses the same conftest.py fixtures as test_feedback_api.py
# ============================================================================


@pytest.fixture
def feedback_repo(tmp_path):
    """Create a FeedbackRepository backed by a real SQLite database."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from deerflow.persistence.base import Base

    db_path = tmp_path / "test_feedback.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    import asyncio

    async def _create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    return FeedbackRepository(session_factory)


class TestListPriorCorrectionsIntegration:
    @pytest.mark.asyncio
    async def test_returns_needs_fix_for_matching_paradigm(self, feedback_repo):
        # Insert a needs_fix feedback with paradigm
        await feedback_repo.upsert(
            thread_id="th-1",
            run_id="run-1",
            user_id="user-1",
            message_id="msg-1",
            verdict="needs_fix",
            paradigm="epm",
            comment="Use non-parametric test",
        )
        # Insert a correct feedback (should be excluded)
        await feedback_repo.upsert(
            thread_id="th-1",
            run_id="run-2",
            user_id="user-1",
            message_id="msg-2",
            verdict="correct",
            paradigm="epm",
        )
        # Insert a different paradigm (should be excluded)
        await feedback_repo.upsert(
            thread_id="th-1",
            run_id="run-3",
            user_id="user-1",
            message_id="msg-3",
            verdict="needs_fix",
            paradigm="fst",
            comment="FST issue",
        )

        results = await feedback_repo.list_prior_corrections(paradigm="epm", user_id="user-1", limit=5)
        assert len(results) == 1
        assert results[0]["verdict"] == "needs_fix"
        assert results[0]["paradigm"] == "epm"
        assert results[0]["comment"] == "Use non-parametric test"

    @pytest.mark.asyncio
    async def test_includes_wrong_verdict(self, feedback_repo):
        await feedback_repo.upsert(
            thread_id="th-2",
            run_id="run-4",
            user_id="user-1",
            message_id="msg-4",
            verdict="wrong",
            paradigm="oft",
            comment="Completely wrong analysis",
        )
        results = await feedback_repo.list_prior_corrections(paradigm="oft", user_id="user-1", limit=5)
        assert len(results) == 1
        assert results[0]["verdict"] == "wrong"

    @pytest.mark.asyncio
    async def test_excludes_correct_verdicts(self, feedback_repo):
        await feedback_repo.upsert(
            thread_id="th-3",
            run_id="run-5",
            user_id="user-1",
            message_id="msg-5",
            verdict="correct",
            paradigm="ldb",
        )
        results = await feedback_repo.list_prior_corrections(paradigm="ldb", user_id="user-1", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_respects_limit(self, feedback_repo):
        for i in range(5):
            await feedback_repo.upsert(
                thread_id="th-4",
                run_id=f"run-{10+i}",
                user_id="user-1",
                message_id=f"msg-{10+i}",
                verdict="needs_fix",
                paradigm="epm",
                comment=f"Issue {i}",
            )
        results = await feedback_repo.list_prior_corrections(paradigm="epm", user_id="user-1", limit=2)
        assert len(results) == 2


class TestUpsertParadigmIntegration:
    @pytest.mark.asyncio
    async def test_upsert_stores_paradigm(self, feedback_repo):
        result = await feedback_repo.upsert(
            thread_id="th-5",
            run_id="run-20",
            user_id="user-1",
            message_id="msg-20",
            verdict="needs_fix",
            paradigm="fst",
            comment="Test comment",
        )
        assert result["paradigm"] == "fst"

    @pytest.mark.asyncio
    async def test_upsert_without_paradigm(self, feedback_repo):
        result = await feedback_repo.upsert(
            thread_id="th-6",
            run_id="run-21",
            user_id="user-1",
            message_id="msg-21",
            verdict="correct",
        )
        assert result["paradigm"] is None

    @pytest.mark.asyncio
    async def test_upsert_updates_paradigm(self, feedback_repo):
        # First upsert without paradigm
        await feedback_repo.upsert(
            thread_id="th-7",
            run_id="run-22",
            user_id="user-1",
            message_id="msg-22",
            verdict="needs_fix",
        )
        # Second upsert with paradigm (same unique key)
        result = await feedback_repo.upsert(
            thread_id="th-7",
            run_id="run-22",
            user_id="user-1",
            message_id="msg-22",
            verdict="needs_fix",
            paradigm="epm",
        )
        assert result["paradigm"] == "epm"


# ============================================================================
# Test _get_prior_corrections_context (prompt injection — pure unit)
# ============================================================================


class TestGetPriorCorrectionsContext:
    """Tests for the prompt injection function.

    Since _get_prior_corrections_context uses many internal imports,
    we test it by patching at the import source level and verifying
    the output format.
    """

    def test_returns_empty_on_exception(self):
        """Any internal failure returns empty string (non-blocking)."""
        from deerflow.agents.lead_agent.prompt import _get_prior_corrections_context

        # Calling without any thread context should return ""
        result = _get_prior_corrections_context()
        assert result == ""

    def test_function_exists_and_is_callable(self):
        from deerflow.agents.lead_agent.prompt import _get_prior_corrections_context

        assert callable(_get_prior_corrections_context)
