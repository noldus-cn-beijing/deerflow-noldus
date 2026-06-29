"""Noldus bootstrap regression for ``deerflow.persistence.bootstrap.bootstrap_schema``.

Spec 2026-06-29-sync-deerflow-stage-c-alembic-bootstrap-deploy §四.

We collapsed onto the upstream single-root alembic chain
(``0001_baseline → 0002_runs_token_usage → 0003_noldus_feedback_ext``). The
Noldus increment (``0003``) is what restores the three Noldus ``feedback``
columns (``verdict`` / ``revised_text`` / ``paradigm``), makes ``rating``
nullable, and re-adds the ``runs``/``run_events`` → ``threads_meta`` cascade FK.

These tests pin the **real end-state** of ``bootstrap_schema`` across the two
DB states that exercise ``0003`` — not just "it ran":
  - empty DB  → create_all builds the Noldus shape directly from our ORM;
  - legacy DB (upstream-baseline tables, no alembic_version, rating NOT NULL,
    no Noldus cols, no cascade FK) → ``0003`` ALTERs/adds them in.

In BOTH states the terminal ``feedback`` table must contain ``verdict`` /
``revised_text`` / ``paradigm`` with ``rating`` nullable, and the cascade FK
must exist on ``runs.thread_id``. ``alembic_version`` must read the new head
``0003_noldus_feedback_ext``.

Honours memory feedback_processpoolexecutor_test_runner_injection_and_ssot_parity
(assert the real product — columns genuinely present, not just "no exception")
and feedback_deterministic_html_image_product_needs_end_to_end_reparse_assertion
(end-to-end reality, not placeholder substitution).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

# Pre-import models so Base.metadata is populated before bootstrap reads it.
import deerflow.persistence.models  # noqa: F401
from deerflow.persistence.bootstrap import bootstrap_schema

HEAD = "0003_noldus_feedback_ext"

# Cascade FK constraint name assigned by the 0003 migration (legacy/upgraded DBs).
# On a fresh-DB create_all path SQLAlchemy leaves the FK unnamed (SQLite reflection
# reports name=None), so the fresh-DB assertion matches by semantics, not by name.
_RUNS_THREAD_FK = "fk_runs_thread_id_threads_meta"


def _has_runs_thread_cascade_fk(runs_fks: list) -> bool:
    """True if runs has a thread_id→threads_meta FK with ondelete=CASCADE."""
    return any(
        fk.get("constrained_columns") == ["thread_id"]
        and fk.get("referred_table") == "threads_meta"
        and fk.get("options", {}).get("ondelete") == "CASCADE"
        for fk in runs_fks
    )


def _url(tmp_path: Path, name: str = "test.db") -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}"


async def _alembic_version(engine) -> str | None:
    async with engine.connect() as conn:
        row = await conn.execute(sa.text("SELECT version_num FROM alembic_version"))
        return row.scalar()


async def _feedback_columns(engine) -> dict[str, dict]:
    async with engine.connect() as conn:
        cols = await conn.run_sync(lambda c: sa.inspect(c).get_columns("feedback"))
    return {c["name"]: c for c in cols}


async def _feedback_fks(engine):
    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: sa.inspect(c).get_foreign_keys("feedback"))


async def _runs_fks(engine):
    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: sa.inspect(c).get_foreign_keys("runs"))


async def _seed_legacy_upstream_baseline(engine) -> None:
    """A legacy DB the new chain has to repair.

    Models a DB provisioned by the *upstream* ``0001_baseline`` shape (e.g. a
    DB that lived under the upstream chain before we folded Noldus in, or a
    hand-built prod-like fixture): DeerFlow tables present, NO
    ``alembic_version`` table, ``feedback.rating`` NOT NULL, none of the three
    Noldus columns, no cascade FK. Bootstrap's legacy branch must stamp
    ``0001_baseline`` and run ``upgrade head`` — which applies ``0003`` and
    restores the full Noldus shape.
    """
    async with engine.begin() as conn:
        await conn.run_sync(_create_upstream_baseline_schema)


def _create_upstream_baseline_schema(sync_conn) -> None:
    """Create the upstream ``0001_baseline`` tables directly via raw DDL.

    We deliberately do NOT use ``Base.metadata.create_all`` here (that would
    build the Noldus shape from our ORM and defeat the "legacy" premise). The
    DDL mirrors ``0001_baseline.upgrade()``: ``rating`` NOT NULL, no
    verdict/revised_text/paradigm, no cascade FK on runs/run_events.
    """
    sync_conn.exec_driver_sql(
        """
        CREATE TABLE threads_meta (
            thread_id VARCHAR(64) NOT NULL,
            assistant_id VARCHAR(128),
            user_id VARCHAR(64),
            display_name VARCHAR(256),
            status VARCHAR(20) NOT NULL,
            metadata_json JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (thread_id)
        )
        """
    )
    sync_conn.exec_driver_sql(
        """
        CREATE TABLE runs (
            run_id VARCHAR(64) NOT NULL,
            thread_id VARCHAR(64) NOT NULL,
            status VARCHAR(20) NOT NULL,
            multitask_strategy VARCHAR(20) NOT NULL,
            metadata_json JSON NOT NULL,
            kwargs_json JSON NOT NULL,
            message_count INTEGER NOT NULL,
            total_input_tokens INTEGER NOT NULL,
            total_output_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            llm_call_count INTEGER NOT NULL,
            lead_agent_tokens INTEGER NOT NULL,
            subagent_tokens INTEGER NOT NULL,
            middleware_tokens INTEGER NOT NULL,
            token_usage_by_model JSON NOT NULL DEFAULT '{}',
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (run_id)
        )
        """
    )
    sync_conn.exec_driver_sql(
        """
        CREATE TABLE feedback (
            feedback_id VARCHAR(64) NOT NULL,
            run_id VARCHAR(64) NOT NULL,
            thread_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64),
            message_id VARCHAR(64),
            rating INTEGER NOT NULL,
            comment TEXT,
            created_at DATETIME NOT NULL,
            PRIMARY KEY (feedback_id)
        )
        """
    )


def _assert_noldus_feedback_shape(cols: dict[str, dict]) -> None:
    """The terminal feedback table must carry the Noldus columns + nullable rating."""
    for noldus_col in ("verdict", "revised_text", "paradigm"):
        assert noldus_col in cols, f"feedback.{noldus_col} missing after bootstrap (cols: {sorted(cols)})"
    assert cols["rating"]["nullable"] is True, (
        f"feedback.rating must be nullable (verdict-only path writes no rating); got nullable={cols['rating']['nullable']}"
    )


@pytest.mark.anyio
async def test_empty_db_bootstrap_reaches_noldus_head(tmp_path: Path):
    """Empty DB → create_all + stamp head; feedback has the full Noldus shape."""
    engine = create_async_engine(_url(tmp_path, "empty.db"))
    try:
        await bootstrap_schema(engine, backend="sqlite")

        assert await _alembic_version(engine) == HEAD
        cols = await _feedback_columns(engine)
        _assert_noldus_feedback_shape(cols)

        # Cascade FK present (built by create_all from our ORM on fresh DB).
        runs_fks = await _runs_fks(engine)
        assert _has_runs_thread_cascade_fk(runs_fks), (
            f"runs.thread_id cascade FK (ondelete=CASCADE) missing on fresh DB (fks: {runs_fks})"
        )
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_legacy_db_bootstrap_applies_0003_noldus_increment(tmp_path: Path):
    """Legacy upstream-baseline DB → stamp baseline + upgrade head applies 0003.

    The legacy fixture has feedback.rating NOT NULL, no verdict/revised_text/
    paradigm, no cascade FK. After bootstrap the table must match the Noldus
    ORM shape exactly — proving 0003's ALTER + safe_add_column + FK add all
    landed on a real pre-existing table (not a fresh create_all).
    """
    engine = create_async_engine(_url(tmp_path, "legacy.db"))
    try:
        await _seed_legacy_upstream_baseline(engine)
        await bootstrap_schema(engine, backend="sqlite")

        assert await _alembic_version(engine) == HEAD
        cols = await _feedback_columns(engine)
        _assert_noldus_feedback_shape(cols)

        # 0003 added the cascade FK onto the pre-existing runs table.
        runs_fks = await _runs_fks(engine)
        assert _has_runs_thread_cascade_fk(runs_fks), (
            f"runs.thread_id cascade FK (ondelete=CASCADE) missing after 0003 on legacy DB (fks: {runs_fks})"
        )
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_bootstrap_is_idempotent_at_head(tmp_path: Path):
    """Re-running bootstrap on a DB already at head is a no-op (still head, shape intact)."""
    engine = create_async_engine(_url(tmp_path, "idem.db"))
    try:
        await bootstrap_schema(engine, backend="sqlite")
        await bootstrap_schema(engine, backend="sqlite")  # second run must not error

        assert await _alembic_version(engine) == HEAD
        cols = await _feedback_columns(engine)
        _assert_noldus_feedback_shape(cols)
    finally:
        await engine.dispose()
