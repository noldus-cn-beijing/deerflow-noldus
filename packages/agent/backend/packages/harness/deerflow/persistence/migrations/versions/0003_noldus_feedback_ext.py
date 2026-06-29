"""Noldus feedback extension + thread-cascade FK (folds our pre-alembic chain).

Revision ID: 0003_noldus_feedback_ext
Revises: 0002_runs_token_usage
Create Date: 2026-06-29

This revision is the **single Noldus increment on top of the upstream
alembic chain** (``0001_baseline → 0002_runs_token_usage → this``). It folds
two pieces of pre-alembic Noldus schema into one clean increment, replacing
the four old date-stamped revisions that were deleted when we collapsed onto
the upstream single-root chain (spec
``2026-06-29-sync-deerflow-stage-c-alembic-bootstrap-deploy``):

1. **feedback table Noldus columns + nullable rating.**
   Upstream ``0001_baseline`` declares ``feedback.rating`` as ``NOT NULL`` and
   omits the three Noldus columns ``verdict`` / ``revised_text`` / ``paradigm``.
   Our ORM (``deerflow.persistence.feedback.model``) declares ``rating`` as
   ``nullable=True`` (the verdict-only feedback path never writes a rating)
   plus the three columns above. On a **fresh DB** ``Base.metadata.create_all``
   already builds the table with the Noldus shape, so this revision only
   matters for **legacy/upgraded DBs** that were stamped at ``0001``/``0002`` —
   it ALTERs the rating nullability and adds the three columns idempotently.

2. **runs / run_events → threads_meta ON DELETE CASCADE FK.**
   Upstream ``0001_baseline`` declares these columns with no foreign key. Our
   ORM (``run/model.py`` + ``models/run_event.py``) declares
   ``ForeignKey(..., ondelete="CASCADE")``. On a fresh DB ``create_all`` builds
   the FK directly; on a legacy/upgraded DB this revision adds it idempotently.
   Preserves the cascade semantic of the deleted ``20260626_1700`` revision
   (孤儿 run/event 行 cleanup when a thread is deleted).

Idempotency
-----------

Column adds go through ``safe_add_column`` (no-op + drift warning when the
column already exists). The nullability ALTER and FK adds are guarded by
``inspect``-based checks so re-running against a DB already at this revision
is a safe no-op. SQLite cannot ``ALTER COLUMN`` / ``ADD CONSTRAINT`` in place,
so both use ``batch_alter_table`` (which transparently rebuilds the table via
the temp-copy + rename dance and preserves existing data).
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from deerflow.persistence.migrations._helpers import safe_add_column

# revision identifiers, used by Alembic.
revision: str = "0003_noldus_feedback_ext"
down_revision: str | Sequence[str] | None = "0002_runs_token_usage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Cascade FK constraint names — kept identical to the deleted
# ``20260626_1700_thread_cascade_fk`` revision so a DB previously migrated by
# that revision (then re-stamped onto the upstream chain during the wipe &
# re-bootstrap) already has these names and the guard below no-ops.
_RUNS_THREAD_FK = "fk_runs_thread_id_threads_meta"
_EVENTS_THREAD_FK = "fk_run_events_thread_id_threads_meta"
_EVENTS_RUN_FK = "fk_run_events_run_id_runs"


def _existing_fk_names(bind, table: str) -> set[str]:
    return {fk["name"] for fk in inspect(bind).get_foreign_keys(table) if fk.get("name")}


def _column_nullable(bind, table: str, column: str) -> bool:
    cols = {c["name"]: c for c in inspect(bind).get_columns(table)}
    return bool(cols.get(column, {}).get("nullable", True))


def upgrade() -> None:
    bind = op.get_bind()

    # --- 1. feedback.rating -> nullable (upstream baseline is NOT NULL) ---
    if "feedback" in inspect(bind).get_table_names() and _column_nullable(bind, "feedback", "rating"):
        # Already nullable (fresh DB built from our ORM, or already migrated) — no-op.
        pass
    else:
        with op.batch_alter_table("feedback", schema=None) as batch:
            batch.alter_column("rating", existing_type=sa.Integer(), nullable=True)

    # --- 2. Noldus feedback columns (idempotent via safe_add_column) ---
    safe_add_column("feedback", sa.Column("verdict", sa.String(length=16), nullable=True))
    safe_add_column("feedback", sa.Column("revised_text", sa.Text(), nullable=True))
    safe_add_column("feedback", sa.Column("paradigm", sa.String(length=64), nullable=True))

    # --- 3. runs / run_events -> threads_meta ON DELETE CASCADE FK ---
    if "runs" in inspect(bind).get_table_names() and _RUNS_THREAD_FK not in _existing_fk_names(bind, "runs"):
        with op.batch_alter_table("runs", schema=None) as batch:
            batch.create_foreign_key(
                _RUNS_THREAD_FK,
                "threads_meta",
                ["thread_id"],
                ["thread_id"],
                ondelete="CASCADE",
            )

    if "run_events" in inspect(bind).get_table_names():
        existing = _existing_fk_names(bind, "run_events")
        need_thread = _EVENTS_THREAD_FK not in existing
        need_run = _EVENTS_RUN_FK not in existing
        if need_thread or need_run:
            with op.batch_alter_table("run_events", schema=None) as batch:
                if need_thread:
                    batch.create_foreign_key(
                        _EVENTS_THREAD_FK,
                        "threads_meta",
                        ["thread_id"],
                        ["thread_id"],
                        ondelete="CASCADE",
                    )
                if need_run:
                    batch.create_foreign_key(
                        _EVENTS_RUN_FK,
                        "runs",
                        ["run_id"],
                        ["run_id"],
                        ondelete="CASCADE",
                    )


def downgrade() -> None:
    bind = op.get_bind()

    if "run_events" in inspect(bind).get_table_names():
        existing = _existing_fk_names(bind, "run_events")
        with op.batch_alter_table("run_events", schema=None) as batch:
            if _EVENTS_RUN_FK in existing:
                batch.drop_constraint(_EVENTS_RUN_FK, type_="foreignkey")
            if _EVENTS_THREAD_FK in existing:
                batch.drop_constraint(_EVENTS_THREAD_FK, type_="foreignkey")

    if "runs" in inspect(bind).get_table_names() and _RUNS_THREAD_FK in _existing_fk_names(bind, "runs"):
        with op.batch_alter_table("runs", schema=None) as batch:
            batch.drop_constraint(_RUNS_THREAD_FK, type_="foreignkey")

    # Drop the Noldus feedback columns (best-effort; skip if already gone).
    from deerflow.persistence.migrations._helpers import safe_drop_column

    safe_drop_column("feedback", "paradigm")
    safe_drop_column("feedback", "revised_text")
    safe_drop_column("feedback", "verdict")

    # Restore feedback.rating NOT NULL (legacy DBs that came from upstream baseline).
    if "feedback" in inspect(bind).get_table_names() and not _column_nullable(bind, "feedback", "rating"):
        pass
    else:
        with op.batch_alter_table("feedback", schema=None) as batch:
            batch.alter_column("rating", existing_type=sa.Integer(), nullable=False)
