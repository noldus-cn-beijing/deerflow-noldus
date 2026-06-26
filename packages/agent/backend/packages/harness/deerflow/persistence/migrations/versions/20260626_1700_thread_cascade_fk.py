"""runs/run_events → threads_meta ON DELETE CASCADE 外键

Revision ID: 20260626_1700
Revises: 20260622_1700
Create Date: 2026-06-26 17:00:00.000000

spec 2026-06-26-file-path-reliability-loadbearing-convergence-spec §任务1。
给 ``runs.thread_id`` 与 ``run_events.{thread_id, run_id}`` 加 ON DELETE CASCADE
外键，让删 thread 时 DB 确定性连带清掉 runs / run_events，消除「meta 删了、runs
残留指向不存在 thread」的孤儿（历史路径类 bug 的家族病根之一）。

DeerFlow 原本这三列均仅 ``nullable=False`` / ``index=True`` 无 FK。本迁移补约束：

- ``runs.thread_id → threads_meta.thread_id ON DELETE CASCADE``
- ``run_events.thread_id → threads_meta.thread_id ON DELETE CASCADE``
- ``run_events.run_id → runs.run_id ON DELETE CASCADE``

SQLite 建表无 ``REFERENCES`` 时，加外键必须 ``batch_alter_table`` 重建表
（SQLite 的 ALTER TABLE 不支持 ADD CONSTRAINT）。``batch_alter_table`` 在
检测到无法就地 ALTER 时自动走「临时表复制 + rename」流程，并把现有数据搬过去；
``PRAGMA foreign_keys=ON`` 由 ``engine.py`` 在每条连接上设置，故级联生效。

部署链不跑 alembic autogenerate；现网库需手动 ``alembic upgrade head``（守
memory ``feedback_deploy_alembic_migration_for_added_columns`` +
``feedback_local_dev_db_also_needs_manual_alembic_migration_after_sync``）。
``create_all`` 不会 ALTER 已存在表，故已建表的现网必须走本迁移；新建库则由
ORM 模型上的 ``ForeignKey(ondelete='CASCADE')`` 直接建成。

幂等：迁移前先 ``inspect`` 现有外键，已有同名约束则跳过（支持重复 upgrade，
例如 dev 库重置后重跑）。PostgreSQL 上 ``ADD CONSTRAINT`` 可就地完成。
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "20260626_1700"
down_revision = "20260622_1700"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    return name in inspect(bind).get_table_names()


def _existing_fk_names(bind, table: str) -> set[str]:
    """Return the set of foreign-key constraint names currently on ``table``."""
    return {fk["name"] for fk in inspect(bind).get_foreign_keys(table) if fk.get("name")}


# 目标外键约束名（downgrade 引用 + 幂等检查）。SQLite batch 重建时会按
# ``batch.create_foreign_key(name=...)`` 给约束命名；PostgreSQL 就地 ADD CONSTRAINT
# 同名。两后端统一用这套名字。
_RUNS_THREAD_FK = "fk_runs_thread_id_threads_meta"
_EVENTS_THREAD_FK = "fk_run_events_thread_id_threads_meta"
_EVENTS_RUN_FK = "fk_run_events_run_id_runs"


def upgrade() -> None:
    bind = op.get_bind()

    # --- runs.thread_id → threads_meta(thread_id) ON DELETE CASCADE ---
    if _has_table(bind, "runs") and _RUNS_THREAD_FK not in _existing_fk_names(bind, "runs"):
        with op.batch_alter_table("runs", schema=None) as batch:
            batch.create_foreign_key(
                _RUNS_THREAD_FK,
                "threads_meta",
                ["thread_id"],
                ["thread_id"],
                ondelete="CASCADE",
            )

    # --- run_events.{thread_id, run_id} 双外键级联 ---
    if _has_table(bind, "run_events"):
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

    if _has_table(bind, "run_events"):
        existing = _existing_fk_names(bind, "run_events")
        with op.batch_alter_table("run_events", schema=None) as batch:
            if _EVENTS_RUN_FK in existing:
                batch.drop_constraint(_EVENTS_RUN_FK, type_="foreignkey")
            if _EVENTS_THREAD_FK in existing:
                batch.drop_constraint(_EVENTS_THREAD_FK, type_="foreignkey")

    if _has_table(bind, "runs") and _RUNS_THREAD_FK in _existing_fk_names(bind, "runs"):
        with op.batch_alter_table("runs", schema=None) as batch:
            batch.drop_constraint(_RUNS_THREAD_FK, type_="foreignkey")
