"""feedback paradigm column (Sprint 8)

Revision ID: 20260601_1500
Revises: 20260512_1200
Create Date: 2026-06-01 15:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260601_1500"
down_revision = "20260512_1200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """加 paradigm 字段（nullable，历史行留 null）。

    Sprint 8 按范式检索历史纠正记录需要这一列。现网已有 feedback 表，
    create_all 不会补列，必须显式 ALTER（SQLite 走 batch_alter_table）。
    """
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.add_column(sa.Column("paradigm", sa.String(64), nullable=True))


def downgrade() -> None:
    """回滚：删除 paradigm 列。"""
    with op.batch_alter_table("feedback", schema=None) as batch:
        batch.drop_column("paradigm")
