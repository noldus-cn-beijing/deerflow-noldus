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
