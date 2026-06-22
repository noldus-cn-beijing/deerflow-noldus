"""DeerFlow sync e418d729 schema deltas: runs.token_usage_by_model + channel active-identity index

Revision ID: 20260622_1700
Revises: 20260601_1500
Create Date: 2026-06-22 17:00:00.000000

本次 sync 引入两处 ORM schema 变更。本仓库部署链不跑 alembic autogenerate，
且 ``Base.metadata.create_all`` 只建缺失的表、**绝不 ALTER 已存在的表**
（见 20260601_1500 feedback.paradigm 同款教训）。现网 runs / channel_connections
表早已由更早的 create_all 建好，故这两处变更必须显式迁移，否则现网静默缺列/缺索引。

1. runs.token_usage_by_model（上游 #3645，per-model token 明细）——**活跃路径**：
   RunJournal 完成时写、aggregate_tokens_by_thread 读，缺列则写入/聚合全部
   no-such-column 报错。

2. channel_connections.uq_channel_connection_active_identity（单活跃 owner 不变式
   的 partial unique index）——当前 channel SQL 层在本仓库无调用方（消费它的
   router 是尚未接入的上游代码），故此刻 inert；但同根因仍成立，随 channel 功能
   接入即触发。partial unique index 直接 CREATE 可能在已有重复活跃行的现网表上
   失败，故先按 (provider, external_account_id, workspace_id) 去重（每组保留
   created_at 最新一行，其余置 revoked）再建索引。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text

# revision identifiers, used by Alembic.
revision = "20260622_1700"
down_revision = "20260601_1500"
branch_labels = None
depends_on = None


_ACTIVE_IDENTITY_INDEX = "uq_channel_connection_active_identity"

# 去重：每个 (provider, external_account_id, workspace_id) 身份组里只保留最新的
# 一条非 revoked 行，其余置 revoked，以便随后能建 partial unique index。相关子查询
# 形式在 SQLite 与 PostgreSQL 上均合法。
_DEDUPE_ACTIVE_CONNECTIONS = text(
    """
    UPDATE channel_connections
    SET status = 'revoked'
    WHERE status != 'revoked'
      AND id NOT IN (
        SELECT keep.id FROM (
          SELECT c2.id AS id
          FROM channel_connections AS c2
          WHERE c2.status != 'revoked'
            AND c2.provider = channel_connections.provider
            AND c2.external_account_id = channel_connections.external_account_id
            AND c2.workspace_id = channel_connections.workspace_id
          ORDER BY c2.created_at DESC, c2.id DESC
          LIMIT 1
        ) AS keep
      )
    """
)


def _has_table(bind, name: str) -> bool:
    return name in inspect(bind).get_table_names()


def _has_index(bind, table: str, name: str) -> bool:
    return any(ix["name"] == name for ix in inspect(bind).get_indexes(table))


def upgrade() -> None:
    bind = op.get_bind()

    # --- 1. runs.token_usage_by_model（nullable，历史行留 NULL；读路径 or {} 兜底）---
    if _has_table(bind, "runs"):
        run_cols = {c["name"] for c in inspect(bind).get_columns("runs")}
        if "token_usage_by_model" not in run_cols:
            with op.batch_alter_table("runs", schema=None) as batch:
                batch.add_column(sa.Column("token_usage_by_model", sa.JSON(), nullable=True))

    # --- 2. channel_connections 单活跃身份 partial unique index ---
    if _has_table(bind, "channel_connections") and not _has_index(bind, "channel_connections", _ACTIVE_IDENTITY_INDEX):
        op.execute(_DEDUPE_ACTIVE_CONNECTIONS)
        op.create_index(
            _ACTIVE_IDENTITY_INDEX,
            "channel_connections",
            ["provider", "external_account_id", "workspace_id"],
            unique=True,
            sqlite_where=text("status != 'revoked'"),
            postgresql_where=text("status != 'revoked'"),
        )


def downgrade() -> None:
    bind = op.get_bind()

    if _has_table(bind, "channel_connections") and _has_index(bind, "channel_connections", _ACTIVE_IDENTITY_INDEX):
        op.drop_index(_ACTIVE_IDENTITY_INDEX, table_name="channel_connections")

    if _has_table(bind, "runs"):
        run_cols = {c["name"] for c in inspect(bind).get_columns("runs")}
        if "token_usage_by_model" in run_cols:
            with op.batch_alter_table("runs", schema=None) as batch:
                batch.drop_column("token_usage_by_model")
