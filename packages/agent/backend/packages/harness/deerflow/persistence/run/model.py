"""ORM model for run metadata."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 外键级联（spec 2026-06-26 §任务1）：删 threads_meta 行 → 连带删该 thread 的
    # 所有 runs 行。DeerFlow 原本仅 index=True 无约束，删 thread 后 runs 残留指向
    # 不存在的 thread_id（孤儿）。ON DELETE CASCADE 由 DB 确定性兜底，应用层三步
    # 删除顺序失败时不再留孤儿。SQLite 需 PRAGMA foreign_keys=ON（engine.py 已设）。
    thread_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("threads_meta.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assistant_id: Mapped[str | None] = mapped_column(String(128))
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # "pending" | "running" | "success" | "error" | "timeout" | "interrupted"

    model_name: Mapped[str | None] = mapped_column(String(128))
    multitask_strategy: Mapped[str] = mapped_column(String(20), default="reject")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    kwargs_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)

    # Convenience fields (for listing pages without querying RunEventStore)
    message_count: Mapped[int] = mapped_column(default=0)
    first_human_message: Mapped[str | None] = mapped_column(Text)
    last_ai_message: Mapped[str | None] = mapped_column(Text)

    # Token usage (accumulated in-memory by RunJournal, written on run completion)
    total_input_tokens: Mapped[int] = mapped_column(default=0)
    total_output_tokens: Mapped[int] = mapped_column(default=0)
    total_tokens: Mapped[int] = mapped_column(default=0)
    llm_call_count: Mapped[int] = mapped_column(default=0)
    lead_agent_tokens: Mapped[int] = mapped_column(default=0)
    subagent_tokens: Mapped[int] = mapped_column(default=0)
    middleware_tokens: Mapped[int] = mapped_column(default=0)
    # Per-model token breakdown. nullable=True so the column can be ALTER-added
    # onto the pre-existing prod runs table (migration 20260622_1700; create_all
    # never alters existing tables). Legacy rows read back NULL; the read path in
    # run/sql.py coalesces with ``or {}`` and writes always pass a dict.
    token_usage_by_model: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Follow-up association
    follow_up_to_run_id: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    __table_args__ = (Index("ix_runs_thread_status", "thread_id", "status"),)
