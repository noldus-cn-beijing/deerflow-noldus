"""ORM model for run events."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class RunEventRow(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # 外键级联（spec 2026-06-26 §任务1）：删 thread → cascade runs → cascade
    # run_events；删 run → cascade run_events。原本两列均无约束，删 thread/run 后
    # 事件行残留成孤儿。ON DELETE CASCADE 由 DB 兜底。
    thread_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("threads_meta.thread_id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Owner of the conversation this event belongs to. Nullable for data
    # created before auth was introduced; populated by auth middleware on
    # new writes and by the boot-time orphan migration on existing rows.
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    # "message" | "trace" | "lifecycle"
    content: Mapped[str] = mapped_column(Text, default="")
    event_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    seq: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("thread_id", "seq", name="uq_events_thread_seq"),
        Index("ix_events_thread_cat_seq", "thread_id", "category", "seq"),
        Index("ix_events_run", "thread_id", "run_id", "seq"),
    )
