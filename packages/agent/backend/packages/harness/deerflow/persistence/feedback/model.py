"""ORM model for user feedback on runs."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class FeedbackRow(Base):
    __tablename__ = "feedback"

    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "run_id",
            "user_id",
            "message_id",
            name="uq_feedback_thread_run_user_message",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    message_id: Mapped[str | None] = mapped_column(String(64))
    # message_id is an optional RunEventStore event identifier —
    # allows feedback to target a specific message or the entire run.
    # Noldus extension: 进入 unique constraint，使同 run 多 message 反馈互不覆盖。

    # Noldus 业务路径不写 rating；上游 thumbs-up/down 路径仍写。
    rating: Mapped[int | None] = mapped_column(nullable=True)

    comment: Mapped[str | None] = mapped_column(Text)

    # Noldus 扩展：三分类 verdict + 专家修订版本（SFT 训练种子）
    verdict: Mapped[str | None] = mapped_column(String(16))
    revised_text: Mapped[str | None] = mapped_column(Text)

    # Sprint 8: 范式标识，用于按范式检索历史纠正（从 experiment-context.json 读取）
    paradigm: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
