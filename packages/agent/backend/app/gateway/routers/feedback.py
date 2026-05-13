"""Feedback router — Noldus verdict-based feedback for SFT 训练数据飞轮.

Backed by FeedbackRepository (SQLite). URL aligned with upstream:
POST/GET /api/threads/{thread_id}/runs/{run_id}/feedback

Noldus 语义：verdict ∈ {correct, needs_fix, wrong} + 可选 revised_text 作为
SFT 训练种子。verdict 自动映射到上游 rating 字段，使上游 thumbs-up/down
aggregate_by_run 统计仍可用。
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_current_user, get_feedback_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/threads", tags=["feedback"])

VerdictT = Literal["correct", "needs_fix", "wrong"]
_VERDICT_TO_RATING: dict[str, int | None] = {
    "correct": 1,
    "wrong": -1,
    "needs_fix": None,
}


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    verdict: VerdictT
    revised_text: str | None = None
    note: str | None = None


class FeedbackItem(BaseModel):
    feedback_id: str
    thread_id: str
    run_id: str
    user_id: str | None
    message_id: str | None
    verdict: VerdictT | None
    revised_text: str | None
    note: str | None
    created_at: str


def _to_item(record: dict[str, Any]) -> FeedbackItem:
    return FeedbackItem(
        feedback_id=record["feedback_id"],
        thread_id=record["thread_id"],
        run_id=record["run_id"],
        user_id=record.get("user_id"),
        message_id=record.get("message_id"),
        verdict=record.get("verdict"),
        revised_text=record.get("revised_text"),
        note=record.get("comment"),
        created_at=record["created_at"],
    )


@router.post(
    "/{thread_id}/runs/{run_id}/feedback",
    response_model=FeedbackItem,
)
@require_permission("threads", "write", owner_check=True)
async def submit_feedback(
    thread_id: str,
    run_id: str,
    body: FeedbackRequest,
    request: Request,
) -> dict[str, Any]:
    """Submit Noldus verdict-based feedback for a specific message in a run.

    Note: 不校验 run_id 是否存在于 run_store —— 当前架构下没有任何 producer
    往 run_store 写入（LangGraph 模式 run 由 langgraph 自己管），导致校验
    永远 404。feedback 表是写给 SFT 飞轮用的，verdict + revised_text 才是
    核心信息，少量脏 run_id 不影响下游训练数据筛选。

    Note: 装饰器用 owner_check=True 但 **不用 require_existing=True**。
    threads_meta 表在 LangGraph 直连模式下不会被写入（前端通过
    langgraph-sdk 写 checkpointer.db、跳过 Gateway 的 POST /api/threads
    路由），导致 require_existing=True 会因 row missing 永远抛 404。
    feedback 是审计 / 训练数据写入，不是 destructive 操作，宽松校验是
    正确语义；与 GET list_run_feedback 装饰器对齐。
    """
    user_id = await get_current_user(request)
    feedback_repo = get_feedback_repo(request)
    record = await feedback_repo.upsert(
        thread_id=thread_id,
        run_id=run_id,
        user_id=user_id,
        message_id=body.message_id,
        verdict=body.verdict,
        rating=_VERDICT_TO_RATING[body.verdict],
        revised_text=body.revised_text,
        comment=body.note,
    )
    return _to_item(record).model_dump()


@router.get(
    "/{thread_id}/runs/{run_id}/feedback",
    response_model=list[FeedbackItem],
)
@require_permission("threads", "read", owner_check=True)
async def list_run_feedback(
    thread_id: str,
    run_id: str,
    request: Request,
) -> list[dict[str, Any]]:
    """List current user's feedback for a run."""
    feedback_repo = get_feedback_repo(request)
    user_id = await get_current_user(request)
    rows = await feedback_repo.list_by_run(thread_id, run_id, user_id=user_id)
    return [_to_item(r).model_dump() for r in rows]
