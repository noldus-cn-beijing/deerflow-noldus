"""Feedback router for training-data flywheel.

Accepts expert ✅/⚠️/❌ verdicts on assistant messages and appends each
feedback to ``.deer-flow/training-data/feedback/<thread_id>.jsonl``.

Robustness contract: disk errors are logged and surfaced as 500, never
allowed to crash the Gateway process.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threads", tags=["feedback"])


class FeedbackRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    verdict: Literal["correct", "needs_fix", "wrong"]
    revised_text: str | None = None
    note: str | None = None


class FeedbackResponse(BaseModel):
    success: bool


def _base_dir() -> Path:
    """Return path to backend/.deer-flow. Overridden in tests."""
    from deerflow.config.paths import get_paths

    return Path(get_paths().base_dir)


@router.post("/{thread_id}/feedback", response_model=FeedbackResponse)
def post_feedback(thread_id: str, req: FeedbackRequest) -> FeedbackResponse:
    try:
        out_dir = _base_dir() / "training-data" / "feedback"
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "thread_id": thread_id,
            "message_id": req.message_id,
            "verdict": req.verdict,
            "revised_text": req.revised_text,
            "note": req.note,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        path = out_dir / f"{thread_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return FeedbackResponse(success=True)
    except OSError as exc:
        logger.error("Feedback write failed for thread %s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail="Failed to persist feedback")
