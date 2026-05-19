"""Feedback-issue router — public feedback form that creates GitHub issues.

Internal engineers (behavioral science / hardware sales / post-sales)
submit feedback through a simple web form. The form is public (no login
required) and creates a GitHub issue in the configured repo.

Requires GITHUB_TOKEN and GITHUB_REPO env vars to be set.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback-issue"])

_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "feedback.html"
_HTML_CONTENT: str | None = None

CategoryT = Literal["bug", "enhancement", "experience", "other"]

CATEGORY_LABELS: dict[CategoryT, str] = {
    "bug": "feedback,bug",
    "enhancement": "feedback,enhancement",
    "experience": "feedback,ux",
    "other": "feedback",
}

CATEGORY_NAMES: dict[CategoryT, str] = {
    "bug": "问题反馈",
    "enhancement": "功能建议",
    "experience": "使用体验",
    "other": "其他",
}


class FeedbackIssueRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    category: CategoryT
    description: str = Field(..., min_length=1, max_length=5000)
    name: str | None = Field(default=None, max_length=100)
    contact: str | None = Field(default=None, max_length=200)


class FeedbackIssueResponse(BaseModel):
    ok: bool
    issue_url: str | None = None
    message: str


def _load_html() -> str:
    global _HTML_CONTENT
    if _HTML_CONTENT is None:
        if not _HTML_PATH.exists():
            logger.warning("feedback.html not found at %s", _HTML_PATH)
            _HTML_CONTENT = "<html><body><h1>Feedback form not found</h1></body></html>"
        else:
            _HTML_CONTENT = _HTML_PATH.read_text(encoding="utf-8")
    return _HTML_CONTENT


def _build_issue_body(title: str, category: CategoryT, description: str, name: str | None, contact: str | None) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## 反馈类型",
        f"{CATEGORY_NAMES[category]}",
        "",
        f"## 详细描述",
        description,
        "",
        "---",
        f"**提交人**: {name or '未填写'}",
        f"**联系方式**: {contact or '未填写'}",
        f"**提交时间**: {now}",
    ]
    return "\n".join(lines)


@router.get("/api/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request) -> HTMLResponse:
    """Serve the feedback form HTML page (public, no auth)."""
    return HTMLResponse(content=_load_html())


@router.post("/api/feedback-issue", response_model=FeedbackIssueResponse)
async def submit_feedback_issue(body: FeedbackIssueRequest, request: Request) -> dict:
    """Create a GitHub issue from user feedback (public, no auth)."""
    github_token = os.getenv("GITHUB_TOKEN", "").strip()
    github_repo = os.getenv("GITHUB_REPO", "").strip()

    if not github_token:
        logger.error("GITHUB_TOKEN not configured")
        raise HTTPException(status_code=503, detail="反馈系统未配置（缺少 GITHUB_TOKEN），请联系管理员。")

    if not github_repo:
        logger.error("GITHUB_REPO not configured")
        raise HTTPException(status_code=503, detail="反馈系统未配置（缺少 GITHUB_REPO），请联系管理员。")

    if "/" not in github_repo:
        raise HTTPException(status_code=503, detail="GITHUB_REPO 格式错误，应为 owner/repo。")

    url = f"https://api.github.com/repos/{github_repo}/issues"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "EthoInsight-Feedback/1.0",
    }

    issue_title = f"[用户反馈] {body.title}"
    issue_body = _build_issue_body(body.title, body.category, body.description, body.name, body.contact)
    labels = CATEGORY_LABELS[body.category]

    payload = {
        "title": issue_title,
        "body": issue_body,
        "labels": labels.split(","),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 401:
            logger.error("GitHub API 401 — GITHUB_TOKEN may be invalid or expired")
            raise HTTPException(status_code=503, detail="GitHub 认证失败，请检查 GITHUB_TOKEN 配置。")

        if resp.status_code == 404:
            logger.error("GitHub API 404 — repo %s not found or token lacks access", github_repo)
            raise HTTPException(status_code=503, detail=f"仓库 {github_repo} 不存在或 Token 无权访问。")

        if resp.status_code >= 400:
            logger.error("GitHub API error %d: %s", resp.status_code, resp.text[:500])
            raise HTTPException(status_code=502, detail=f"GitHub API 返回错误 ({resp.status_code})，请稍后重试。")

        data = resp.json()
        issue_url = data.get("html_url", "")
        logger.info("Feedback issue created: %s", issue_url)

        return {"ok": True, "issue_url": issue_url, "message": "反馈提交成功！"}

    except httpx.TimeoutException:
        logger.error("GitHub API timeout")
        raise HTTPException(status_code=504, detail="GitHub API 超时，请稍后重试。")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error creating feedback issue")
        raise HTTPException(status_code=500, detail=f"提交失败: {exc}") from exc
