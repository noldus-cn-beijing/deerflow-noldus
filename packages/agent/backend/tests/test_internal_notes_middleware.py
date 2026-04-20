"""Tests for InternalNotesMiddleware + summary message tagging.

Commit 6a — hide the lead agent's internal orchestration status notes
(e.g. "## 提取的关键上下文") and the SummarizationMiddleware's injected
"Here is a summary of the conversation to date:" HumanMessage from the
frontend. Both stay in LangGraph state so the model can still see them;
only `additional_kwargs.hide_from_ui = True` is set, which the frontend's
existing `isHiddenFromUIMessage` already respects.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.internal_notes_middleware import (
    InternalNotesMiddleware,
    _is_internal_notes_ai_message,
)

INTERNAL_DUMP_CONTENT = """## 提取的关键上下文

### 任务概述
用户完成了斑马鱼鱼群行为测试，请求分析 EthoVision 轨迹数据。

### 当前执行状态（Todo）
1. ✅ 规划完成
2. 🔄 进行中：派遣 code-executor
"""


USER_FACING_SYNTHESIS_CONTENT = """### 分析结果

本次斑马鱼群体行为（Shoaling）实验共分析了 5 条鱼...

### 关键指标

| 指标 | 对照组 | 实验组 |
|------|--------|--------|
| 移动距离 | 24329 | 21053 |

### 关键洞察

- NND 中等效应量值得关注
"""


class TestIsInternalNotesAIMessage:
    def test_matches_chinese_extraction_heading(self):
        msg = AIMessage(content=INTERNAL_DUMP_CONTENT)
        assert _is_internal_notes_ai_message(msg) is True

    def test_matches_english_extracted_context_heading(self):
        msg = AIMessage(
            content="## Extracted Context\n\n**Task**: Analyze zebrafish data..."
        )
        assert _is_internal_notes_ai_message(msg) is True

    def test_matches_after_leading_whitespace(self):
        msg = AIMessage(content="\n  \n## 提取的关键上下文\n\n### 任务概述\n...")
        assert _is_internal_notes_ai_message(msg) is True

    def test_does_not_match_user_facing_synthesis(self):
        """### 分析结果 starts with h3, not h2, and uses a different phrase."""
        msg = AIMessage(content=USER_FACING_SYNTHESIS_CONTENT)
        assert _is_internal_notes_ai_message(msg) is False

    def test_does_not_match_casual_markdown_response(self):
        msg = AIMessage(content="## 关于 IID 指标\n\nIID 是衡量群体间距的指标...")
        assert _is_internal_notes_ai_message(msg) is False

    def test_does_not_match_ai_message_with_tool_calls(self):
        """Even if content looks like notes, a tool-call turn is not notes."""
        msg = AIMessage(
            content=INTERNAL_DUMP_CONTENT,
            tool_calls=[{"name": "write_file", "args": {}, "id": "call_1"}],
        )
        assert _is_internal_notes_ai_message(msg) is False

    def test_does_not_match_human_message(self):
        msg = HumanMessage(content=INTERNAL_DUMP_CONTENT)
        assert _is_internal_notes_ai_message(msg) is False

    def test_does_not_match_tool_message(self):
        msg = ToolMessage(content=INTERNAL_DUMP_CONTENT, tool_call_id="call_1")
        assert _is_internal_notes_ai_message(msg) is False

    def test_does_not_match_list_content(self):
        """AIMessage content may be a list of blocks (multimodal); we only tag string content."""
        msg = AIMessage(content=[{"type": "text", "text": INTERNAL_DUMP_CONTENT}])
        assert _is_internal_notes_ai_message(msg) is False


class TestInternalNotesMiddlewareAfterModel:
    @pytest.fixture
    def middleware(self) -> InternalNotesMiddleware:
        return InternalNotesMiddleware()

    def test_tags_last_message_when_internal_notes(self, middleware):
        internal_msg = AIMessage(id="ai-1", content=INTERNAL_DUMP_CONTENT)
        state = {"messages": [HumanMessage(content="分析数据"), internal_msg]}
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result is not None
        updated = result["messages"][0]
        assert updated.id == "ai-1"
        assert updated.additional_kwargs.get("hide_from_ui") is True

    def test_skips_when_last_message_is_user_facing_synthesis(self, middleware):
        synth_msg = AIMessage(id="ai-2", content=USER_FACING_SYNTHESIS_CONTENT)
        state = {"messages": [synth_msg]}
        assert middleware.after_model(state, runtime=None) is None  # type: ignore[arg-type]

    def test_preserves_existing_additional_kwargs(self, middleware):
        internal_msg = AIMessage(
            id="ai-3",
            content=INTERNAL_DUMP_CONTENT,
            additional_kwargs={"some_existing_flag": "keep-me"},
        )
        state = {"messages": [internal_msg]}
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result is not None
        updated = result["messages"][0]
        assert updated.additional_kwargs.get("some_existing_flag") == "keep-me"
        assert updated.additional_kwargs.get("hide_from_ui") is True

    def test_idempotent_when_already_tagged(self, middleware):
        """Re-running on an already-tagged message returns None (no state churn)."""
        internal_msg = AIMessage(
            id="ai-4",
            content=INTERNAL_DUMP_CONTENT,
            additional_kwargs={"hide_from_ui": True},
        )
        state = {"messages": [internal_msg]}
        assert middleware.after_model(state, runtime=None) is None  # type: ignore[arg-type]

    def test_empty_messages_returns_none(self, middleware):
        assert middleware.after_model({"messages": []}, runtime=None) is None  # type: ignore[arg-type]

    def test_leaves_message_in_state_for_model(self, middleware):
        """The tagged message must remain in state — it is hidden only from UI,
        not removed from the agent's context window."""
        internal_msg = AIMessage(id="ai-5", content=INTERNAL_DUMP_CONTENT)
        state = {"messages": [internal_msg]}
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result is not None
        # Still present, same id (LangGraph reducer replaces by id).
        assert result["messages"][0].id == "ai-5"
        # Content unchanged.
        assert result["messages"][0].content == INTERNAL_DUMP_CONTENT
