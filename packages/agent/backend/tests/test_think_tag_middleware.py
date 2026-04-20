"""Tests for ThinkTagMiddleware and its helpers."""

import asyncio
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.think_tag_middleware import (
    ThinkTagMiddleware,
    extract_think_tags,
    split_think,
    strip_think_tags,
)


def _runtime():
    r = MagicMock()
    r.context = {"thread_id": "test"}
    return r


class TestStripThinkTags:
    def test_removes_single_block(self):
        assert strip_think_tags("<think>hidden</think>visible") == "visible"

    def test_removes_multiple_blocks(self):
        text = "<think>one</think>between<think>two</think>end"
        assert strip_think_tags(text) == "betweenend"

    def test_case_insensitive(self):
        assert strip_think_tags("<THINK>x</THINK>visible") == "visible"

    def test_strips_whitespace(self):
        assert strip_think_tags("  <think>x</think>  visible  ") == "visible"

    def test_no_tag_returns_trimmed(self):
        assert strip_think_tags("  hello  ") == "hello"


class TestExtractThinkTags:
    def test_extracts_inner_trimmed(self):
        assert extract_think_tags("<think>  reasoning here  </think>body") == ["reasoning here"]

    def test_extracts_multiple_in_order(self):
        text = "<think>one</think>mid<think>two</think>"
        assert extract_think_tags(text) == ["one", "two"]

    def test_skips_empty_blocks(self):
        assert extract_think_tags("<think>  </think>body") == []

    def test_no_tags_returns_empty(self):
        assert extract_think_tags("plain text") == []


class TestSplitThink:
    def test_returns_none_when_no_tag(self):
        body, reasoning = split_think("just body")
        assert body == "just body"
        assert reasoning is None

    def test_splits_single(self):
        body, reasoning = split_think("<think>think</think>body")
        assert body == "body"
        assert reasoning == "think"

    def test_joins_multiple_with_blank_line(self):
        body, reasoning = split_think("<think>a</think>X<think>b</think>Y")
        assert body == "XY"
        assert reasoning == "a\n\nb"


class TestThinkTagMiddlewareStringContent:
    def test_moves_think_content_to_reasoning(self):
        mw = ThinkTagMiddleware()
        msg = AIMessage(content="<think>私下推理</think>最终答案")
        state = {"messages": [msg]}

        result = mw._apply(state)

        assert result is not None
        new_msg = result["messages"][0]
        assert new_msg.content == "最终答案"
        assert new_msg.additional_kwargs["reasoning_content"] == "私下推理"

    def test_appends_to_existing_reasoning(self):
        mw = ThinkTagMiddleware()
        msg = AIMessage(
            content="<think>new</think>body",
            additional_kwargs={"reasoning_content": "old"},
        )
        state = {"messages": [msg]}

        result = mw._apply(state)

        assert result["messages"][0].additional_kwargs["reasoning_content"] == "old\n\nnew"

    def test_noop_when_no_think_tag(self):
        mw = ThinkTagMiddleware()
        msg = AIMessage(content="no thinking here")
        state = {"messages": [msg]}

        assert mw._apply(state) is None

    def test_ignores_human_messages(self):
        mw = ThinkTagMiddleware()
        state = {"messages": [HumanMessage(content="<think>ignored</think>stays")]}

        assert mw._apply(state) is None

    def test_rewrites_only_newest_ai_message(self):
        mw = ThinkTagMiddleware()
        old = AIMessage(content="<think>old</think>old body")
        new = AIMessage(content="<think>new</think>new body")
        state = {"messages": [old, new]}

        result = mw._apply(state)

        # The older AIMessage is left alone; only the newest is rewritten
        # (avoids re-writing history on every turn).
        assert result["messages"][0] is old
        assert result["messages"][1].content == "new body"

    def test_empty_messages(self):
        mw = ThinkTagMiddleware()
        assert mw._apply({"messages": []}) is None


class TestThinkTagMiddlewareListContent:
    def test_rewrites_text_parts(self):
        mw = ThinkTagMiddleware()
        content = [
            {"type": "text", "text": "<think>t1</think>body1"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text", "text": "plain body2"},
        ]
        msg = AIMessage(content=content)

        result = mw._apply({"messages": [msg]})

        assert result is not None
        new_content = result["messages"][0].content
        assert new_content[0] == {"type": "text", "text": "body1"}
        assert new_content[1] == {"type": "image_url", "image_url": {"url": "http://x"}}
        assert new_content[2] == {"type": "text", "text": "plain body2"}
        assert result["messages"][0].additional_kwargs["reasoning_content"] == "t1"

    def test_drops_empty_text_part_after_strip(self):
        mw = ThinkTagMiddleware()
        content = [
            {"type": "text", "text": "<think>only reasoning</think>"},
            {"type": "text", "text": "keep me"},
        ]
        msg = AIMessage(content=content)

        result = mw._apply({"messages": [msg]})

        # The first part became empty after stripping; it's dropped entirely.
        new_content = result["messages"][0].content
        assert new_content == [{"type": "text", "text": "keep me"}]
        assert result["messages"][0].additional_kwargs["reasoning_content"] == "only reasoning"

    def test_noop_list_without_think(self):
        mw = ThinkTagMiddleware()
        content = [{"type": "text", "text": "plain"}]
        msg = AIMessage(content=content)

        assert mw._apply({"messages": [msg]}) is None


class TestThinkTagMiddlewareAfterModel:
    def test_after_model_delegates_to_apply(self):
        mw = ThinkTagMiddleware()
        msg = AIMessage(content="<think>r</think>body")
        state = {"messages": [msg]}

        result = mw.after_model(state, _runtime())

        assert result is not None
        assert result["messages"][0].content == "body"

    def test_aafter_model_delegates_to_apply(self):
        mw = ThinkTagMiddleware()
        msg = AIMessage(content="<think>r</think>body")
        state = {"messages": [msg]}

        async def _run() -> dict | None:
            return await mw.aafter_model(state, _runtime())

        result = asyncio.run(_run())

        assert result is not None
        assert result["messages"][0].content == "body"
