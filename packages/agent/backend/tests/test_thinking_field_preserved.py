"""复现 Issue #2: messages[N]: missing field 'thinking' 400 错误。

thread 5046a6e6 实际触发链路：
1. 用户问"时间分段分析"
2. lead 自己 write_file 写脚本（触发 Issue #1 reload）
3. 重启后从 checkpoint 恢复历史 messages
4. 历史 message 中某条 AIMessage 的 thinking content block 丢失
5. langchain_anthropic 发请求时校验失败 → newapi 返 400

本测试模拟链路中"消息含 thinking content block → 走过某个中间件 → 校验仍含 thinking"。
诊断时先跑这个测试，看哪条链路最先 fail。Task 4 修复对应的中间件后此测试转 PASS。

诊断结论 (2026-05-14):
    根本原因在 langchain_anthropic 的 streaming event handler 中，
    signature_delta 事件创建的 AIMessageChunk 有 type="thinking" 但没有 thinking 键。
    当合并后的消息通过 checkpointer 序列化再反序列化后作为历史发送给模型时，
    _format_messages 过滤这个块只保留 thinking/signature 键 → 缺少 thinking 键 → 400。

    修复 (Task 4): 在 ClaudeChatModel._get_request_payload 中调用
    _strip_malformed_thinking_blocks() 剔除 type="thinking" 但缺 thinking 键的块。
"""

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def ai_message_with_thinking() -> AIMessage:
    """A canonical AIMessage with a thinking content block.

    Mirrors what deepseek-v4-pro returns via newapi when thinking_enabled=True.
    """
    return AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me reason about this: ..."},
            {"type": "text", "text": "The answer is 42."},
        ],
        additional_kwargs={"reasoning_content": "Let me reason about this: ..."},
    )


def _has_thinking_block(msg: AIMessage) -> bool:
    """Check if msg.content has at least one type=thinking block."""
    if not isinstance(msg.content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "thinking" for b in msg.content
    )


class TestThinkTagMiddlewarePreservesThinking:
    """think_tag_middleware 把内联 <think> 标签搬到 reasoning_content 时
    应保留 type=thinking 的 content block（不应删除）。
    """

    def test_strip_think_tags_exists_and_callable(self) -> None:
        from deerflow.agents.middlewares.think_tag_middleware import strip_think_tags

        assert callable(strip_think_tags)

    def test_thinking_block_survives_rewrite_with_list_content(
        self, ai_message_with_thinking: AIMessage
    ) -> None:
        """ThinkTagMiddleware._rewrite 看到 list-content 消息时，
        type=thinking block 应原样保留（不被当成 text block 处理）。
        """
        from deerflow.agents.middlewares.think_tag_middleware import ThinkTagMiddleware

        mw = ThinkTagMiddleware()
        msg = ai_message_with_thinking
        assert _has_thinking_block(msg), "fixture 必须含 thinking block（前提）"

        rewritten = mw._rewrite(msg)
        # _rewrite returns None if no changes needed (no <think> tags in text blocks)
        # In that case, the original message is unchanged by _apply
        if rewritten is not None:
            assert _has_thinking_block(rewritten), (
                "_rewrite 返回的 AIMessage 必须仍含 thinking block"
            )
        # If None, the original message is untouched (idempotent no-op)


class TestSummarizationPartitionPreservesThinking:
    """_partition_messages 是简单 list slicing，不应修改消息内容。"""

    def test_partition_does_not_mutate_content(
        self, ai_message_with_thinking: AIMessage
    ) -> None:
        """分区后 preserved 消息的 thinking block 应完好无损。"""
        from unittest.mock import MagicMock

        from deerflow.agents.middlewares.archiving_summarization import (
            ArchivingSummarizationMiddleware,
        )

        mw = ArchivingSummarizationMiddleware(model=MagicMock())
        messages = [
            AIMessage(content="msg 1"),
            AIMessage(content="msg 2"),
            AIMessage(content="msg 3"),
            AIMessage(content="msg 4"),
            ai_message_with_thinking,      # index 4
            AIMessage(content="msg 6"),     # index 5
            AIMessage(content="msg 7"),     # index 6
            AIMessage(content="msg 8"),     # index 7
        ]

        # cutoff_index=5: messages 0-4 to archive, 5-7 preserved
        to_summarize, preserved = mw._partition_messages(messages, cutoff_index=5)

        # to_summarize has 5 items (indices 0-4), the fixture is at index 4
        assert len(to_summarize) == 5
        assert to_summarize[4] is ai_message_with_thinking
        assert _has_thinking_block(to_summarize[4]), (
            "归档区的消息也必须仍含 thinking block（归档用 messages_to_dict 序列化，块应保留）"
        )

        # preserved has 3 items (indices 5-7)
        assert len(preserved) == 3
        assert preserved[0].content == "msg 6"


class TestCheckpointerRoundTrip:
    """Checkpointer 通过 JsonPlusSerializer (ormsgpack) 序列化 Pydantic v2 对象。
    对于 list-of-dict 的 content，序列化-反序列化应保留所有键值。
    """

    def test_model_dump_and_reconstruct_preserves_thinking(
        self, ai_message_with_thinking: AIMessage
    ) -> None:
        """AIMessage.model_dump() + AIMessage(**data) round-trip retains thinking block."""
        data = ai_message_with_thinking.model_dump()
        restored = AIMessage(**data)
        assert _has_thinking_block(restored), (
            "model_dump → AIMessage(**data) 应保留 thinking block"
        )

    def test_messages_to_dict_round_trip(self, ai_message_with_thinking: AIMessage) -> None:
        """messages_to_dict + messages_from_dict 序列化往返后 thinking block 应在。"""
        from langchain_core.messages import messages_from_dict, messages_to_dict

        dicts = messages_to_dict([ai_message_with_thinking])
        restored = messages_from_dict(dicts)
        assert len(restored) == 1
        assert _has_thinking_block(restored[0]), (
            "messages_to_dict → messages_from_dict 应保留 thinking block"
        )


class TestAnthropicFormatMessages:
    """验证 langchain_anthropic._format_messages 对 thinking block 的处理。

    关键发现：_format_messages 在 line 569-577 处过滤 thinking block 的键，
    只保留 type/thinking/cache_control/signature。如果原始块缺失 thinking 键，
    输出会是 {"type": "thinking"} 或 {"type": "thinking", "signature": "..."}，
    导致 newapi 400 错误。
    """

    def test_format_messages_preserves_thinking_block(self) -> None:
        """type=thinking 且 thinking 键存在的块应被 _format_messages 正确保留。"""
        try:
            from langchain_anthropic.chat_models import _format_messages
        except ImportError:
            pytest.skip("langchain_anthropic 未安装")

        msg = AIMessage(
            content=[{"type": "thinking", "thinking": "reasoning..."}, {"type": "text", "text": "answer"}],
            response_metadata={"model_provider": "anthropic"},
        )

        system, formatted = _format_messages([msg])
        assert system is None
        assert len(formatted) == 1
        assert "content" in formatted[0]
        content = formatted[0]["content"]
        assert isinstance(content, list)

        thinking_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
        assert len(thinking_blocks) >= 1, "应有至少 1 个 thinking block"
        for block in thinking_blocks:
            assert "thinking" in block, (
                f"每个 thinking block 必须有 thinking 键，got keys={list(block.keys())}"
            )

    def test_signature_delta_without_thinking_creates_malformed_block(self) -> None:
        """复现 + 修复验证：signature_delta 产生 type=thinking 但无 thinking 键的块。

        在 langchain_anthropic streaming handler 中（line 2071-2075），
        signature_delta 事件创建 content_block 后强制 type="thinking"，
        但 signature_delta 的 model_dump 不含 thinking 键，
        导致合并后的消息中包含 type=thinking 但缺 thinking 键的块。

        本测试验证两步：
        1. _format_messages 确实会产生 malformed block（上游 bug）
        2. _strip_malformed_thinking_blocks 修复能正确剔除 malformed block（Task 4 fix）
        """
        try:
            from langchain_anthropic.chat_models import _format_messages
        except ImportError:
            pytest.skip("langchain_anthropic 未安装")

        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        # 模拟 signature_delta 产生的不完整 thinking block
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "real reasoning..."},
                # signature_delta 创建的块：有 type=thinking + signature 但无 thinking 键
                {"type": "thinking", "signature": "EqQBCgIY..."},
                {"type": "text", "text": "answer"},
            ],
            response_metadata={"model_provider": "anthropic"},
        )

        system, formatted = _format_messages([msg])

        content = formatted[0]["content"]
        thinking_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
        assert len(thinking_blocks) == 2, "应有 2 个 thinking block（1 正常 + 1 signature-only）"

        # 验证上游 bug 存在：signature-only block 缺少 thinking 键
        malformed = [b for b in thinking_blocks if "thinking" not in b]
        assert len(malformed) == 1, "应有 1 个 malformed block（上游 bug 存在）"
        assert malformed[0].get("signature") == "EqQBCgIY..."

        # 应用 Task 4 修复
        _strip_malformed_thinking_blocks(formatted)

        # 修复后：malformed block 应被移除，valid thinking block 应保留
        content_after = formatted[0]["content"]
        thinking_after = [b for b in content_after if isinstance(b, dict) and b.get("type") == "thinking"]
        assert len(thinking_after) == 1, (
            f"修复后应有 1 个 thinking block（signature-only 应被移除），got {len(thinking_after)}"
        )
        assert "thinking" in thinking_after[0], "保留的 thinking block 必须有 thinking 键"
        assert thinking_after[0]["thinking"] == "real reasoning..."

        text_after = [b for b in content_after if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_after) == 1, "text block 应保留"


class TestStripMalformedThinkingBlocks:
    """验证 _strip_malformed_thinking_blocks 修复逻辑。

    Task 4 修复：在 ClaudeChatModel._get_request_payload 中调用此函数，
    剔除 type="thinking" 但缺少 thinking 键的 malformed blocks（signature_delta artifact）。
    """

    def test_strips_signature_only_block(self) -> None:
        """type=thinking 但无 thinking 键的块应被移除。"""
        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "valid reasoning..."},
                    # signature_delta artifact — must be stripped
                    {"type": "thinking", "signature": "EqQBCgIY..."},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]

        _strip_malformed_thinking_blocks(messages)

        content = messages[0]["content"]
        assert len(content) == 2, f"应有 2 个块（valid thinking + text），got {len(content)}"
        assert content[0] == {"type": "thinking", "thinking": "valid reasoning..."}, "valid thinking block 应保留"
        assert content[1] == {"type": "text", "text": "answer"}, "text block 应保留"

    def test_preserves_valid_thinking_block(self) -> None:
        """type=thinking 且有 thinking 键的块应完整保留。"""
        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]

        _strip_malformed_thinking_blocks(messages)

        content = messages[0]["content"]
        assert len(content) == 2, f"应有 2 个块，got {len(content)}"
        assert content[0] == {"type": "thinking", "thinking": "reasoning", "signature": "sig"}

    def test_no_thinking_blocks_unaffected(self) -> None:
        """没有 thinking block 的消息不应被修改。"""
        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]

        _strip_malformed_thinking_blocks(messages)

        assert messages[0]["content"] == "hello"
        assert messages[1]["content"] == [{"type": "text", "text": "hi"}]

    def test_empty_content_unaffected(self) -> None:
        """空 content 不应崩溃。"""
        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        messages = [
            {"role": "assistant", "content": []},
            {"role": "assistant", "content": ""},
        ]

        _strip_malformed_thinking_blocks(messages)

        assert messages[0]["content"] == []
        assert messages[1]["content"] == ""

    def test_all_malformed_thinking_blocks_stripped(self) -> None:
        """如果消息只有 malformed thinking blocks，应全部移除。"""
        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "signature": "sig1"},
                    {"type": "thinking", "signature": "sig2"},
                ],
            }
        ]

        _strip_malformed_thinking_blocks(messages)

        content = messages[0]["content"]
        assert len(content) == 0, f"所有 malformed blocks 都应移除，got {content}"

    def test_end_to_end_simulated_round_trip(self) -> None:
        """端到端模拟：signature_delta malformed block → checkpointer → fix → clean。

        模拟完整链路：
        1. streaming 产生带 malformed thinking block 的 AIMessage content
        2. checkpointer 序列化后作为历史发送
        3. _strip_malformed_thinking_blocks 清理
        4. 验证清理后的消息没有 malformed blocks
        """
        try:
            from langchain_anthropic.chat_models import _format_messages
        except ImportError:
            pytest.skip("langchain_anthropic 未安装")

        from deerflow.models.claude_provider import _strip_malformed_thinking_blocks

        # Step 1: 模拟 streaming 产生的消息（含 signature_delta artifact）
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "real reasoning..."},
                {"type": "thinking", "signature": "EqQBCgIY..."},  # malformed
                {"type": "text", "text": "answer"},
            ],
            response_metadata={"model_provider": "anthropic"},
        )

        # Step 2: 模拟 _format_messages（checkpointer 恢复后作为历史发送时经过的格式化）
        system, formatted = _format_messages([msg])

        # Step 3: 应用修复 — 剔除 malformed thinking blocks
        _strip_malformed_thinking_blocks(formatted)

        # Step 4: 验证
        content = formatted[0]["content"]
        thinking_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1, (
            f"修复后应有 1 个 thinking block（signature-only 应被移除），got {len(thinking_blocks)}"
        )
        assert "thinking" in thinking_blocks[0], "保留的 thinking block 必须有 thinking 键"
        assert thinking_blocks[0]["thinking"] == "real reasoning..."

        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_blocks) == 1, "text block 应保留"
        assert text_blocks[0]["text"] == "answer"
