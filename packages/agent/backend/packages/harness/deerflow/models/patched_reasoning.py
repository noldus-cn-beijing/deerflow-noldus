"""Patched ChatOpenAI that preserves ``reasoning_content`` from DeepSeek-compatible APIs.

DashScope / DeepSeek APIs return a ``reasoning_content`` field in the streaming
delta that the standard ``langchain_openai.ChatOpenAI`` adapter silently drops.
This module patches ChatOpenAI to route that field into
``additional_kwargs.reasoning_content`` so the frontend can display it via the
existing reasoning panel (ThinkTagMiddleware is not needed when reasoning
arrives as a separate field).

This is a lightweight alternative to ``langchain_deepseek.ChatDeepSeek`` when
the API endpoint is OpenAI-compatible and only needs ``reasoning_content``
preserved — no extra API-specific URL construction or multi-turn logic.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _merge_reasoning_content(existing: Any, new: str | None) -> str | None:
    """Append ``new`` reasoning to an existing value (if any), separated by ``\\n\\n``."""
    if not new:
        return existing if isinstance(existing, str) and existing else None
    if isinstance(existing, str) and existing:
        return existing + "\n\n" + new
    return new


class PatchedReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that preserves ``reasoning_content`` in streaming deltas.

    For OpenAI-compatible APIs that return ``reasoning_content`` alongside
    ``content`` in the delta (DeepSeek, DashScope, etc.), this adapter extracts
    and accumulates ``reasoning_content`` into
    ``additional_kwargs.reasoning_content`` on the AIMessage / AIMessageChunk.

    Use this in ``config.yaml`` as the ``use`` class for DeepSeek models served
    via an OpenAI-compatible gateway:

    .. code-block:: yaml

        use: deerflow.models.patched_reasoning:PatchedReasoningChatOpenAI
        model: deepseek-v4-pro
        openai_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
        supports_thinking: true
        when_thinking_enabled:
          extra_body:
            enable_thinking: true
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Extract ``reasoning_content`` from the streaming delta and preserve it."""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices", [])
        if not choices:
            return generation_chunk

        delta = choices[0].get("delta", {})
        reasoning = delta.get("reasoning_content")

        if reasoning and isinstance(generation_chunk.message, AIMessageChunk):
            kwargs = dict(generation_chunk.message.additional_kwargs)
            kwargs["reasoning_content"] = _merge_reasoning_content(
                kwargs.get("reasoning_content"), reasoning
            )
            generation_chunk.message.additional_kwargs = kwargs

        return generation_chunk

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Extract ``reasoning_content`` from non-streaming responses."""
        result = super()._create_chat_result(response, generation_info)
        if not result.generations:
            return result

        raw_response = response if isinstance(response, dict) else getattr(response, "model_dump", lambda: {})()
        choices = raw_response.get("choices", [])
        if not choices:
            return result

        message_payload = choices[0].get("message", {})
        reasoning = message_payload.get("reasoning_content")
        if reasoning:
            for gen in result.generations:
                if isinstance(gen, ChatGeneration) and isinstance(gen.message, AIMessage):
                    gen.message.additional_kwargs["reasoning_content"] = reasoning

        return result
