# Issue #2 Thinking Field 400 Error: Diagnosis Notes

**Date**: 2026-05-14
**Status**: Diagnosed — root cause identified in upstream `langchain_anthropic`
**Task**: Task 3 of Issue #2 diagnosis (fix goes in Task 4)

## Error Summary

```
messages[7]: missing field 'thinking' at line 1 column 18538
```

thread 5046a6e6 dogfood: 400 error from newapi that killed the session entirely.
After gateway reload (Issue #1), historical messages were restored from checkpoint,
and `_format_messages` in `langchain_anthropic` sent a content block with
`type="thinking"` but no `thinking` key to the API.

## Investigation Scope

Checked all possible loss locations in order of probability:

### (a) archiving_summarization.py — NOT the culprit

- `SummarizationMiddleware._partition_messages` (summarization.py:479-488): Simple list slicing, no message mutation.
- `ArchivingSummarizationMiddleware` overrides `abefore_model`/`before_model` but still calls the parent's `_partition_messages`.
- `_archive_messages` uses `messages_to_dict()` only for archiving to disk — it does NOT modify the preserved messages in state.

### (b) think_tag_middleware.py — NOT the culprit

- `ThinkTagMiddleware._rewrite` (think_tag_middleware.py:116-139): For list content, iterates through parts. Only processes `type="text"` blocks. NON-text blocks (including `type="thinking"`) are appended unchanged at line 131: `new_parts.append(part)`.
- The middleware runs in `after_model` and only rewrites the NEWEST AI message, not historical ones.

### (c) Checkpointer serialization — NOT the culprit

- Uses `JsonPlusSerializer` (ormsgpack) with Pydantic v2 serialization: `model_dump()` → msgpack → `AIMessage(**data)`.
- For dict content like `{"type": "thinking", "thinking": "..."}`, this round-trips correctly.
- Verified: `AIMessage.model_dump()` + `AIMessage(**data)` preserves thinking blocks.
- Verified: `messages_to_dict()` + `messages_from_dict()` preserves thinking blocks.

### (d) TitleMiddleware — NOT the culprit

- Only reads content for title generation. Does NOT modify message content in state.

## Root Cause Found

### Location

`langchain_anthropic/chat_models.py` lines 2071-2075 (streaming event handler)

```python
elif event.delta.type in {"thinking_delta", "signature_delta"}:
    content_block = event.delta.model_dump()
    content_block["index"] = event.index
    content_block["type"] = "thinking"
    message_chunk = AIMessageChunk(content=[content_block])
```

### What Happens

1. When Anthropic streaming returns a thinking response, it emits:
   - One or more `thinking_delta` events (each with `thinking` text)
   - One `signature_delta` event (with `signature`, no `thinking` text)

2. The event handler treats BOTH as the same: `type = "thinking"`.

3. For `signature_delta`, `event.delta.model_dump()` produces:
   ```python
   {"type": "signature_delta", "signature": "EqQBCgIY..."}
   ```
   Then `content_block["type"] = "thinking"` overwrites the type:
   ```python
   {"type": "thinking", "signature": "EqQBCgIY..."}
   ```
   Note: NO `thinking` key!

4. Multiple `AIMessageChunk`s are merged via `add_ai_message_chunks` → `merge_content` → `merge_lists`. The result has SEPARATE thinking blocks:
   ```python
   [
       {"type": "thinking", "thinking": "actual reasoning...", "index": 0},
       {"type": "thinking", "signature": "EqQBCgIY...", "index": 0},  # <-- no thinking key!
       {"type": "text", "text": "final answer"},
   ]
   ```

5. The message is stored in checkpointer. On the NEXT turn, when this message is sent back as conversation history, `_format_messages` (line 569-577) filters thinking blocks:
   ```python
   elif block["type"] == "thinking":
       content.append({k: v for k, v in block.items()
                       if k in ("type", "thinking", "cache_control", "signature")})
   ```
   For the signature-only block, the result is:
   ```python
   {"type": "thinking", "signature": "EqQBCgIY..."}
   ```
   Missing `thinking` key → newapi validates and returns 400: `missing field 'thinking'`.

### Impact Flow for thread 5046a6e6

1. User interacts with agent, thinking is enabled
2. DeepSeek (via newapi) returns Anthropic-format streaming response with thinking
3. `thinking_delta` + `signature_delta` events create malformed thinking blocks in AIMessage content
4. Summarization triggers ("Archived 8 messages") — the preserved messages include the malformed one
5. Gateway reload (Issue #1) — checkpoint restored from SQLite
6. Next user turn: message with malformed thinking block sent back as history → 400 error

### Why This Passes on the First Turn

On the FIRST turn, the malformed thinking block is only in the message that's BEING GENERATED, not sent as input. The input messages to the FIRST turn don't have thinking blocks (the conversation starts fresh). The malformation only causes an error when the message IS SENT BACK as conversation history on a SUBSEQUENT turn.

## Confirmation

The diagnostic test `test_signature_delta_without_thinking_creates_malformed_block` in
`tests/test_thinking_field_preserved.py` reproduces this bug:
- Creates an AIMessage simulating the merged content after streaming
- Passes it through `_format_messages`
- Asserts thinking blocks have the `thinking` key
- FAILS for the signature-only block

## Fix Recommendation (for Task 4)

The fix should be in the Noldus codebase (not upstream), since modifying `langchain_anthropic` directly would be fragile. Options:

### Option A: Fix in ClaudeChatModel._get_request_payload (Recommended)

In `claude_provider.py`, override `_get_request_payload` to filter out thinking blocks that lack the `thinking` key before passing to the parent:

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs):
    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
    # Strip thinking blocks without 'thinking' key (signature_delta artifact)
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") == "thinking" and "thinking" not in b)
            ]
    return payload
```

### Option B: Fix in a new middleware (after_message)

Add a middleware that cleans up malformed thinking blocks before they enter the checkpointer:

```python
class FixThinkingBlocksMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        """Merge signature into thinking blocks, remove orphaned signature-only blocks."""
        ...
```

### Option C: Merge signature into the thinking text block

After streaming, merge the `signature` from the signature-only block into the preceding thinking block that has the `thinking` key:

```python
for msg in messages:
    if isinstance(msg.content, list):
        merged = []
        pending_sig = None
        for block in msg.content:
            if block.get("type") == "thinking":
                if "thinking" in block:
                    if pending_sig:
                        block["signature"] = pending_sig
                        pending_sig = None
                    merged.append(block)
                elif "signature" in block:
                    pending_sig = block["signature"]
            else:
                merged.append(block)
        msg.content = merged
```

**Recommendation**: Option A is simplest and safest — it's a surgical fix that only affects the serialization path, not message storage.

## Files Examined

| File | Lines | Finding |
|------|-------|---------|
| `archiving_summarization.py` | 479-488 | `_partition_messages` is pure slicing, no mutation |
| `think_tag_middleware.py` | 116-141 | `_rewrite` correctly preserves non-text blocks (line 131) |
| `langgraph/checkpoint/serde/jsonplus.py` | 224-235 | Pydantic v2 msgpack serialization, should preserve dicts |
| `langchain_anthropic/chat_models.py` | 569-577 | `_format_messages` filters thinking blocks to 4 keys |
| `langchain_anthropic/chat_models.py` | 2071-2075 | **ROOT CAUSE**: `signature_delta` creates thinking block without `thinking` key |
| `langchain_core/messages/base.py` | 200-260 | `content_blocks` wraps unknown types, but raw content preserved |
| `langchain_core/messages/ai.py` | 243-299 | `AIMessage.content_blocks` adds reasoning from `additional_kwargs` |
| `langchain_core/messages/content.py` | 856-877 | `KNOWN_BLOCK_TYPES` does NOT include "thinking" |
| `_compat.py` (langchain_anthropic) | 143-151 | `_convert_from_v1_to_anthropic` converts reasoning→thinking (v1 only) |

## Verification Commands

```bash
# Run the diagnostic tests
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_thinking_field_preserved.py -v

# Expected result: TestAnthropicFormatMessages::test_signature_delta_without_thinking_creates_malformed_block FAILS
# All other tests should PASS or SKIP
```
