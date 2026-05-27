"""Reproduce IntentClassificationGuardrailProvider false-positive bug (M2: diagnose only).

The bug: 843fe2b8 recorded 231 false rejections, f3fbce44 recorded 343, all while
[intent] E2E_FULL_ASKVIZ was present in messages. The provider rejected tool calls
even though the intent line was already declared.

Hypothesis (for PR-4 investigation):
- ContextVar _lead_messages not set across async/sync middleware boundaries
- Middleware ordering: IntentBridgeMiddleware may run after GuardrailMiddleware
  on some code paths
- Content extraction: _extract_declared_intents scans content strings, but
  content blocks (list-of-dicts) might not be normalized correctly

This test file REPRODUCES the expected correct behavior and documents the
gap between unit-test behavior and production behavior.
"""

from langchain_core.messages import AIMessage

from deerflow.guardrails.intent_classification_provider import (
    IntentClassificationGuardrailProvider,
    _extract_declared_intents,
    _lead_messages,
    _INTENT_LINE_RE,
)
from deerflow.guardrails.provider import GuardrailRequest


# ── Content extraction correctness ──────────────────────────────────────────


def test_intent_line_re_matches_typical_format():
    """The regex must match [intent] E2E_FULL_ASKVIZ in typical lead output."""
    cases = [
        "[intent] E2E_FULL_ASKVIZ",
        "[intent] E2E_FULL",
        "[intent] E2E_MIN",
        "[intent] CHART\n下一步派 chart-maker",
        "[intent] E2E_FULL_ASKVIZ\n用户上传了数据，我来分析",
    ]
    for text in cases:
        match = _INTENT_LINE_RE.search(text)
        assert match is not None, f"Failed to match: {text!r}"


def test_extract_declared_intents_finds_intent_in_middle_message():
    """intent in messages[-2] should still be found."""
    msgs = [
        AIMessage(content="[intent] E2E_FULL_ASKVIZ\n开始分析"),
        AIMessage(content="已经完成了代码执行，现在看数据解读结果"),
    ]
    declared = _extract_declared_intents(msgs)
    assert "E2E_FULL_ASKVIZ" in declared


def test_extract_declared_intents_finds_intent_in_earliest_message():
    """intent only in messages[0] should be found (scan all, not just last)."""
    msgs = [
        AIMessage(content="[intent] E2E_FULL_ASKVIZ"),
        AIMessage(content="task done"),
        AIMessage(content="reading handoff"),
        AIMessage(content="dispatching chart-maker"),
    ]
    declared = _extract_declared_intents(msgs)
    assert "E2E_FULL_ASKVIZ" in declared


def test_extract_declared_intents_handles_list_content():
    """Content blocks as lists of dicts should be normalized."""
    msgs = [
        AIMessage(content=[{"type": "text", "text": "[intent] E2E_FULL_ASKVIZ\n开始分析"}]),
    ]
    declared = _extract_declared_intents(msgs)
    assert "E2E_FULL_ASKVIZ" in declared


# ── Provider allow/deny with explicit intent ────────────────────────────────


def test_provider_allows_when_intent_present(tmp_path):
    """With intent in messages, provider should allow non-read_file tool calls."""
    msgs = [AIMessage(content="[intent] E2E_FULL_ASKVIZ\n开始分析")]
    _lead_messages.set(msgs)

    provider = IntentClassificationGuardrailProvider()
    req = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "chart-maker"})
    decision = provider.evaluate(req)
    assert decision.allow is True, f"Expected allow but got: {decision.reasons}"


def test_provider_blocks_when_no_intent():
    """Without intent in messages, non-read_file calls should be blocked."""
    msgs = [AIMessage(content="开始分析")]
    _lead_messages.set(msgs)

    provider = IntentClassificationGuardrailProvider()
    req = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "chart-maker"})
    decision = provider.evaluate(req)
    assert decision.allow is False
    assert any(r.code == "ethoinsight.intent_not_declared" for r in decision.reasons)


def test_provider_always_allows_read_file():
    """read_file must always be allowed regardless of intent state."""
    _lead_messages.set([])
    provider = IntentClassificationGuardrailProvider()
    req = GuardrailRequest(tool_name="read_file", tool_input={})
    decision = provider.evaluate(req)
    assert decision.allow is True


# ── ContextVar edge cases (documenting known gap) ───────────────────────────


def test_contextvar_not_set_fails_open():
    """When _lead_messages is not set (default=None), provider should fail-open."""
    _lead_messages.set(None)
    provider = IntentClassificationGuardrailProvider()
    req = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "code-executor"})
    decision = provider.evaluate(req)
    assert decision.allow is True, (
        "Expected fail-open when ContextVar is None (bootstrap/test safety)"
    )


def test_contextvar_persistence_across_sync_calls():
    """ContextVar should persist within the same thread/event-loop.

    This is the key test for the async/sync boundary hypothesis. If this test
    passes but production still shows false positives, the bug is likely in
    middleware ordering, not ContextVar semantics.
    """
    _lead_messages.set([AIMessage(content="[intent] E2E_FULL_ASKVIZ")])

    provider = IntentClassificationGuardrailProvider()
    # Simulate two sequential tool calls in the same turn
    req1 = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "code-executor"})
    decision1 = provider.evaluate(req1)
    assert decision1.allow is True

    req2 = GuardrailRequest(tool_name="task", tool_input={"subagent_type": "chart-maker"})
    decision2 = provider.evaluate(req2)
    assert decision2.allow is True
