"""Regression tests for the summarization config baked into the running config.yaml.

These tests are NOT trying to validate Pydantic — that is covered by SummarizationConfig
itself. They protect against accidental regressions of the trigger/keep/summary_prompt
shape that previously caused mid-workflow archiving and subagent re-dispatch (see
2026-05-19 dogfood thread 5288a885).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from deerflow.config.summarization_config import SummarizationConfig


def _locate_config_yaml() -> Path | None:
    """Find the active config.yaml. Mirrors the search order used by AppConfig.from_file.

    1. DEER_FLOW_CONFIG_PATH env var
    2. ./config.yaml relative to cwd
    3. ../config.yaml (when cwd is backend/)
    4. The well-known location at <repo_root>/packages/agent/config.yaml
    """
    env = os.environ.get("DEER_FLOW_CONFIG_PATH")
    if env and Path(env).exists():
        return Path(env)

    cwd = Path.cwd()
    for candidate in (cwd / "config.yaml", cwd.parent / "config.yaml"):
        if candidate.exists():
            return candidate

    # Walk up from this test file looking for packages/agent/config.yaml
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "packages" / "agent" / "config.yaml"
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def loaded_summarization() -> SummarizationConfig:
    config_path = _locate_config_yaml()
    if config_path is None:
        pytest.skip("config.yaml not found (likely fresh checkout without local config)")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SummarizationConfig(**raw["summarization"])


def test_summarization_enabled(loaded_summarization: SummarizationConfig) -> None:
    assert loaded_summarization.enabled is True


def test_trigger_is_token_based_with_workflow_headroom(loaded_summarization: SummarizationConfig) -> None:
    """A token-based trigger with >= 30000 tokens leaves room for at least one full
    EPM/OFT analysis pipeline before archiving kicks in. A pure messages-based trigger
    of ~15 (the previous value) was found to fire mid-subagent-dispatch and erase the
    dispatched ToolMessage."""
    triggers = loaded_summarization.trigger
    assert triggers is not None
    if not isinstance(triggers, list):
        triggers = [triggers]

    token_triggers = [t for t in triggers if t.type == "tokens"]
    assert token_triggers, "Expected at least one token-based trigger"
    assert max(t.value for t in token_triggers) >= 30000, "Token trigger too low — will archive mid-workflow"


def test_keep_preserves_enough_recent_messages(loaded_summarization: SummarizationConfig) -> None:
    """`keep` must retain enough recent messages to span an entire subagent dispatch
    chain (task tool_call + ToolMessage + lead's interpretation + present_files pair)."""
    keep = loaded_summarization.keep
    if keep.type == "messages":
        assert keep.value >= 20, f"keep={keep.value} messages is too small — risks splitting subagent dispatch chains"
    elif keep.type == "tokens":
        assert keep.value >= 10000, f"keep={keep.value} tokens is too small"


def test_summary_prompt_preserves_dispatched_subagents(loaded_summarization: SummarizationConfig) -> None:
    """The summary prompt MUST instruct the model to record dispatched subagent names
    and completion status, otherwise the lead agent loses memory of what it already
    dispatched after archiving (root cause of 2026-05-19 thread 5288a885 re-dispatch
    loop)."""
    prompt = loaded_summarization.summary_prompt
    assert prompt is not None, "summary_prompt must not be null — LangChain default loses subagent dispatch context"
    lowered = prompt.lower()
    assert "subagent" in lowered or "派遣" in prompt, "Prompt must mention dispatched subagents"
    assert "产物" in prompt or "artifact" in lowered or "path" in lowered, "Prompt must mention artifact paths"


def test_summary_prompt_contains_messages_placeholder(loaded_summarization: SummarizationConfig) -> None:
    """Sanity check: LangChain SummarizationMiddleware substitutes {messages} into the
    prompt. Without this placeholder the model sees an empty conversation."""
    prompt = loaded_summarization.summary_prompt
    assert prompt is not None
    assert "{messages}" in prompt, "Prompt must contain {messages} placeholder for LangChain to inject history"
