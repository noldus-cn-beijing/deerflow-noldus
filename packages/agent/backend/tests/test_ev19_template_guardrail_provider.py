"""Tests for Ev19TemplateGuardrailProvider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from deerflow.guardrails.ev19_template_provider import Ev19TemplateGuardrailProvider
from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def workspace_with_ev19(tmp_path):
    """Workspace with experiment-context.json containing ev19_template."""
    ctx = {
        "paradigm": "epm",
        "ev19_template": "PlusMaze-AllZones",
        "gate_completed": ["gate1_paradigm"],
    }
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
    return tmp_path


@pytest.fixture
def workspace_without_ev19(tmp_path):
    """Workspace with experiment-context.json missing ev19_template."""
    ctx = {"paradigm": "epm", "gate_completed": ["gate1_paradigm"]}
    (tmp_path / "experiment-context.json").write_text(json.dumps(ctx), encoding="utf-8")
    return tmp_path


@pytest.fixture
def empty_workspace(tmp_path):
    """Workspace without experiment-context.json."""
    return tmp_path


def _make_request(tool_name: str, args: dict) -> GuardrailRequest:
    return GuardrailRequest(tool_name=tool_name, tool_input=args, agent_id=None, timestamp="2026-05-08T00:00:00Z")


def _provider_with_workspace(ws: Path) -> Ev19TemplateGuardrailProvider:
    return Ev19TemplateGuardrailProvider(workspace_resolver=lambda: str(ws))


def test_allows_non_task_tools(workspace_without_ev19):
    """Provider only inspects task() calls, others pass through."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("read_file", {"path": "x"}))
    assert decision.allow is True


def test_allows_task_to_non_code_executor_subagents(workspace_without_ev19):
    """Provider only blocks task(code-executor); other subagents pass."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "knowledge-assistant", "prompt": "..."}))
    assert decision.allow is True


def test_blocks_task_code_executor_when_ev19_template_missing(workspace_without_ev19):
    """Block task(code-executor) when ev19_template field is missing."""
    p = _provider_with_workspace(workspace_without_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is False
    assert any(r.code == "ethoinsight.no_ev19_template" for r in decision.reasons)


def test_blocks_task_code_executor_when_workspace_has_no_context(empty_workspace):
    """Block task(code-executor) when experiment-context.json doesn't exist at all."""
    p = _provider_with_workspace(empty_workspace)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is False


def test_allows_task_code_executor_when_ev19_template_set(workspace_with_ev19):
    """Allow task(code-executor) when ev19_template is set."""
    p = _provider_with_workspace(workspace_with_ev19)
    decision = p.evaluate(_make_request("task", {"subagent_type": "code-executor", "prompt": "..."}))
    assert decision.allow is True


def test_async_evaluate_matches_sync(workspace_with_ev19):
    """aevaluate returns the same decision as evaluate."""
    p = _provider_with_workspace(workspace_with_ev19)

    async def run():
        sync_dec = p.evaluate(_make_request("task", {"subagent_type": "code-executor"}))
        async_dec = await p.aevaluate(_make_request("task", {"subagent_type": "code-executor"}))
        return sync_dec, async_dec

    sync_dec, async_dec = asyncio.run(run())
    assert sync_dec.allow == async_dec.allow
