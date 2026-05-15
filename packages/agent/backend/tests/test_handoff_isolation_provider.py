"""Tests for HandoffIsolationProvider — gates subagent read_file on handoff_*.json."""

import pytest

from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
from deerflow.guardrails.provider import GuardrailRequest


def _request(
    *,
    tool_name: str = "read_file",
    file_path: str = "",
    agent_id: str | None = None,
) -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input={"file_path": file_path},
        agent_id=agent_id,
    )


def test_no_passport_always_allowed():
    """No passport = lead call, always allowed."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id=None,
    ))
    assert decision.allow is True


def test_non_read_file_tools_always_allowed():
    """Other tools (write_file, ls, bash...) are not gated."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(
        tool_name="write_file",
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is True


def test_non_handoff_files_always_allowed():
    """read_file on non-handoff files is not gated."""
    p = HandoffIsolationProvider(authorized_paths=set())
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/metrics.csv",
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is True


def test_authorized_handoff_path_allowed():
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_code_executor.json"
    })
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is True


def test_unauthorized_handoff_path_denied():
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_data_analyst.json"
    })
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:knowledge-assistant",
    ))
    assert decision.allow is False
    assert decision.reasons[0].code == "handoff_isolation.unauthorized"
    assert "handoff_code_executor.json" in decision.reasons[0].message


def test_self_outbox_always_allowed():
    """data-analyst can read its own handoff_data_analyst.json."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_data_analyst.json",
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is True


def test_self_outbox_with_hyphenated_name():
    """Subagent names use '-' (e.g. 'code-executor'), filename uses '_'."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="code-executor",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:code-executor",
    ))
    assert decision.allow is True


def test_self_outbox_does_not_allow_peer_handoff():
    """data-analyst CANNOT read handoff_code_executor.json via _is_own_handoff."""
    p = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    decision = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:data-analyst",
    ))
    assert decision.allow is False


@pytest.mark.asyncio
async def test_aevaluate_delegates_to_evaluate():
    """Async path returns same decision as sync."""
    p = HandoffIsolationProvider(authorized_paths={
        "/mnt/user-data/workspace/handoff_code_executor.json"
    })
    sync = p.evaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:data-analyst",
    ))
    async_d = await p.aevaluate(_request(
        file_path="/mnt/user-data/workspace/handoff_code_executor.json",
        agent_id="subagent:data-analyst",
    ))
    assert sync.allow == async_d.allow
