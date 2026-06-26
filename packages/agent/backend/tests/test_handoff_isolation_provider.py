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


# ---------------------------------------------------------------------------
# Task 4 (P2) spec 2026-06-26 §任务4：handoff 读取显式归属校验
# ---------------------------------------------------------------------------


class TestHandoffThreadOwnership:
    """显式断言 handoff 路径归属当前 thread 的 workspace/shared 虚拟根。

    即使 lead 通过 {{handoff://X}} 把某路径写进 authorized_paths，路径本身也必须
    落在 thread-scoped 虚拟根（/mnt/user-data/workspace/ 或 /mnt/shared/）下——
    这是结构约束，独立于字符串 allowlist。catch：foreign-thread 注入、host 绝对
    路径、路径穿越变体。
    """

    def test_handoff_under_workspace_root_allowed(self):
        """workspace 根下的 handoff（哪怕未单独授权）走 self-outbox/allowlist 判断。"""
        p = HandoffIsolationProvider(
            authorized_paths={"/mnt/user-data/workspace/handoff_data_analyst.json"},
            self_outbox_subagent_name="data-analyst",
        )
        decision = p.evaluate(_request(
            file_path="/mnt/user-data/workspace/handoff_data_analyst.json",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow is True

    def test_handoff_under_shared_root_evaluated_normally(self):
        """/mnt/shared/ 根下的 handoff 不被归属校验本身拒（走 allowlist）。"""
        p = HandoffIsolationProvider(authorized_paths={
            "/mnt/shared/handoff_code_executor.json"
        })
        decision = p.evaluate(_request(
            file_path="/mnt/shared/handoff_code_executor.json",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow is True

    def test_handoff_foreign_root_denied_even_if_authorized_string_matches(self):
        """路径不在 thread-scoped 根下（如 host 绝对路径 / 跨 thread 注入）→ 结构性拒。

        即使调用方误把一个 foreign-root 路径塞进 authorized_paths，归属校验也兜底拒掉，
        fail-closed。这是「显式断言 is_relative_to(workspace/shared_root)」的结构实现。
        """
        p = HandoffIsolationProvider(authorized_paths={
            "/opt/secret/threads/other-thread/handoff_code_executor.json"
        })
        decision = p.evaluate(_request(
            file_path="/opt/secret/threads/other-thread/handoff_code_executor.json",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow is False
        assert decision.reasons[0].code == "handoff_isolation.foreign_root"

    def test_handoff_relative_traversal_denied(self):
        """路径穿越变体（不以 /mnt 虚拟根开头）→ 拒。"""
        p = HandoffIsolationProvider(authorized_paths={"../../etc/handoff_x.json"})
        decision = p.evaluate(_request(
            file_path="../../etc/handoff_x.json",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow is False
        assert decision.reasons[0].code == "handoff_isolation.foreign_root"

    def test_own_outbox_under_foreign_root_still_denied(self):
        """归属校验优先于 self-outbox：own handoff 名但路径在 foreign root → 拒。

        否则 subagent 可构造 ``/opt/x/handoff_data_analyst.json`` 绕过 self-outbox
        读任意位置的「自己的 handoff」名文件。
        """
        p = HandoffIsolationProvider(
            authorized_paths=set(),
            self_outbox_subagent_name="data-analyst",
        )
        decision = p.evaluate(_request(
            file_path="/opt/foreign/handoff_data_analyst.json",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow is False
        assert decision.reasons[0].code == "handoff_isolation.foreign_root"
