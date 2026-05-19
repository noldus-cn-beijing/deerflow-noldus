"""Tests for ScriptInvocationOnlyProvider."""

from __future__ import annotations

import pytest

from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def provider():
    from deerflow.guardrails.script_invocation_only_provider import (
        ScriptInvocationOnlyProvider,
    )
    return ScriptInvocationOnlyProvider()


def _req(tool_name: str, command: str = "", agent_id: str = "subagent:code-executor") -> GuardrailRequest:
    return GuardrailRequest(
        tool_name=tool_name,
        tool_input={"command": command} if command else {},
        agent_id=agent_id,
    )


class TestNonBashAlwaysAllowed:
    def test_read_file_allowed(self, provider):
        decision = provider.evaluate(_req("read_file"))
        assert decision.allow

    def test_write_file_allowed(self, provider):
        decision = provider.evaluate(_req("write_file"))
        assert decision.allow

    def test_ls_allowed(self, provider):
        decision = provider.evaluate(_req("ls"))
        assert decision.allow


class TestNonCodeExecutorAgentNotGated:
    def test_lead_agent_bash_allowed(self, provider):
        # lead agent has no code-executor in agent_id
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'", agent_id=""))
        assert decision.allow

    def test_other_subagent_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'",
                                          agent_id="subagent:data-analyst"))
        assert decision.allow


class TestCodeExecutorBashAllowList:
    def test_script_invocation_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio --input /tmp/a.txt --output /tmp/o.json",
        ))
        assert decision.allow

    def test_common_script_invocation_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts._common.plot_trajectory --input /tmp/a.txt --output /tmp/p.png",
        ))
        assert decision.allow

    def test_mkdir_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mkdir -p /mnt/user-data/workspace/outputs"))
        assert decision.allow

    def test_cp_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cp /a /b"))
        assert decision.allow

    def test_ls_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="ls /tmp"))
        assert decision.allow


class TestCodeExecutorBashDenyList:
    def test_python_c_help_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight import parse; help(parse.parse_trajectory)"',
        ))
        assert not decision.allow
        assert decision.reasons
        assert "script" in decision.reasons[0].message.lower()

    def test_python_c_import_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight import charts; print(charts.box_plot)"',
        ))
        assert not decision.allow

    def test_python_c_smoke_test_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "from ethoinsight.metrics.epm import *; print(\'OK\')"',
        ))
        assert not decision.allow

    def test_pip_install_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="pip install pandas"))
        assert not decision.allow

    def test_arbitrary_python_script_denied(self, provider):
        # python invoking a custom script outside ethoinsight.scripts is denied
        decision = provider.evaluate(_req(
            "bash",
            command="python /tmp/my_analysis.py --input /tmp/a.txt",
        ))
        assert not decision.allow

    def test_rm_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="rm /tmp/file"))
        assert not decision.allow

    def test_curl_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com"))
        assert not decision.allow


class TestDenyReason:
    def test_reason_message_contains_correct_path_hint(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "python -m ethoinsight.scripts" in msg
        assert "by-paradigm" in msg

    def test_reason_code_is_stable(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'help(x)'"))
        assert decision.reasons[0].code == "script_invocation_only.not_a_script_call"


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate(provider):
    req = _req("bash", command="python -c 'help(x)'")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
    assert sync_decision.reasons[0].code == async_decision.reasons[0].code
