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
        decision = provider.evaluate(_req("bash", command="cp /mnt/user-data/workspace/a.json /mnt/user-data/workspace/b.json"))
        assert decision.allow

    def test_ls_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="ls /tmp"))
        assert decision.allow


class TestChartMakerResolveAndDumpHeadersAllowed:
    """2026-06-04 regression fix: chart-maker's documented workflow runs
    `ethoinsight.catalog.resolve --mode charts` (SKILL.md step 2) which has a
    REQUIRED --columns-file produced by `ethoinsight.parse.dump_headers`. The
    allow-pattern only matched `ethoinsight.scripts.*`, so chart-maker was blocked
    at step 2 and could never produce a chart. These anchors lock the two CLIs in.
    """

    def test_catalog_resolve_charts_allowed_for_chart_maker(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command=(
                'python -m ethoinsight.catalog.resolve --mode charts --paradigm zero_maze '
                '--columns-file /mnt/user-data/workspace/columns.json '
                '--raw-files-json /mnt/user-data/workspace/raw_files.json '
                '--workspace-dir /mnt/user-data/workspace --user-intent "画图" '
                '--output /mnt/user-data/workspace/plan_charts.json'
            ),
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow

    def test_dump_headers_allowed_for_chart_maker(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command=(
                "python -m ethoinsight.parse.dump_headers "
                "--input /mnt/user-data/uploads/a.xlsx "
                "--output /mnt/user-data/workspace/columns.json"
            ),
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow

    def test_catalog_resolve_also_allowed_for_code_executor(self, provider):
        # Whitelisting the module is agent-agnostic; code-executor passing through
        # the same allow branch is harmless (its workflow does not call charts mode).
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.catalog.resolve --mode metrics --paradigm epm "
                    "--columns-file /mnt/user-data/workspace/columns.json "
                    "--raw-files-json /mnt/user-data/workspace/raw_files.json "
                    "--workspace-dir /mnt/user-data/workspace --output /mnt/user-data/workspace/p.json",
            agent_id="subagent:code-executor",
        ))
        assert decision.allow

    def test_lookalike_other_ethoinsight_module_still_denied(self, provider):
        # The fix must stay precise: only scripts.* / catalog.resolve / parse.dump_headers.
        # A different ethoinsight module must NOT be opened up by the widened pattern.
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.parse.parse_trajectory --input /tmp/a.txt",
            agent_id="subagent:chart-maker",
        ))
        assert not decision.allow

    def test_catalog_other_submodule_still_denied(self, provider):
        # Only catalog.resolve is allowed, not arbitrary catalog.* modules.
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.catalog.loader --paradigm epm",
            agent_id="subagent:chart-maker",
        ))
        assert not decision.allow


class TestValidateCatalogGuardrail:
    """2026-06-06: L-B catalog-driven validation CLI added to whitelist.

    ``ethoinsight.validate_catalog`` must be allowed for code-executor;
    lookalike modules (validate_catalogX, validate) must still be denied.
    """

    def test_validate_catalog_allowed_for_code_executor(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command=(
                "python -m ethoinsight.validate_catalog "
                "--plan /mnt/user-data/workspace/plan_metrics.json"
            ),
            agent_id="subagent:code-executor",
        ))
        assert decision.allow

    def test_validate_catalog_allowed_for_chart_maker_too(self, provider):
        # Whitelisting is agent-agnostic; chart-maker passing through is harmless.
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.validate_catalog --plan /mnt/user-data/workspace/plan_metrics.json",
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow

    def test_lookalike_validate_catalog_x_still_denied(self, provider):
        # Only validate_catalog exactly, not validate_catalogX.
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.validate_catalogX --plan /tmp/p.json",
            agent_id="subagent:code-executor",
        ))
        assert not decision.allow

    def test_lookalike_ethoinsight_validate_still_denied(self, provider):
        # Only validate_catalog, not ethoinsight.validate.
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.validate --plan /tmp/p.json",
            agent_id="subagent:code-executor",
        ))
        assert not decision.allow


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
