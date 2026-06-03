"""Tests for ScriptInvocationOnlyProvider — path hardening + chart-maker gate.

Covers PR-3 changes:
- chart-maker bash gating (was previously un-gated)
- File-ops path validation (cp/mkdir/mv target must be in /mnt/user-data/)
- Backward compatibility (existing allowed patterns still work)
"""

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


# ============================================================
# Chart-maker bash gating (was previously un-gated)
# ============================================================

class TestChartMakerBashGating:
    """chart-maker now shares the same bash whitelist as code-executor."""

    def test_chart_maker_python_c_denied(self, provider):
        """chart-maker running python -c should be denied."""
        decision = provider.evaluate(_req(
            "bash",
            command="python -c 'print(1)'",
            agent_id="subagent:chart-maker",
        ))
        assert not decision.allow

    def test_chart_maker_cp_into_venv_denied(self, provider):
        """chart-maker copying into .venv should be denied."""
        decision = provider.evaluate(_req(
            "bash",
            command="cp /tmp/x.py /mnt/user-data/.venv/lib/site-packages/x.py",
            agent_id="subagent:chart-maker",
        ))
        assert not decision.allow

    def test_chart_maker_plot_script_allowed(self, provider):
        """chart-maker running a legitimate plot script should be allowed."""
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts.oft.plot_trajectory --input /mnt/user-data/workspace/a.json --output /mnt/user-data/workspace/p.png",
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow

    def test_chart_maker_mkdir_workspace_allowed(self, provider):
        """chart-maker creating workspace dirs should be allowed."""
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p /mnt/user-data/workspace/charts",
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow

    def test_chart_maker_ls_allowed(self, provider):
        """chart-maker read-only file ops should be allowed."""
        decision = provider.evaluate(_req(
            "bash",
            command="ls /mnt/user-data/workspace/",
            agent_id="subagent:chart-maker",
        ))
        assert decision.allow


# ============================================================
# File-ops path validation (cp/mkdir/mv)
# ============================================================

class TestFileOpsPathValidation:
    """Write file-ops must target paths within /mnt/user-data/ and not .venv/site-packages/skills."""

    # --- Deny: .venv paths ---

    def test_cp_into_venv_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp /tmp/x.py /mnt/user-data/.venv/lib/python3.12/site-packages/foo.py",
        ))
        assert not decision.allow
        assert any("site-packages" in r.message or ".venv" in r.message for r in decision.reasons)

    def test_mkdir_into_site_packages_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p /mnt/user-data/.venv/lib/python3.12/site-packages/ethoinsight/scripts/open_field",
        ))
        assert not decision.allow

    def test_cp_relative_dot_venv_denied(self, provider):
        """Relative path containing .venv should still be denied."""
        decision = provider.evaluate(_req(
            "bash",
            command="cp x.py .venv/lib/site-packages/x.py",
        ))
        assert not decision.allow

    def test_mkdir_relative_site_packages_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p lib/site-packages/foo",
        ))
        assert not decision.allow

    # --- Deny: /mnt/skills (read-only) ---

    def test_cp_into_skills_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp /tmp/x.py /mnt/skills/custom/foo/SKILL.md",
        ))
        assert not decision.allow

    def test_mkdir_in_skills_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir /mnt/skills/custom/new-skill",
        ))
        assert not decision.allow

    # --- Deny: absolute paths outside /mnt/user-data/ ---

    def test_cp_absolute_outside_sandbox_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp /tmp/x.py /etc/config",
        ))
        assert not decision.allow

    def test_mv_to_tmp_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mv /mnt/user-data/workspace/a.txt /tmp/a.txt",
        ))
        assert not decision.allow

    # --- Allow: valid sandbox paths ---

    def test_mkdir_workspace_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p /mnt/user-data/workspace/outputs",
        ))
        assert decision.allow

    def test_cp_within_workspace_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp /mnt/user-data/workspace/a.json /mnt/user-data/workspace/b.json",
        ))
        assert decision.allow

    def test_mv_within_workspace_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mv /mnt/user-data/workspace/tmp.json /mnt/user-data/workspace/done.json",
        ))
        assert decision.allow

    def test_mkdir_outputs_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p /mnt/user-data/outputs/charts",
        ))
        assert decision.allow

    def test_cp_to_uploads_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp /mnt/user-data/workspace/report.html /mnt/user-data/uploads/report.html",
        ))
        assert decision.allow

    # --- Allow: read-only file ops (no path validation) ---

    def test_ls_anywhere_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="ls /tmp",
        ))
        assert decision.allow

    def test_cat_anywhere_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cat /etc/hostname",
        ))
        assert decision.allow

    def test_grep_anywhere_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="grep -r 'pattern' /mnt/skills/",
        ))
        assert decision.allow

    def test_head_anywhere_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="head -n 5 /mnt/user-data/workspace/data.txt",
        ))
        assert decision.allow

    def test_tail_anywhere_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="tail -f /var/log/app.log",
        ))
        assert decision.allow

    # --- Allow: relative paths without .venv/site-packages ---

    def test_mkdir_relative_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="mkdir -p outputs/charts",
        ))
        assert decision.allow

    def test_cp_relative_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="cp a.txt b.txt",
        ))
        assert decision.allow

    # --- Edge: unparseable commands ---

    def test_unclosed_quote_denied(self, provider):
        """Unparseable command (unclosed quote) should be denied."""
        decision = provider.evaluate(_req(
            "bash",
            command="cp file 'unclosed",
        ))
        assert not decision.allow


# ============================================================
# Backward compatibility (existing patterns still work)
# ============================================================

class TestBackwardCompatibility:
    """Ensure existing allowed patterns are not broken by the hardening."""

    def test_code_executor_script_invocation_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio --input /tmp/a.txt --output /tmp/o.json",
        ))
        assert decision.allow

    def test_code_executor_common_script_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.scripts._common.plot_trajectory --input /tmp/a.txt --output /tmp/p.png",
        ))
        assert decision.allow

    def test_code_executor_python_c_denied(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command='python -c "print(1)"',
        ))
        assert not decision.allow

    def test_code_executor_pip_install_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="pip install pandas"))
        assert not decision.allow

    def test_lead_agent_not_gated(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -c 'help(x)'",
            agent_id="",
        ))
        assert decision.allow

    def test_data_analyst_not_gated(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -c 'help(x)'",
            agent_id="subagent:data-analyst",
        ))
        assert decision.allow

    def test_non_bash_tool_allowed(self, provider):
        decision = provider.evaluate(_req("read_file"))
        assert decision.allow

    def test_write_file_non_handoff_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", agent_id="subagent:code-executor"))
        # No command/ path → no handoff_ in path → allowed
        assert decision.allow


# ============================================================
# Async parity
# ============================================================

class TestAsyncParity:
    """aevaluate must match evaluate for all new code paths."""

    @pytest.mark.asyncio
    async def test_chart_maker_deny_async(self, provider):
        req = _req("bash", command="python -c '1'", agent_id="subagent:chart-maker")
        sync_decision = provider.evaluate(req)
        async_decision = await provider.aevaluate(req)
        assert sync_decision.allow == async_decision.allow

    @pytest.mark.asyncio
    async def test_path_validation_deny_async(self, provider):
        req = _req("bash", command="cp /tmp/x.py /mnt/user-data/.venv/lib/x.py")
        sync_decision = provider.evaluate(req)
        async_decision = await provider.aevaluate(req)
        assert sync_decision.allow == async_decision.allow

    @pytest.mark.asyncio
    async def test_path_validation_allow_async(self, provider):
        req = _req("bash", command="mkdir -p /mnt/user-data/workspace/outputs")
        sync_decision = provider.evaluate(req)
        async_decision = await provider.aevaluate(req)
        assert sync_decision.allow == async_decision.allow
