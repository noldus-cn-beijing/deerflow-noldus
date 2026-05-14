"""Tests for LeadAgentExecutionBoundaryProvider.

Modeled after test_script_invocation_only_provider.py.

Coverage:
- write_file with executable extension (.py/.sh/.ipynb/.bash/.zsh) → deny
- write_file with data extension (.md/.json/.csv/.txt) → allow
- bash whitelist: ethoinsight.parse / ethoinsight.catalog / safe file ops
- bash deny: python -c, python /path/to/file.py, pip install, arbitrary commands
- subagent passport (agent_id starts with "subagent:") → always allow (passthrough)
- non-bash/non-write_file tools (read_file, ls, task, ask_clarification) → always allow
- deny reason code stability + message helpfulness
- sync evaluate() and async aevaluate() agree
"""

from __future__ import annotations

import pytest

from deerflow.guardrails.provider import GuardrailRequest


@pytest.fixture
def provider():
    from deerflow.guardrails.lead_execution_boundary_provider import (
        LeadAgentExecutionBoundaryProvider,
    )
    return LeadAgentExecutionBoundaryProvider()


def _req(tool_name: str, *, path: str = "", command: str = "", agent_id: str | None = None) -> GuardrailRequest:
    """Build a GuardrailRequest. agent_id=None simulates lead; "subagent:foo" simulates subagent."""
    tool_input: dict = {}
    if path:
        tool_input["path"] = path
    if command:
        tool_input["command"] = command
    return GuardrailRequest(tool_name=tool_name, tool_input=tool_input, agent_id=agent_id)


class TestSubagentPassportPassthrough:
    """Subagents (agent_id starting with 'subagent:') are never gated by this provider."""

    def test_subagent_write_py_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py", agent_id="subagent:code-executor"))
        assert decision.allow

    def test_subagent_bash_python_c_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="python -c 'import x'", agent_id="subagent:data-analyst"))
        assert decision.allow

    def test_subagent_bash_arbitrary_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com", agent_id="subagent:report-writer"))
        assert decision.allow


class TestNonGatedToolsAlwaysAllowed:
    """For lead (agent_id=None), tools other than write_file/bash always pass."""

    def test_read_file_allowed(self, provider):
        decision = provider.evaluate(_req("read_file"))
        assert decision.allow

    def test_ls_allowed(self, provider):
        decision = provider.evaluate(_req("ls"))
        assert decision.allow

    def test_glob_allowed(self, provider):
        decision = provider.evaluate(_req("glob"))
        assert decision.allow

    def test_grep_allowed(self, provider):
        decision = provider.evaluate(_req("grep"))
        assert decision.allow

    def test_task_allowed(self, provider):
        decision = provider.evaluate(_req("task"))
        assert decision.allow

    def test_ask_clarification_allowed(self, provider):
        decision = provider.evaluate(_req("ask_clarification"))
        assert decision.allow

    def test_present_files_allowed(self, provider):
        decision = provider.evaluate(_req("present_files"))
        assert decision.allow

    def test_str_replace_allowed(self, provider):
        # str_replace is for editing existing files; not gated here
        decision = provider.evaluate(_req("str_replace"))
        assert decision.allow


class TestWriteFileForbiddenExtensions:
    """Lead writing executable script files is denied."""

    def test_write_py_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/gen_charts.py"))
        assert not decision.allow
        assert decision.reasons
        assert decision.reasons[0].code == "lead_execution_boundary.script_write_forbidden"

    def test_write_sh_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/run.sh"))
        assert not decision.allow

    def test_write_ipynb_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/analysis.ipynb"))
        assert not decision.allow

    def test_write_bash_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/script.bash"))
        assert not decision.allow

    def test_write_zsh_denied(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/script.zsh"))
        assert not decision.allow

    def test_write_py_case_insensitive(self, provider):
        # Tolerate uppercase extension (.PY)
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/X.PY"))
        assert not decision.allow


class TestWriteFileAllowedExtensions:
    """Lead writing data/doc files is allowed."""

    def test_write_md_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/notes.md"))
        assert decision.allow

    def test_write_json_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/metric_plan.json"))
        assert decision.allow

    def test_write_csv_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/data.csv"))
        assert decision.allow

    def test_write_txt_allowed(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/log.txt"))
        assert decision.allow

    def test_write_no_extension_allowed(self, provider):
        # Files without extension are typically data/config — allow
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/README"))
        assert decision.allow

    def test_write_empty_path_allowed(self, provider):
        # Defensive: if path missing in tool_input, don't crash
        decision = provider.evaluate(_req("write_file"))
        assert decision.allow


class TestBashAllowList:
    """Lead bash whitelist: ethoinsight.parse / ethoinsight.catalog / safe file ops."""

    def test_dump_headers_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.parse.dump_headers --input /mnt/user-data/uploads/x.txt --output /mnt/user-data/workspace/columns.json",
        ))
        assert decision.allow

    def test_catalog_resolve_allowed(self, provider):
        decision = provider.evaluate(_req(
            "bash",
            command="python -m ethoinsight.catalog.resolve --paradigm epm --columns-file /mnt/user-data/workspace/columns.json --output /mnt/user-data/workspace/metric_plan.json",
        ))
        assert decision.allow

    def test_mkdir_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mkdir -p /mnt/user-data/workspace/outputs"))
        assert decision.allow

    def test_cp_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cp /mnt/user-data/uploads/a.txt /mnt/user-data/workspace/a.txt"))
        assert decision.allow

    def test_mv_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="mv /tmp/a /tmp/b"))
        assert decision.allow

    def test_ls_bash_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="ls /mnt/user-data/uploads/"))
        assert decision.allow

    def test_cat_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="cat /mnt/user-data/workspace/metric_plan.json"))
        assert decision.allow

    def test_grep_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="grep -r 'open_arm' /mnt/user-data/workspace/"))
        assert decision.allow

    def test_head_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="head -10 /mnt/user-data/uploads/data.txt"))
        assert decision.allow

    def test_tail_allowed(self, provider):
        decision = provider.evaluate(_req("bash", command="tail -10 /mnt/user-data/workspace/log.txt"))
        assert decision.allow

    def test_leading_whitespace_allowed(self, provider):
        # Tolerate leading whitespace (lead sometimes emits indented commands)
        decision = provider.evaluate(_req("bash", command="  python -m ethoinsight.parse.dump_headers"))
        assert decision.allow


class TestBashDenyList:
    """Lead bash deny: arbitrary scripts, python -c, pip, etc."""

    def test_python_c_denied(self, provider):
        decision = provider.evaluate(_req("bash", command='python -c "print(1)"'))
        assert not decision.allow
        assert decision.reasons
        assert decision.reasons[0].code == "lead_execution_boundary.bash_not_allowed"

    def test_python_run_script_denied(self, provider):
        # Lead's exact failure mode in thread b0d3a611
        decision = provider.evaluate(_req("bash", command="python3 /mnt/user-data/workspace/gen_charts.py"))
        assert not decision.allow

    def test_python_heredoc_denied(self, provider):
        # Heredoc form (also lead's failure mode)
        decision = provider.evaluate(_req("bash", command="python3 << 'PYEOF'\nimport pandas\nPYEOF"))
        assert not decision.allow

    def test_python_dash_m_other_module_denied(self, provider):
        # Only ethoinsight.parse and ethoinsight.catalog are allowed
        decision = provider.evaluate(_req("bash", command="python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio"))
        assert not decision.allow

    def test_python_dash_m_pip_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="python -m pip install pandas"))
        assert not decision.allow

    def test_pip_install_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="pip install pandas"))
        assert not decision.allow

    def test_rm_denied(self, provider):
        # rm is destructive, not in safe-op whitelist
        decision = provider.evaluate(_req("bash", command="rm -rf /mnt/user-data/workspace/"))
        assert not decision.allow

    def test_curl_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="curl https://example.com"))
        assert not decision.allow

    def test_git_denied(self, provider):
        decision = provider.evaluate(_req("bash", command="git status"))
        assert not decision.allow

    def test_empty_command_denied(self, provider):
        # Defensive: empty bash command should not be allowed silently
        decision = provider.evaluate(_req("bash", command=""))
        assert not decision.allow


class TestDenyReasonContent:
    """Deny reasons should guide the agent to the correct path."""

    def test_write_file_reason_mentions_redispatch_path(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "metric_plan.json" in msg or "code-executor" in msg
        assert "ask_clarification" in msg

    def test_bash_reason_mentions_allowed_modules(self, provider):
        decision = provider.evaluate(_req("bash", command="python /tmp/x.py"))
        assert not decision.allow
        msg = decision.reasons[0].message
        assert "ethoinsight.parse" in msg
        assert "ethoinsight.catalog" in msg


class TestPolicyId:
    """policy_id is stable for log aggregation."""

    def test_write_file_policy_id(self, provider):
        decision = provider.evaluate(_req("write_file", path="/mnt/user-data/workspace/x.py"))
        assert decision.policy_id == "lead_execution_boundary"

    def test_bash_policy_id(self, provider):
        decision = provider.evaluate(_req("bash", command="curl x"))
        assert decision.policy_id == "lead_execution_boundary"


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate_on_deny(provider):
    req = _req("write_file", path="/mnt/user-data/workspace/x.py")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
    assert sync_decision.reasons[0].code == async_decision.reasons[0].code
    assert sync_decision.policy_id == async_decision.policy_id


@pytest.mark.asyncio
async def test_aevaluate_matches_evaluate_on_allow(provider):
    req = _req("write_file", path="/mnt/user-data/workspace/x.md")
    sync_decision = provider.evaluate(req)
    async_decision = await provider.aevaluate(req)
    assert sync_decision.allow == async_decision.allow
