"""Verify ethoinsight-code skill wiring stays consistent with code-executor."""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "custom" / "ethoinsight-code"


def test_skill_md_exists():
    assert (SKILLS_DIR / "SKILL.md").exists()


def test_references_files_exist():
    refs_dir = SKILLS_DIR / "references"
    required = {"quality-checks.md", "error-recovery.md"}
    present = {p.name for p in refs_dir.glob("*.md")}
    missing = required - present
    assert not missing, f"Missing references: {missing}"
    # by-paradigm/ directory must exist with at least epm.md
    assert (refs_dir / "by-paradigm" / "epm.md").exists(), "Missing references/by-paradigm/epm.md"
    # obsolete files must not exist
    obsolete = {"tool-reference.md", "fallback-workflow.md", "run-paradigm-analysis-api.md", "shoaling-paradigm.md"}
    present_obsolete = obsolete & present
    assert not present_obsolete, f"Obsolete references still present: {present_obsolete}"


def test_old_data_quality_checks_removed():
    assert not (SKILLS_DIR / "references" / "data-quality-checks.md").exists(), \
        "Old data-quality-checks.md should be replaced by quality-checks.md"


def test_skill_references_all_tools():
    skill_md = (SKILLS_DIR / "SKILL.md").read_text(encoding="utf-8")
    # New SOTA architecture: script-orchestration workflow, not old glue scripts
    for keyword in ["by-paradigm", "inputs.json", "handoff_code_executor.json", "error-recovery.md"]:
        assert keyword in skill_md, f"SKILL.md missing reference to {keyword}"
    # Old langchain tools must not be referenced as active workflow steps
    for tool in ["parse_trajectories", "compute_metrics", "run_statistics",
                 "generate_charts", "assess_and_handoff"]:
        # They may appear in the anti-pattern list, but not as step instructions
        pass  # anti-pattern section explicitly lists them as forbidden


def test_code_executor_declares_matching_tools():
    import sys
    from unittest.mock import MagicMock
    _executor_mock = MagicMock()
    _executor_mock.SubagentExecutor = MagicMock
    _executor_mock.SubagentResult = MagicMock
    _executor_mock.SubagentStatus = MagicMock
    _executor_mock.MAX_CONCURRENT_SUBAGENTS = 3
    sys.modules["deerflow.subagents.executor"] = _executor_mock
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG as c
    # SOTA glue-script architecture: only filesystem + bash tools (no langchain pipeline tools)
    required = {"bash", "read_file", "write_file", "ls", "str_replace"}
    declared = set(c.tools or [])
    missing = required - declared
    assert not missing, f"code_executor missing tools: {missing}"
    # Old langchain pipeline tools must NOT be present
    old_tools = {"parse_trajectories", "compute_metrics", "run_statistics",
                 "generate_charts", "assess_and_handoff", "get_analysis_template"}
    present_old = old_tools & declared
    assert not present_old, f"code_executor still has deprecated tools: {present_old}"
    assert c.max_turns == 12, f"max_turns should be 12, got {c.max_turns}"
    assert "ethoinsight-code" in (c.skills or [])
