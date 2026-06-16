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
    # by-paradigm/ directory must NOT exist (knowledge moved to ethoinsight catalog YAML)
    assert not (refs_dir / "by-paradigm").exists(), "references/by-paradigm/ should have been removed (knowledge now in ethoinsight catalog YAML)"
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
    for keyword in ["plan_metrics.json", "handoff_code_executor.json", "error-recovery.md"]:
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
    # Spec S4: code-executor 工具粒度从裸 bash 改为 run_metric_plan 确定性工具。
    # bash/write_file/str_replace 已收走，run_metric_plan 是执行 metrics+stats 的唯一路径。
    required = {"run_metric_plan", "read_file", "ls", "seal_code_executor_handoff"}
    declared = set(c.tools or [])
    missing = required - declared
    assert not missing, f"code_executor missing tools: {missing}"
    # S4 收走的旧 bash 编排工具必须不在
    removed = {"bash", "write_file", "str_replace"}
    present_removed = removed & declared
    assert not present_removed, f"S4: code_executor should no longer have bash/write_file/str_replace: {present_removed}"
    # Old langchain pipeline tools must NOT be present
    old_tools = {"parse_trajectories", "compute_metrics", "run_statistics",
                 "generate_charts", "assess_and_handoff", "get_analysis_template"}
    present_old = old_tools & declared
    assert not present_old, f"code_executor still has deprecated tools: {present_old}"
    # S4: happy path 薄到 2-3 message，max_turns 从 40（bash 编排时代）降到 20。
    assert c.max_turns == 20, f"S4: max_turns should be 20, got {c.max_turns}"
    assert "ethoinsight-code" in (c.skills or [])
