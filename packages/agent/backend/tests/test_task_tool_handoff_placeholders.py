"""Tests for {{handoff://<name>}} placeholder resolution in task_tool.

Placeholder serves two roles:
1. Replace {{handoff://code_executor}} → full workspace path
2. Collect authorized paths for HandoffIsolationProvider
"""

import pytest


def test_resolve_handoff_placeholder_basic():
    """Single placeholder resolves to full path and returns authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "请分析 {{handoff://code_executor}} 中的数据"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert "/mnt/user-data/workspace/handoff_code_executor.json" in resolved
    assert "{{handoff://" not in resolved
    assert authorized == {"/mnt/user-data/workspace/handoff_code_executor.json"}


def test_resolve_handoff_placeholder_multiple():
    """Multiple placeholders all resolved + all collected in authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = (
        "结合 {{handoff://code_executor}} 的数据和 "
        "{{handoff://data_analyst}} 的解读回答"
    )
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert "handoff_code_executor.json" in resolved
    assert "handoff_data_analyst.json" in resolved
    assert authorized == {
        "/mnt/user-data/workspace/handoff_code_executor.json",
        "/mnt/user-data/workspace/handoff_data_analyst.json",
    }


def test_resolve_handoff_placeholder_no_placeholder():
    """Prompt without placeholders: unchanged prompt + empty authorized set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "这是一个不含占位符的普通 prompt"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    assert resolved == prompt
    assert authorized == set()


def test_resolve_handoff_placeholder_unknown_name_raises():
    """Unknown subagent name → ValueError (fail-fast on typo)."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    with pytest.raises(ValueError, match="Unknown handoff subagent 'foo'"):
        _resolve_handoff_placeholders("分析 {{handoff://foo}}")


def test_resolve_handoff_placeholder_duplicate_same_name():
    """Same placeholder appearing twice: both resolved, dedup'd in set."""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "{{handoff://code_executor}} 和再次 {{handoff://code_executor}}"
    resolved, authorized = _resolve_handoff_placeholders(prompt)

    # Both occurrences in prompt are replaced
    assert resolved.count("handoff_code_executor.json") == 2
    # Set dedup
    assert authorized == {"/mnt/user-data/workspace/handoff_code_executor.json"}


def test_registry_known_subagents():
    """HANDOFF_FILE_REGISTRY exposes the canonical mapping."""
    from deerflow.tools.builtins.task_tool import HANDOFF_FILE_REGISTRY

    assert HANDOFF_FILE_REGISTRY["code_executor"] == "handoff_code_executor.json"
    assert HANDOFF_FILE_REGISTRY["data_analyst"] == "handoff_data_analyst.json"
    assert HANDOFF_FILE_REGISTRY["report_writer"] == "handoff_report_writer.json"
    assert HANDOFF_FILE_REGISTRY["planning"] == "handoff_planning.json"
