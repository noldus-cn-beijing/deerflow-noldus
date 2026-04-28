"""Each builtin subagent prompt must contain a user-language-matching rule
so handoff text, write_file content, and final messages do not leak
English into a Chinese conversation (or vice versa).
"""
from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def _assert_language_rule(prompt: str, subagent_name: str):
    # Positive phrasing: match user language
    assert "用户语言" in prompt, (
        f"{subagent_name} prompt missing user-language rule"
    )
    assert "与用户语言一致" in prompt or "匹配用户语言" in prompt, (
        f"{subagent_name} prompt missing explicit language-matching instruction"
    )


def test_code_executor_has_language_rule():
    _assert_language_rule(CODE_EXECUTOR_CONFIG.system_prompt, "code-executor")


def test_data_analyst_has_language_rule():
    _assert_language_rule(DATA_ANALYST_CONFIG.system_prompt, "data-analyst")


def test_report_writer_has_language_rule():
    _assert_language_rule(REPORT_WRITER_CONFIG.system_prompt, "report-writer")
