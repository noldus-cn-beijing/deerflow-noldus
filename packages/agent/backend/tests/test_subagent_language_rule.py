"""Each builtin subagent prompt must contain a language rule.

Spec 2026-06-25: 所有 4 个 subagent（code-executor / data-analyst / chart-maker /
report-writer）统一「跟随用户语言」策略，覆盖思考过程 + 所有输出（详见
test_subagent_thinking_language.py）。code-executor / chart-maker 不再单独用
旧「中文优先」措辞——改为与用户语言一致，与 data-analyst / report-writer 对齐。
"""
from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG
from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def _assert_language_rule(prompt: str, subagent_name: str):
    # Positive phrasing: match user language
    assert "<语言>" in prompt, f"{subagent_name} prompt missing <语言> section"
    assert "用户" in prompt and "语言" in prompt, (
        f"{subagent_name} prompt missing user-language rule"
    )
    assert "与用户" in prompt and "语言" in prompt, (
        f"{subagent_name} prompt missing explicit language-matching instruction"
    )


def test_code_executor_has_language_rule():
    """code-executor 跟随用户语言（思考 + 输出），不再用旧「中文优先」措辞。"""
    _assert_language_rule(CODE_EXECUTOR_CONFIG.system_prompt, "code-executor")


def test_chart_maker_has_language_rule():
    """chart-maker 跟随用户语言（思考 + 输出），不再用旧「中文优先」措辞。"""
    _assert_language_rule(CHART_MAKER_CONFIG.system_prompt, "chart-maker")


def test_data_analyst_has_language_rule():
    _assert_language_rule(DATA_ANALYST_CONFIG.system_prompt, "data-analyst")


def test_report_writer_has_language_rule():
    _assert_language_rule(REPORT_WRITER_CONFIG.system_prompt, "report-writer")
