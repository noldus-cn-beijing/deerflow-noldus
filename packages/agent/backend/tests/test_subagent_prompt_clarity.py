"""Acceptance tests for subagent prompt clarity.

Commit 4 — §5-L3 of the EthoInsight pipeline redesign requires each subagent
system prompt to explicitly state:
1. Input format (with concrete path or schema reference)
2. Output format (with schema / file location)
3. Failure handling (how to return status=failed instead of silent bypass)

These tests lock in the presence of those sections so future prompt edits
cannot drop them silently.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Same mock pattern used by test_ethoinsight_analysis_skill.py to avoid circular imports.
_executor_mock = MagicMock()
_executor_mock.SubagentExecutor = MagicMock
_executor_mock.SubagentResult = MagicMock
_executor_mock.SubagentStatus = MagicMock
_executor_mock.MAX_CONCURRENT_SUBAGENTS = 3
sys.modules.setdefault("deerflow.subagents.executor", _executor_mock)

from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG  # noqa: E402
from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG  # noqa: E402
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG  # noqa: E402


class TestCodeExecutorPromptClarity:
    @property
    def prompt(self) -> str:
        return CODE_EXECUTOR_CONFIG.system_prompt or ""

    def test_describes_handoff_schema(self):
        assert "handoff" in self.prompt.lower()
        assert "status" in self.prompt
        assert "metrics_summary" in self.prompt
        assert "data_quality_warnings" in self.prompt

    def test_references_handoff_schema_module(self):
        assert "CodeExecutorHandoff" in self.prompt

    def test_has_failure_section(self):
        assert "<failure>" in self.prompt
        assert "failed" in self.prompt

    def test_forbids_silent_bypass(self):
        """Must tell subagent NOT to silently hard-write results."""
        p = self.prompt.lower()
        assert "不要硬写" in self.prompt or "bypass" in p


class TestDataAnalystPromptClarity:
    @property
    def prompt(self) -> str:
        return DATA_ANALYST_CONFIG.system_prompt or ""

    def test_declares_input_contract(self):
        # Primary input is now the code-executor handoff; code_summary.json
        # remains as a secondary/fallback reference.
        assert "handoff_code_executor.json" in self.prompt
        assert "per_subject" in self.prompt

    def test_declares_handoff_output(self):
        # Data-analyst must write a structured handoff JSON — it's the single
        # interface to report-writer and the lead agent, replacing the old
        # free-form analysis_summary.md.
        assert "handoff_data_analyst.json" in self.prompt
        # Ensure the old markdown output path is NOT re-introduced.
        assert "analysis_summary.md" not in self.prompt
        assert "analysis_report.md" not in self.prompt

    def test_has_failure_section(self):
        assert "<failure>" in self.prompt
        # Must refer to lead-agent follow-up, not silent "best-effort" analysis.
        assert "lead" in self.prompt.lower()

    def test_forbids_fake_analysis(self):
        assert "不要硬编造" in self.prompt or "不要基于猜测" in self.prompt


class TestReportWriterPromptClarity:
    @property
    def prompt(self) -> str:
        return REPORT_WRITER_CONFIG.system_prompt or ""

    def test_declares_input_contract(self):
        # Report-writer now reads two handoff JSONs, not analysis_summary.md.
        assert "handoff_code_executor.json" in self.prompt
        assert "handoff_data_analyst.json" in self.prompt
        assert "analysis_summary.md" not in self.prompt

    def test_declares_handoff_output(self):
        assert "handoff_report_writer.json" in self.prompt

    def test_mentions_write_file_chunking(self):
        """Report is typically 5-15K chars and write_file caps at 8000."""
        assert "8000" in self.prompt
        assert "append" in self.prompt

    def test_has_failure_section(self):
        assert "<failure>" in self.prompt
