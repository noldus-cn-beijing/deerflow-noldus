"""Built-in subagent configurations."""

from .bash_agent import BASH_AGENT_CONFIG
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_EXECUTOR_CONFIG",
    "DATA_ANALYST_CONFIG",
    "REPORT_WRITER_CONFIG",
]

# Registry of built-in subagents
BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
}
