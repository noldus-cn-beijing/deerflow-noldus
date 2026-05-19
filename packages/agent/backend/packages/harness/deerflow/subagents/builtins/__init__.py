"""Built-in subagent configurations."""

from .bash_agent import BASH_AGENT_CONFIG
from .code_executor import CODE_EXECUTOR_CONFIG
from .data_analyst import DATA_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG
from .report_writer import REPORT_WRITER_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_EXECUTOR_CONFIG",
    "DATA_ANALYST_CONFIG",
    "KNOWLEDGE_ASSISTANT_CONFIG",
    "REPORT_WRITER_CONFIG",
]

# Registry of built-in subagents
# EthoInsight uses dedicated subagents; DeerFlow defaults disabled
BUILTIN_SUBAGENTS = {
    "code-executor": CODE_EXECUTOR_CONFIG,
    "data-analyst": DATA_ANALYST_CONFIG,
    "report-writer": REPORT_WRITER_CONFIG,
    "knowledge-assistant": KNOWLEDGE_ASSISTANT_CONFIG,
}

# Fail-fast import-time validation: every required_upstream_handoffs entry must be
# a known key in HANDOFF_FILE_REGISTRY. Catches typos immediately on module import.
# Imports from handoff_registry (a zero-dependency thin module) to avoid the
# circular import chain that goes through task_tool -> deerflow.subagents.
from deerflow.subagents.config import validate_subagent_handoff_refs
from deerflow.subagents.handoff_registry import HANDOFF_FILE_REGISTRY

validate_subagent_handoff_refs(BUILTIN_SUBAGENTS, HANDOFF_FILE_REGISTRY)



