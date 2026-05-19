"""Handoff file registry — standalone module with no heavy imports.

Extracted from task_tool.py so that deerflow.subagents.builtins can import it
at init time without triggering the circular import chain:
  subagents.builtins -> tools.builtins.task_tool -> deerflow.subagents (partially init)

Usage:
    from deerflow.subagents.handoff_registry import HANDOFF_FILE_REGISTRY
"""

# Mapping from subagent-name key (used in {{handoff://...}} placeholders) to the
# handoff filename placed in /mnt/user-data/workspace/.
# Authoritative single source: adding an entry here AND in task_tool enables the
# handoff mechanism; both imports re-export the same dict object.
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",
}
