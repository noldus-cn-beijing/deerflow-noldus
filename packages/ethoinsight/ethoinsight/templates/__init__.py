"""Analysis templates for behavioral data paradigms.

Public API:
- get_analysis_template_tool: LangChain tool for code-executor subagent (legacy)
- run_paradigm_analysis_tool: LangChain tool that executes analysis in one call
- run_paradigm_analysis_core: Pure function for testing (no langchain dependency)
- get_available_paradigms: list available paradigm template names
- render_template: render a template with given parameters (used internally)
"""

from .tool import (
    get_analysis_template_tool,
    get_available_paradigms,
    render_template,
    run_paradigm_analysis_core,
    run_paradigm_analysis_tool,
)

__all__ = [
    "get_analysis_template_tool",
    "get_available_paradigms",
    "render_template",
    "run_paradigm_analysis_core",
    "run_paradigm_analysis_tool",
]
