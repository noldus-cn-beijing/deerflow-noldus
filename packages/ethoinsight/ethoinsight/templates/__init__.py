"""Analysis templates for behavioral data paradigms.

Public API:
- get_analysis_template_tool: LangChain tool for code-executor subagent
- get_available_paradigms: list available paradigm template names
- render_template: render a template with given parameters (used internally)
"""

from .tool import get_analysis_template_tool, get_available_paradigms, render_template

__all__ = ["get_analysis_template_tool", "get_available_paradigms", "render_template"]
