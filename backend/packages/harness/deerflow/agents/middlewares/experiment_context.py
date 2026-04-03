"""Middleware to auto-extract EthoVision experiment context from uploaded files.

When EthoVision trajectory files are uploaded, this middleware parses their
headers to extract experiment metadata (paradigm, subjects, duration, etc.)
and injects it into the agent's context as an <experiment_context> block.
"""

import logging
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class ExperimentContextState(AgentState):
    """State schema for experiment context middleware."""

    uploaded_files: NotRequired[list[dict] | None]
    experiment_context: NotRequired[dict | None]


class ExperimentContextMiddleware(AgentMiddleware[ExperimentContextState]):
    """Auto-detects EthoVision files in uploads and extracts experiment metadata.

    Injects an <experiment_context> block into the last human message so the
    lead agent knows the paradigm, subjects, and duration before asking the user.
    Only triggers when new EthoVision files are detected.
    """

    state_schema = ExperimentContextState

    @override
    async def before_model(self, state: ExperimentContextState, **kwargs) -> ExperimentContextState:
        uploaded = state.get("uploaded_files") or []
        if not uploaded:
            return state

        # Check if any uploaded files look like EthoVision trajectory files
        ethovision_files = [
            f for f in uploaded
            if f.get("ethovision") or (f.get("name", "").startswith("轨迹-") and f.get("name", "").endswith(".txt"))
        ]

        if not ethovision_files:
            return state

        # Skip if context already extracted for these files
        existing_ctx = state.get("experiment_context")
        if existing_ctx and existing_ctx.get("files_hash") == _files_hash(ethovision_files):
            return state

        # Try to parse headers from one file to extract metadata
        context = _extract_context(ethovision_files)
        if not context:
            return state

        # Store in state
        state = {**state, "experiment_context": context}

        # Inject into last human message
        messages = list(state.get("messages", []))
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                context_block = _format_context_block(context)
                new_content = f"{context_block}\n\n{last_msg.content}"
                messages[-1] = HumanMessage(content=new_content, additional_kwargs=last_msg.additional_kwargs)
                state = {**state, "messages": messages}

        return state


def _files_hash(files: list[dict]) -> str:
    """Simple hash of file names for change detection."""
    names = sorted(f.get("name", "") for f in files)
    return "|".join(names)


def _extract_context(ethovision_files: list[dict]) -> dict | None:
    """Extract experiment context from EthoVision file headers.

    Uses ethoinsight.parse if available, otherwise returns basic info.
    """
    try:
        from ethoinsight.parse import parse_header
    except ImportError:
        logger.debug("ethoinsight not available, skipping context extraction")
        return None

    # Try parsing the first file
    first_file = ethovision_files[0]
    file_path = first_file.get("path", "")
    if not file_path or not Path(file_path).exists():
        return None

    try:
        header = parse_header(file_path)
    except Exception as e:
        logger.warning("Failed to parse EthoVision header: %s", e)
        return None

    # Collect subject names from all files
    subjects = set()
    for f in ethovision_files:
        p = f.get("path", "")
        if p and Path(p).exists():
            try:
                h = parse_header(p)
                subj = h.get("subject", "")
                if subj:
                    subjects.add(subj)
            except Exception:
                pass

    return {
        "experiment": header.get("experiment", ""),
        "paradigm": header.get("paradigm"),
        "trial_name": header.get("trial_name", ""),
        "subjects": sorted(subjects),
        "n_files": len(ethovision_files),
        "duration": header.get("duration", ""),
        "columns": header.get("columns", []),
        "files_hash": _files_hash(ethovision_files),
    }


def _format_context_block(context: dict) -> str:
    """Format experiment context as XML block for the agent."""
    lines = ["<experiment_context>"]
    lines.append(f"实验名称: {context.get('experiment', 'Unknown')}")
    paradigm = context.get("paradigm")
    if paradigm:
        lines.append(f"检测到范式: {paradigm}")
    subjects = context.get("subjects", [])
    if subjects:
        lines.append(f"受试对象 ({len(subjects)}): {', '.join(subjects)}")
    lines.append(f"轨迹文件数: {context.get('n_files', 0)}")
    duration = context.get("duration", "")
    if duration:
        lines.append(f"试验持续时间: {duration}")
    columns = context.get("columns", [])
    if columns:
        lines.append(f"数据列: {', '.join(columns[:15])}")
    lines.append("</experiment_context>")
    return "\n".join(lines)
