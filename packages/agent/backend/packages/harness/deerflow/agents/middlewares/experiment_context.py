"""Read/write experiment-context.json for Gate state persistence.

TWO PATH DOMAINS:
1. Container-side (lead agent): /mnt/user-data/workspace/experiment-context.json
   - Lead agent calls set_experiment_paradigm tool (container path, sandbox translates)
   - Code-executor reads via read_file tool (container path, sandbox translates)

2. Host-side (middleware): {workspace_path}/experiment-context.json
   - GateEnforcementMiddleware reads host-side path from state["thread_data"]["workspace_path"]
   - This module provides the host-side read functions

Robustness: file-not-found returns None (never raises).
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadDataState, ThreadState

logger = logging.getLogger(__name__)

# Container-side path — used by lead agent prompt instructions and code-executor
CONTAINER_CONTEXT_PATH = "/mnt/user-data/workspace/experiment-context.json"


def read_context(workspace_dir: str) -> dict | None:
    """Read experiment-context.json from host-side workspace_dir. Returns None if absent."""
    path = Path(workspace_dir) / "experiment-context.json"
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, PermissionError, OSError) as e:
        logger.warning("Failed to read experiment-context.json: %s", e)
        return None


def context_exists(workspace_dir: str) -> bool:
    """Check if experiment-context.json exists in host-side workspace_dir."""
    return (Path(workspace_dir) / "experiment-context.json").exists()


def resolve_workspace_from_state(state: dict) -> str | None:
    """Extract host-side workspace path from agent state (set by ThreadDataMiddleware).

    Returns None if state lacks thread_data — caller should treat as auto mode or old thread.
    """
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        return None
    return thread_data.get("workspace_path")


@tool("set_experiment_paradigm", parse_docstring=True)
def set_experiment_paradigm_tool(
    paradigm: str,
    paradigm_cn: str,
    category: str,
    subject: str,
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's experiment paradigm choice for the analysis pipeline.

    Call this after the user has confirmed their experiment type via ask_clarification.
    Writes experiment-context.json to the workspace so downstream agents know the paradigm.

    Args:
        paradigm: English paradigm name key (e.g. "shoaling", "epm", "open_field")
        paradigm_cn: Chinese display name (e.g. "斑马鱼鱼群行为")
        category: Category name (e.g. "zebrafish", "anxiety", "spatial_memory")
        subject: Subject type — "rodent" | "fish" | "insect" | "other"
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/"

    Returns:
        JSON confirmation with paradigm, category, subject, and file path.
    """
    # Resolve the actual host workspace path from thread state.
    # The default workspace_dir is a sandbox virtual path; the tool runs in the
    # lead agent host process so we must write to the host-side workspace.
    actual_workspace = workspace_dir
    if runtime is not None and runtime.state is not None:
        thread_data: ThreadDataState | None = runtime.state.get("thread_data")
        if thread_data is not None:
            host_workspace = thread_data.get("workspace_path")
            if host_workspace is not None:
                actual_workspace = host_workspace

    data = {
        "paradigm": paradigm,
        "paradigm_cn": paradigm_cn,
        "category": category,
        "subject": subject,
        "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
        "gate_completed": ["gate1"],
    }
    path = Path(actual_workspace) / "experiment-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return json.dumps({"status": "ok", "path": str(path), "paradigm": paradigm}, ensure_ascii=False)
