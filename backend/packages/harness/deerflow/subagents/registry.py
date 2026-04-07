"""Subagent registry for managing available subagents."""

import logging
from dataclasses import replace

from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


def get_subagent_config(name: str) -> SubagentConfig | None:
    """Get a subagent configuration by name, with config.yaml overrides applied.

    Args:
        name: The name of the subagent.

    Returns:
        SubagentConfig if found (with any config.yaml overrides applied), None otherwise.
    """
    config = BUILTIN_SUBAGENTS.get(name)
    if config is None:
        return None

    # Apply per-agent timeout override from config.yaml (lazy import to avoid circular deps).
    # Only override if config.yaml has an explicit per-agent entry; the global default
    # should NOT replace the code-level SubagentConfig.timeout_seconds.
    from deerflow.config.subagents_config import get_subagents_app_config

    app_config = get_subagents_app_config()
    override = app_config.agents.get(name)
    if override is not None and override.timeout_seconds is not None:
        if override.timeout_seconds != config.timeout_seconds:
            logger.debug(f"Subagent '{name}': timeout overridden by config.yaml ({config.timeout_seconds}s -> {override.timeout_seconds}s)")
            config = replace(config, timeout_seconds=override.timeout_seconds)

    return config


def list_subagents() -> list[SubagentConfig]:
    """List all available subagent configurations (with config.yaml overrides applied).

    Returns:
        List of all registered SubagentConfig instances.
    """
    return [get_subagent_config(name) for name in BUILTIN_SUBAGENTS]


def get_subagent_names() -> list[str]:
    """Get all available subagent names.

    Returns:
        List of subagent names.
    """
    return list(BUILTIN_SUBAGENTS.keys())


def get_available_subagent_names() -> list[str]:
    """Get subagent names that should be exposed to the active runtime.

    Returns:
        List of subagent names visible to the current sandbox configuration.
    """
    names = list(BUILTIN_SUBAGENTS.keys())
    try:
        host_bash_allowed = is_host_bash_allowed()
    except Exception:
        logger.debug("Could not determine host bash availability; exposing all built-in subagents")
        return names

    if not host_bash_allowed:
        names = [name for name in names if name != "bash"]
    return names
