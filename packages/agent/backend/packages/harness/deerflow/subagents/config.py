"""Subagent configuration definitions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig


@dataclass
class SubagentConfig:
    """Configuration for a subagent.

    Attributes:
        name: Unique identifier for the subagent.
        description: When Claude should delegate to this subagent.
        system_prompt: The system prompt that guides the subagent's behavior.
        tools: Optional list of tool names to allow. If None, inherits all tools.
        disallowed_tools: Optional list of tool names to deny.
        skills: Optional list of skill names to load. If None, inherits all enabled skills.
                If an empty list, no skills are loaded.
        model: Model to use - 'inherit' uses parent's model.
        max_turns: Maximum number of agent turns before stopping.
        timeout_seconds: Maximum execution time in seconds (default: 900 = 15 minutes).
    """

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900
    # ---- W1 capability metadata 新增 ----
    when_to_use: str | None = None
    input_contract: str | None = None
    output_contract: str | None = None
    required_upstream_handoffs: list[str] = field(default_factory=list)


def format_subagent_capability(config: "SubagentConfig") -> str:
    """Render name/description/when_to_use/input_contract/output_contract as Markdown.

    required_upstream_handoffs and system_prompt are intentionally excluded —
    they are harness-internal and must not appear in the lead prompt (spec §3.2).
    """

    def _or_placeholder(v: str | None) -> str:
        return v.strip() if v else "(未声明)"

    return (
        f"### {config.name}\n"
        f"- description: {_or_placeholder(config.description)}\n"
        f"- when_to_use: {_or_placeholder(config.when_to_use)}\n"
        f"- input_contract: {_or_placeholder(config.input_contract)}\n"
        f"- output_contract: {_or_placeholder(config.output_contract)}\n"
    )


def validate_subagent_handoff_refs(
    configs: dict[str, "SubagentConfig"],
    registry: dict[str, str],
) -> None:
    """Fail-fast: every required_upstream_handoffs entry must be a key in HANDOFF_FILE_REGISTRY."""
    for sub_name, cfg in configs.items():
        for upstream in cfg.required_upstream_handoffs:
            if upstream not in registry:
                raise ValueError(
                    f"Subagent '{sub_name}' references unknown upstream handoff '{upstream}'. "
                    f"Known: {sorted(registry)}"
                )


def _default_model_name(app_config: "AppConfig") -> str:
    if not app_config.models:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")
    return app_config.models[0].name


def resolve_subagent_model_name(config: SubagentConfig, parent_model: str | None, *, app_config: "AppConfig | None" = None) -> str:
    """Resolve the effective model name a subagent should use."""
    if config.model != "inherit":
        return config.model

    if parent_model is not None:
        return parent_model

    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()
    return _default_model_name(app_config)
