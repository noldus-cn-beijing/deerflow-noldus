"""Configuration for loop detection middleware."""

from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


class ToolFreqOverride(BaseModel):
    """Per-tool frequency threshold override.

    Can be higher or lower than the global defaults. Commonly used to raise
    thresholds for high-frequency tools like bash in batch workflows (e.g.
    RNA-seq pipelines) without weakening protection on every other tool.
    """

    warn: int = Field(ge=1)
    hard_limit: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate(self) -> "ToolFreqOverride":
        if self.hard_limit < self.warn:
            raise ValueError("hard_limit must be >= warn")
        return self


class LoopDetectionConfig(BaseModel):
    """Configuration for repetitive tool-call loop detection."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable repetitive tool-call loop detection",
    )
    warn_threshold: int = Field(
        default=3,
        ge=1,
        description="Number of identical tool-call sets before injecting a warning",
    )
    hard_limit: int = Field(
        default=5,
        ge=1,
        description="Number of identical tool-call sets before forcing a stop",
    )
    window_size: int = Field(
        default=20,
        ge=1,
        description="Number of recent tool-call sets to track per thread",
    )
    max_tracked_threads: int = Field(
        default=100,
        ge=1,
        description="Maximum number of thread histories to keep in memory",
    )
    tool_freq_warn: int = Field(
        default=30,
        ge=1,
        description="Number of calls to the same tool type before injecting a frequency warning",
    )
    tool_freq_hard_limit: int = Field(
        default=50,
        ge=1,
        description="Number of calls to the same tool type before forcing a stop",
    )
    tool_freq_overrides: dict[str, ToolFreqOverride] = Field(
        default_factory=dict,
        description=("Per-tool overrides for tool_freq_warn / tool_freq_hard_limit, keyed by tool name. Values can be higher or lower than the global defaults. Commonly used to raise thresholds for high-frequency tools like bash."),
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "LoopDetectionConfig":
        """Ensure hard stop cannot happen before the warning threshold."""
        if self.hard_limit < self.warn_threshold:
            raise ValueError("hard_limit must be greater than or equal to warn_threshold")
        if self.tool_freq_hard_limit < self.tool_freq_warn:
            raise ValueError("tool_freq_hard_limit must be greater than or equal to tool_freq_warn")
        self._apply_semantic_overrides()
        return self

    # Bookkeeping / orchestration tools where high call counts are normal in a long
    # E2E (红线四 正模式 1). These thresholds are *floors*: a caller may raise them
    # further, but may not silently tighten them below the floor — that would
    # re-introduce the 2026-06-17 dogfood failure (write_todos killed the E2E).
    _SEMANTIC_OVERRIDE_FLOORS: ClassVar[dict[str, tuple[int, int]]] = {
        "write_todos": (15, 30),
    }

    def _apply_semantic_override(self, name: str, floor_warn: int, floor_hard: int) -> None:
        existing = self.tool_freq_overrides.get(name)
        if existing is None:
            self.tool_freq_overrides[name] = ToolFreqOverride(warn=floor_warn, hard_limit=floor_hard)
            return
        # Floor: keep the more lenient (higher) of caller vs semantic for each field.
        self.tool_freq_overrides[name] = ToolFreqOverride(
            warn=max(existing.warn, floor_warn),
            hard_limit=max(existing.hard_limit, floor_hard),
        )

    def _apply_semantic_overrides(self) -> None:
        for name, (warn, hard) in self._SEMANTIC_OVERRIDE_FLOORS.items():
            self._apply_semantic_override(name, warn, hard)
