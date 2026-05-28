"""Pydantic schemas for subagent handoff JSON contracts.

These schemas define the contract between subagents (code-executor, data-analyst,
report-writer) and the lead agent. Lead validates handoff files against these
schemas to detect contract failures clearly, rather than absorbing malformed
output silently.

See docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md section 5 (L1 —
Handoff contract enforcement).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Virtual path contract: all subagent handoff JSON files must reference user-data
# files via virtual paths (e.g. /mnt/user-data/uploads/x.txt), never host absolute
# paths (e.g. /home/.../user-data/uploads/x.txt). Downstream consumers like
# chart-maker pass these paths to sandbox-enforced CLIs, which reject host paths.
#
# This invariant is enforced at the schema level so that a subagent leaking a host
# path via Path.resolve() fails fast at handoff parse time rather than silently
# propagating through the pipeline (see project_2026-05-26 path-pollution-defense).
_VIRTUAL_USER_DATA_PREFIX = "/mnt/user-data/"


def _validate_virtual_user_data_paths(paths: list[str]) -> list[str]:
    """Validate that every path starts with /mnt/user-data/ (the sandbox virtual prefix).

    Empty list is accepted (handoff may legitimately reference no raw files).
    Raises ValueError listing every offender so the subagent can fix them all at once.
    """
    if not paths:
        return paths
    offenders = [p for p in paths if not (isinstance(p, str) and p.startswith(_VIRTUAL_USER_DATA_PREFIX))]
    if offenders:
        raise ValueError(
            f"raw_files must use virtual paths under {_VIRTUAL_USER_DATA_PREFIX!r}, "
            f"got host-side or malformed entries: {offenders}. "
            "Copy paths verbatim from plan_metrics.json; do not run Path.resolve() / realpath."
        )
    return paths


class MetricStat(BaseModel):
    """Summary statistics for a single metric within a single group.

    Allow arbitrary extra fields from ethoinsight so new metric-level statistics
    (e.g. median, IQR) do not break older handoff files.
    """

    model_config = ConfigDict(extra="allow")

    mean: float | None = None
    std: float | None = None
    n: int | None = None
    applicable: bool = Field(
        default=True,
        description="False when the metric is not applicable (e.g. group metric on single-subject file).",
    )
    reason: str | None = Field(
        default=None,
        description="Present when applicable=False; explains why.",
    )
    parameters_used: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description=(
            "Actual parameters used to compute this metric, e.g. "
            "{'velocity_threshold': 30.0, 'velocity_min_duration': 25}. "
            "Populated by Sprint 2b execution pipeline. "
            "Sprint 0 only defines the field; defaults to empty dict. "
            "None is allowed for individual values when the param is not "
            "applicable to this metric (e.g. pendulum params on EPM)."
        ),
    )


class DataQualityWarning(BaseModel):
    """Single quality warning attached to a handoff."""

    severity: Literal["critical", "warning", "info"]
    metric: str = Field(
        description="Metric name, or 'all' / 'pipeline' when warning applies broadly.",
    )
    message: str
    code: str = Field(
        description=(
            "Warning code in dotted form, e.g. 'SAMPLE.TOO_SMALL', 'MOTOR.LOW_VELOCITY'. "
            "First segment must be one of: SAMPLE / MOTOR / SIGNAL / METHOD / LEGACY. "
            "LEGACY is a Sprint 0/1 transition whitelist entry; remove after Sprint 1. "
            "See Sprint 1 spec for full code taxonomy."
        ),
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured numeric evidence, e.g. "
            "{'velocity_median_mm_s': 5.2, 'threshold_mm_s': 30.0}. "
            "Frontend / data-analyst should not parse `message` for numbers."
        ),
    )
    blocks_downstream: bool = Field(
        default=False,
        description=(
            "True = downstream subagents (data-analyst, chart-maker, report-writer) "
            "should not be dispatched without user acknowledgement. "
            "Used by Sprint 5 DataQualityGuardrailProvider (manual mode) "
            "and Sprint 1 frontend (red vs orange rendering)."
        ),
    )

    @field_validator("code")
    @classmethod
    def _validate_code_namespace(cls, v: str) -> str:
        allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD", "LEGACY"}
        head = v.split(".", 1)[0] if "." in v else ""
        if head not in allowed:
            raise ValueError(
                f"warning code must use literal DOT '.' as separator, e.g. "
                f"'SAMPLE.TOO_SMALL', 'MOTOR.LOW_VELOCITY', 'SIGNAL.TRACKING_LOST', "
                f"'METHOD.SHAPIRO_INAPPLICABLE'. NOT underscore-separated ('SAMPLE_X' is INVALID). "
                f"First segment must be one of {sorted(allowed)} followed by '.'. "
                f"Got {v!r}."
            )
        return v


class GateSignals(BaseModel):
    """Structured decision signals from subagent to lead.

    Lead reads these from subagent's final AIMessage (not from the handoff
    file itself) to make Step 1.5 quality-gate decisions without inflating
    context with the full handoff JSON. Also persisted in handoff JSON for
    audit/replay (optional field).
    """

    model_config = ConfigDict(extra="allow")

    data_quality: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Summary of data_quality_warnings: "
            "{'critical_count': int, 'warning_count': int, "
            "'critical_items': [str, ...]  # 关键 critical 条目摘要，每条 <80 字}"
        ),
    )
    statistical_validity: Literal["ok", "warning", "failed", "skipped"] = "ok"
    errors_count: int = 0


class CodeExecutorInputs(BaseModel):
    """Inputs block recorded by code-executor (raw files + grouping + columns).

    Subagent must copy raw_files verbatim from plan_metrics.json.inputs.raw_files;
    field_validator enforces that paths stay virtual.
    """

    model_config = ConfigDict(extra="allow")

    raw_files: list[str] = Field(
        default_factory=list,
        description="Virtual paths (/mnt/user-data/uploads/...) of input trajectory files.",
    )
    groups: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional group assignment dict, e.g. {'Arena 1': 'Treatment'}.",
    )

    @field_validator("raw_files")
    @classmethod
    def _enforce_virtual_paths(cls, v: list[str]) -> list[str]:
        return _validate_virtual_user_data_paths(v)


class CodeExecutorHandoff(BaseModel):
    """Handoff JSON produced by the code-executor subagent (SOTA glue-script architecture).

    Written by the glue script to handoff_code_executor.json in the workspace.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"]
    summary: str
    inputs: CodeExecutorInputs | None = Field(
        default=None,
        description=(
            "Inputs block (raw_files, groups, ...). Optional for backwards compatibility "
            "with older handoff files; new code-executor invocations should populate it."
        ),
    )
    output_files: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of category (metrics/statistics/charts) to path(s).",
    )
    metrics_summary: dict[str, dict[str, MetricStat]] = Field(
        default_factory=dict,
        description="Nested map: group -> metric -> stats.",
    )
    per_subject: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Raw per-subject metric values: {subject_name: {metric: value}}. "
            "Downstream data-analyst uses this to identify outlier subjects by "
            "name and compute leave-one-out counterfactual group statistics."
        ),
    )
    statistics: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    data_quality_warnings: list[DataQualityWarning] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional overall confidence score in [0,1].",
    )
    gate_signals: GateSignals | None = Field(
        default=None,
        description=(
            "Structured signals for lead's decision-making. "
            "Optional in JSON file (lead reads from final AIMessage instead), "
            "but recommended to include for audit/replay."
        ),
    )
    paradigm: str = Field(
        description=(
            "Experiment paradigm (e.g. 'fst', 'epm'). Redundant with "
            "experiment-context.json so handoff is self-contained for replay."
        ),
    )
    ev19_template: str | None = Field(
        default=None,
        description="EV19 template ID, or None for paradigms not mapped to EV19.",
    )
    analysis_config_id: str = Field(
        description=(
            "16-char hex hash of (catalog_default + parameter_overrides). "
            "Populated by seal_code_executor_handoff tool (Sprint 4.5)."
        ),
    )


class FailedChart(BaseModel):
    """One failed chart entry."""

    chart_id: str = Field(description="Chart ID from catalog, e.g. 'trajectory_heatmap'.")
    reason: str = Field(description="Free-text failure reason.")


class ChartMakerHandoff(BaseModel):
    """Handoff JSON produced by chart-maker subagent.

    Fields align with the schema currently declared in chart-maker subagent prompt
    (subagents/builtins/chart_maker.py <handoff_schema> section).
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"] = "completed"
    paradigm: str = Field(description="Experiment paradigm.")
    chart_files: list[str] = Field(
        default_factory=list,
        description="Virtual paths under /mnt/user-data/outputs/.",
    )
    failed_charts: list[FailedChart] = Field(default_factory=list)
    summary: str = Field(description="One-liner describing generated charts.")
    gate_signals: GateSignals | None = None
    analysis_config_id: str = Field(description="Inherited from CodeExecutorHandoff.")

    @field_validator("chart_files")
    @classmethod
    def _validate_chart_paths(cls, v: list[str]) -> list[str]:
        if not v:
            return v
        prefix = "/mnt/user-data/outputs/"
        offenders = [p for p in v if not p.startswith(prefix)]
        if offenders:
            raise ValueError(
                f"chart_files must be under {prefix!r}, "
                f"got: {offenders}. Outputs must be in outputs/, not workspace/."
            )
        return v


class OutlierFinding(BaseModel):
    """One flagged outlier subject with counterfactual support.

    Used by data-analyst to surface per-subject diagnostics: which subject
    deviates on which metric, by how much, and what the group statistics
    look like with that subject excluded.
    """

    model_config = ConfigDict(extra="allow")

    subject: str = Field(description="Subject identifier, e.g. 'Subject 3'.")
    metric: str = Field(description="Metric on which this subject is an outlier.")
    value: float = Field(description="Raw per-subject value on that metric.")
    deviation: str = Field(
        description=(
            "Qualitative description of deviation, e.g. '2x group median', "
            "'CV=35%', '> 1.5 SD above mean'."
        ),
    )
    counterfactual: str | None = Field(
        default=None,
        description=(
            "Leave-one-out group stats if this subject is excluded, e.g. "
            "'treatment mean_nnd drops 48.2 → 37.2 mm if Subject 3 excluded'."
        ),
    )


class DataAnalystHandoff(BaseModel):
    """Handoff JSON produced by the data-analyst subagent.

    Structured return type so downstream consumers (report-writer, lead agent
    rendering) can act on the analyst's findings without re-parsing natural
    language. This is the single source of truth for data-analyst output —
    the subagent writes nothing else to disk.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "failed"]
    key_findings: list[str] = Field(
        default_factory=list,
        description="1-5 bullet findings surfaced to the user.",
    )
    outlier_findings: list[OutlierFinding] = Field(
        default_factory=list,
        description=(
            "Per-subject outlier diagnostics with leave-one-out counterfactual "
            "context. Empty list when no outlier is flagged."
        ),
    )
    excluded_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics skipped due to applicability or data-quality concerns.",
    )
    method_warnings: list[str] = Field(
        default_factory=list,
        description="Statistical-method concerns raised during quality review.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Suggested next steps or follow-up analyses.",
    )
    errors: list[str] = Field(default_factory=list)
    gate_signals: GateSignals | None = Field(default=None)
    analysis_config_id: str = Field(description="Inherited from CodeExecutorHandoff.")


class ReportWriterHandoff(BaseModel):
    """Handoff JSON produced by the report-writer subagent."""

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "failed"]
    report_path: str
    sections_written: list[str] = Field(
        default_factory=list,
        description="E.g. ['Results', 'Discussion'].",
    )
    errors: list[str] = Field(default_factory=list)
    gate_signals: GateSignals | None = Field(default=None)
    analysis_config_id: str = Field(description="Inherited from CodeExecutorHandoff.")


__all__ = [
    "ChartMakerHandoff",
    "CodeExecutorHandoff",
    "CodeExecutorInputs",
    "DataAnalystHandoff",
    "DataQualityWarning",
    "FailedChart",
    "GateSignals",
    "MetricStat",
    "OutlierFinding",
    "ReportWriterHandoff",
]
