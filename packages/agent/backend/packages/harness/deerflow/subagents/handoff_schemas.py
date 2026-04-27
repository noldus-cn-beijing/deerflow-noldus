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

from pydantic import BaseModel, ConfigDict, Field


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


class DataQualityWarning(BaseModel):
    """Single quality warning attached to a handoff."""

    severity: Literal["critical", "warning", "info"]
    metric: str = Field(
        description="Metric name, or 'all' / 'pipeline' when warning applies broadly.",
    )
    message: str


class CodeExecutorHandoff(BaseModel):
    """Handoff JSON produced by the code-executor subagent's assess_and_handoff tool.

    Matches the shape written by ethoinsight.templates.tool.assess_and_handoff_tool.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["completed", "partial", "failed"]
    summary: str
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


__all__ = [
    "CodeExecutorHandoff",
    "DataAnalystHandoff",
    "DataQualityWarning",
    "MetricStat",
    "OutlierFinding",
    "ReportWriterHandoff",
]
