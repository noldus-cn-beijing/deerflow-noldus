"""Catalog 数据模型（dataclass）。

为什么不用 pydantic：保持 ethoinsight 库的依赖最小化（参见
pyproject.toml）。YAML 校验由 loader.py 手工做。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DirectionEnum = Literal["lower_is_anxious", "higher_is_anxious"] | None
StatDefault = Literal["groupwise_compare", "paired_compare"]
ChartCondition = str  # e.g. "always", "n_per_group >= 3"

ALLOWED_DIRECTIONS: frozenset[str | None] = frozenset({
    "lower_is_anxious", "higher_is_anxious", None,
})
ALLOWED_STAT_DEFAULTS: frozenset[str] = frozenset({
    "groupwise_compare", "paired_compare",
})


@dataclass(frozen=True)
class MetricEntry:
    id: str
    script: str
    requires_columns: list[str]
    output_unit: str
    display_name_zh: str
    unit_zh: str
    one_liner: str
    direction_for_anxiety: str | None  # validated against ALLOWED_DIRECTIONS
    statistical_default: str           # validated against ALLOWED_STAT_DEFAULTS


@dataclass(frozen=True)
class ChartEntry:
    id: str
    script: str
    when: ChartCondition  # "always" | "n_per_group >= K" | "n_groups >= K"


@dataclass(frozen=True)
class StatisticsEntry:
    id: str
    script: str
    when: ChartCondition


@dataclass(frozen=True)
class Catalog:
    paradigm: str
    ev19_templates: list[str]
    default_metrics: list[MetricEntry]
    optional_metrics: list[MetricEntry]
    charts: list[ChartEntry]
    statistics_default: StatisticsEntry | None


# ============================================================================
# Plan (输出结构) — metric_plan.json schema
# ============================================================================


PlanReasonEnum = Literal[
    "paradigm.default", "paradigm.required",
    "user.include", "paradigm.optional.applicable",
]
SkippedReasonEnum = Literal[
    "user.exclude", "columns.missing",
    "paradigm.not_applicable", "catalog.unknown",
]


@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str  # PlanReasonEnum


@dataclass
class PlanSkipped:
    id: str
    reason: str  # SkippedReasonEnum
    detail: str


@dataclass
class PlanStatistics:
    id: str
    script: str
    input: str
    output: str
    skip_reason: str | None  # None = 跑；非空字符串 = 跳过原因


@dataclass
class PlanChart:
    id: str
    script: str
    input: str
    output: str


@dataclass
class PlanInputs:
    raw_files: list[str]
    groups_file: str | None
    columns_file: str | None


@dataclass
class Plan:
    schema_version: str
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    metrics: list[PlanMetric]
    statistics: PlanStatistics | None
    charts: list[PlanChart]
    skipped: list[PlanSkipped]
    notes: list[str]
