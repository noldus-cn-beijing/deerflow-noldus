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

ALLOWED_DIRECTIONS: frozenset[str | None] = frozenset(
    {
        "lower_is_anxious",
        "higher_is_anxious",
        None,
    }
)
ALLOWED_STAT_DEFAULTS: frozenset[str] = frozenset(
    {
        "groupwise_compare",
        "paired_compare",
    }
)


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
    statistical_default: str  # validated against ALLOWED_STAT_DEFAULTS


@dataclass(frozen=True)
class ChartEntry:
    id: str
    script: str
    when: ChartCondition  # "always" | "n_per_group >= K" | "n_groups >= K"
    display_name_zh: str = ""          # 1.1: 中文图名，必填（loader 校验）
    accepts_paradigm: bool = False      # 1.1: 脚本是否接受 --paradigm 参数
    output_mode: str = "per_subject"   # 1.2: "per_subject" expands to N PlanCharts (one inputs.json per file);
                                         # "aggregate" collapses to 1 PlanChart with all files in one inputs.json
    needs_groups: bool = False          # 1.2: aggregate plots that compare across groups need a groups.json arg
    requires_columns: list[str] = field(default_factory=list)
    # 1.3: fnmatch glob 列名模式（如 "velocity"、"mobility_state*"、"in_zone*open*"）。
    # 任一 pattern 在 columns.json 中无列匹配则该 chart 被 resolve_charts 跳过并写入 skipped。
    # 默认 [] 表示"不依赖具体列"——兼容旧 yaml，但所有 audit 过的 catalog 都应显式声明。


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
    "paradigm.default",
    "paradigm.required",
    "user.include",
    "paradigm.optional.applicable",
]
SkippedReasonEnum = Literal[
    "user.exclude",
    "columns.missing",
    "paradigm.not_applicable",
    "catalog.unknown",
]


@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str  # PlanReasonEnum
    subject_index: int = 0  # 0-based index into inputs.raw_files; 0 for single-subject plans
    display_name_zh: str = ""           # 1.1: 中文指标名，透传自 MetricEntry


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
    subject_index: int = 0  # 0-based index into inputs.raw_files; 0 for single-subject plans
    display_name_zh: str = ""           # 1.1: 中文图名，透传自 ChartEntry
    args: list[str] = field(default_factory=list)  # 1.1: resolve 阶段填充的 CLI 参数数组


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


# ============================================================================
# Plan split (W2): 拆 Plan → PlanMetrics + PlanCharts
# 老 Plan dataclass 保留,等 W4/W11/W13 完成后 W22 dogfood 前彻底删。
# ============================================================================


@dataclass
class PlanMetrics:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    metrics: list[PlanMetric]
    statistics: PlanStatistics | None
    skipped: list[PlanSkipped]
    notes: list[str]
    schema_version: str = "1.1"


@dataclass
class PlanCharts:
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    charts: list[PlanChart]
    charts_fallback_available: list[PlanChart]
    skipped: list[PlanSkipped]
    user_intent: str | None
    notes: list[str]
    schema_version: str = "1.1"
