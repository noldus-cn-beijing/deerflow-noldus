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


# ============================================================================
# Sprint 2a: parameter specs — catalog 端参数下沉
# ============================================================================


@dataclass(frozen=True)
class ParamSpec:
    """单个参数的定义。"""

    default: float | int | str
    unit: str
    description: str
    tunable_by_user: bool
    valid_range: list[float | int] | None  # [min, max] for numeric; None for str


@dataclass(frozen=True)
class SharedParameters:
    """跨范式共享的参数集合 (_common.yaml.shared_parameters)。"""

    parameters: dict[str, ParamSpec]


@dataclass(frozen=True)
class ParadigmParameters:
    """范式级共用参数 (各 <paradigm>.yaml 的 paradigm_parameters 段)。"""

    parameters: dict[str, ParamSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricEntry:
    id: str
    script: str
    requires_columns: list[str | list[str]]
    output_unit: str
    display_name_zh: str
    unit_zh: str
    one_liner: str
    direction_for_anxiety: str | None  # validated against ALLOWED_DIRECTIONS
    statistical_default: str  # validated against ALLOWED_STAT_DEFAULTS
    parameters: dict[str, ParamSpec] = field(default_factory=dict)
    parameters_ref: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChartEntry:
    id: str
    script: str
    when: ChartCondition  # "always" | "n_per_group >= K" | "n_groups >= K"
    display_name_zh: str = ""          # 1.1: 中文图名，必填（loader 校验）
    confidence: str = "optional"       # "must_have" | "optional" | "rarely_used"
                                       # must_have: 自动生成不询问；optional: 反问用户要不要出；
                                       # rarely_used: 用户主动提才生成
    accepts_paradigm: bool = False      # 1.1: 脚本是否接受 --paradigm 参数
    output_mode: str = "per_subject"   # 1.2: "per_subject" expands to N PlanCharts (one inputs.json per file);
                                         # "aggregate" collapses to 1 PlanChart with all files in one inputs.json
    needs_groups: bool = False          # 1.2: aggregate plots that compare across groups need a groups.json arg
    requires_columns: list[str | list[str]] = field(default_factory=list)
    # 1.3: fnmatch glob 列名模式（如 "velocity"、"mobility_state*"、"in_zone*open*"）。
    # 任一 pattern 在 columns.json 中无列匹配则该 chart 被 resolve_charts 跳过并写入 skipped。
    # 默认 [] 表示"不依赖具体列"——兼容旧 yaml，但所有 audit 过的 catalog 都应显式声明。


@dataclass(frozen=True)
class StatisticsEntry:
    id: str
    script: str
    when: ChartCondition


@dataclass(frozen=True)
class AnonymousZoneOverride:
    """Per-paradigm translation rule for unified anonymous zone key.

    When a paradigm declares this, the resolve layer:
    1. Accepts the unified key ``anonymous_zone_is`` (instead of
       paradigm-specific keys like center_zone / open_zones / light_zone).
    2. Translates it into the real parameter name (target_param), with
       optional list wrapping for parameters whose compute function expects
       a list (e.g. zero_maze open_zones).
    """

    target_param: str
    wrap_list: bool = False


@dataclass(frozen=True)
class ZoneConceptParam:
    """范式级 zone 概念 → compute 参数映射。

    EPM 的 open_arms/closed_arms → open_arm_zones/closed_arm_zones
    参数注入通过此映射实现，不依赖 convention 推导。
    """

    param: str
    wrap_list: bool = False


@dataclass(frozen=True)
class ParamBinding:
    """概念的运行时注入绑定（param 与 wrap_list 同生共死）。"""

    param: str
    wrap_list: bool = False


@dataclass(frozen=True)
class ResolvedZoneConcept:
    """统一内部 concept 模型（加载期规范化产物）。

    模型本体语义 = 「对齐目标 + 可选的注入绑定」，**不是「注入参数表」**：
    每个可注入概念必须可对齐，但不是每个可对齐概念必须可注入
    （Fable 2026-06-11 决策门 1）。

    binding=None 表示「可被 HITL 对齐/认领（消解歧义），但无运行时注入点」——
    Stage 3 的 OFT border 即此态（脚本靠 regex 自动识别 + 三级降级，不吃注入）。
    用 ParamBinding | None **整体可空**（非裸 param: str | None），让非法状态
    （param=None 但 wrap_list 有值）不可表达。

    来源三态（仅记录，不影响消费）：
      - "zone_concept_params": 直接来自 cat.zone_concept_params（EPM）
      - "anonymous_zone_override": 由 _derive_concept_from_zone_patterns 规范化（OFT/LDB/ZM）
      - "explicit_concept": Stage 3 catalog 显式声明的补集概念（border/dark/closed）
    """

    concept: str
    binding: ParamBinding | None = None
    source: str = "zone_concept_params"


@dataclass(frozen=True)
class Catalog:
    paradigm: str
    ev19_templates: list[str]
    default_metrics: list[MetricEntry]
    optional_metrics: list[MetricEntry]
    charts: list[ChartEntry]
    statistics_default: StatisticsEntry | None
    paradigm_parameters: ParadigmParameters = field(
        default_factory=ParadigmParameters
    )
    anonymous_zone_override: AnonymousZoneOverride | None = None
    zone_concept_params: dict[str, ZoneConceptParam] = field(default_factory=dict)
    resolved_zone_concepts: dict[str, ResolvedZoneConcept] = field(default_factory=dict)


@dataclass(frozen=True)
class CommonCatalog:
    """Paradigm-agnostic fallback resources."""

    common_charts: list[ChartEntry]
    shared_parameters: SharedParameters = field(
        default_factory=lambda: SharedParameters(parameters={})
    )


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
    # W27 (2026-05-27): catalog 判读 / 展示字段透传到 plan,subagent 直接读 plan 即可,
    # 不再 read catalog YAML。详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md
    unit_zh: str = ""
    one_liner: str = ""
    output_unit: str = ""
    direction_for_anxiety: str | None = None
    statistical_default: str = ""
    parameters_in_use: dict[str, float | int | str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)  # === Sprint 2b: CLI args for code-executor ===


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
    # Sprint 列对齐（spec 2026-06-16）：statistics 路径复用 metrics 段同一份
    # zone_aliases_overrides 的投影（SSOT），透传给 run_groupwise_stats → dispatcher。
    # 默认空 dict 向后兼容（无 column_aliases 时 statistics 行为不变）。
    parameters: dict[str, list[str] | str] = field(default_factory=dict)


@dataclass
class PlanChart:
    id: str
    script: str
    input: str
    output: str
    subject_index: int = 0  # 0-based index into inputs.raw_files; 0 for single-subject plans
    display_name_zh: str = ""           # 1.1: 中文图名，透传自 ChartEntry
    confidence: str = "optional"        # 透传自 ChartEntry.confidence
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
