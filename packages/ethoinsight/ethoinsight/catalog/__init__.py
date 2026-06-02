"""ethoinsight.catalog — 范式 → 指标 catalog 模块.

承载 single source of truth:每个 paradigm 一份 YAML 文件,定义默认指标清单 +
脚本路径 + 列要求 + 展示元数据 + 判读方向性。

运行时消费契约(2026-05-27 起):
  - **lead agent**: 通过 deerflow first-party 工具 `prep_metric_plan` 间接消费 —
    工具在 sandbox 外的 deerflow 进程内调 resolve_metrics(),把结果写到
    /mnt/user-data/workspace/plan_metrics.json
  - **subagent(data-analyst / report-writer)**: 不直接读 catalog YAML,
    从 plan_metrics.json 取所有判读 / 展示字段
  - **dispatcher / 单测 / golden-case**: 直接 import 使用(沙箱外环境)

设计 spec:
  - docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md(原始架构)
  - docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md(消费契约修正)
"""

from __future__ import annotations

from ethoinsight.catalog.loader import CatalogError, load_catalog, load_common_catalog
from ethoinsight.catalog.resolve import ResolveError, plan_charts_to_dict, plan_metrics_to_dict, plan_to_dict, resolve, resolve_charts, resolve_metrics
from ethoinsight.catalog.schema import (
    Catalog,
    ChartEntry,
    CommonCatalog,
    MetricEntry,
    ParamSpec,
    ParadigmParameters,
    Plan,
    PlanChart,
    PlanCharts,
    PlanInputs,
    PlanMetric,
    PlanMetrics,
    PlanSkipped,
    PlanStatistics,
    SharedParameters,
    StatisticsEntry,
)

__all__ = [
    "Catalog",
    "CatalogError",
    "ChartEntry",
    "CommonCatalog",
    "MetricEntry",
    "ParamSpec",
    "ParadigmParameters",
    "Plan",
    "PlanChart",
    "PlanCharts",
    "PlanInputs",
    "PlanMetric",
    "PlanMetrics",
    "PlanSkipped",
    "PlanStatistics",
    "ResolveError",
    "SharedParameters",
    "StatisticsEntry",
    "load_catalog",
    "load_common_catalog",
    "plan_charts_to_dict",
    "plan_metrics_to_dict",
    "plan_to_dict",
    "resolve",
    "resolve_charts",
    "resolve_metrics",
]
