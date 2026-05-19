"""ethoinsight.catalog — 范式 → 指标 catalog 模块.

承载 single source of truth：每个 paradigm 一份 YAML 文件，定义默认指标清单 +
脚本路径 + 列要求 + 展示元数据 + 判读方向性。被 lead / data-analyst /
report-writer 多方共读、被 dispatcher / 单测 / golden-case 共消费。

设计 spec: docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md
"""

from __future__ import annotations

from ethoinsight.catalog.loader import CatalogError, CommonCatalog, load_catalog, load_common_catalog
from ethoinsight.catalog.resolve import ResolveError, plan_charts_to_dict, plan_metrics_to_dict, plan_to_dict, resolve, resolve_charts, resolve_metrics
from ethoinsight.catalog.schema import (
    Catalog,
    ChartEntry,
    MetricEntry,
    Plan,
    PlanChart,
    PlanCharts,
    PlanInputs,
    PlanMetric,
    PlanMetrics,
    PlanSkipped,
    PlanStatistics,
    StatisticsEntry,
)

__all__ = [
    "Catalog",
    "CatalogError",
    "ChartEntry",
    "CommonCatalog",
    "MetricEntry",
    "Plan",
    "PlanChart",
    "PlanCharts",
    "PlanInputs",
    "PlanMetric",
    "PlanMetrics",
    "PlanSkipped",
    "PlanStatistics",
    "ResolveError",
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
