"""Metric validation — L-A 安全网（NaN/Inf only, 不依赖 catalog）。

两层分工：
  L-A（本模块） — 进程内安全网，只查 NaN/Inf（name-agnostic 确定性检查）
  L-B（validate_catalog.py）— 语义范围校验，按 catalog output_unit 查范围

范围校验（ratio/pct/非负）已从本模块移除，全部迁移到 L-B。
"""

import math
from typing import Any


def validate_metrics(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Validate computed metrics for NaN / Inf (L-A safety net).

    Range checks (ratio/pct/non-negative) are now done by L-B
    (``ethoinsight.validate_catalog.validate_metrics_against_catalog``)
    which uses the catalog's ``output_unit`` to drive validation.

    Args:
        metrics: {metric_name: value} dict from compute script output.

    Returns:
        List of violations (empty list = all clear).
        Each violation: {"metric": str, "issue": str, "value": str}
    """
    violations: list[dict[str, str]] = []

    for name, value in metrics.items():
        # Skip bool (isinstance(True, int) is True, so check bool first)
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue

        # NaN check (name-agnostic — the core AutoResearch safety net)
        if math.isnan(value):
            violations.append({"metric": name, "issue": "NaN", "value": "NaN"})
            continue

        # Inf check
        if math.isinf(value):
            violations.append({"metric": name, "issue": "Inf", "value": str(value)})
            continue

    return violations
