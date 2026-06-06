"""Metric validation — deterministic range/NaN checks.

AutoResearch-inspired: code-enforced checks that shouldn't be left to LLM judgment.
"""

import math
from typing import Any


def validate_metrics(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Validate computed metrics are in plausible ranges.

    Args:
        metrics: {metric_name: value} dict from compute script output.

    Returns:
        List of violations (empty list = all clear).
        Each violation: {"metric": str, "issue": str, "value": str}
    """
    violations: list[dict[str, str]] = []

    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue

        # NaN/Inf check
        if math.isnan(value):
            violations.append({"metric": name, "issue": "NaN", "value": "NaN"})
            continue
        if math.isinf(value):
            violations.append({"metric": name, "issue": "Inf", "value": str(value)})
            continue

        # Percentage range check (naming convention: *_pct, 0-100)
        if name.endswith("_pct") and not (0.0 <= value <= 100.0):
            violations.append({
                "metric": name,
                "issue": "percentage_out_of_range",
                "value": str(value),
            })

        # Ratio range check (naming convention: *_ratio, 0-1)
        if name.endswith("_ratio") and not (0.0 <= value <= 1.0):
            violations.append({
                "metric": name,
                "issue": "ratio_out_of_range",
                "value": str(value),
            })

        # Non-negative check for duration/distance/velocity/count
        if any(
            name.startswith(prefix)
            for prefix in ("distance_", "duration_", "velocity_", "count_")
        ):
            if value < 0:
                violations.append({
                    "metric": name,
                    "issue": "negative_value",
                    "value": str(value),
                })

    return violations
