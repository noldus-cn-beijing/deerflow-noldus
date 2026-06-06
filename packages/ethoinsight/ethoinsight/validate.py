"""Metric validation — deterministic range/NaN checks.

AutoResearch-inspired: code-enforced checks that shouldn't be left to LLM judgment.

Naming conventions follow the real catalog metric names (suffix-based):
  - *_ratio   → 0–1 range (real: open_arm_time_ratio, center_distance_ratio, …)
  - *_pct     → 0–100 range (future percentage metrics)
  - *_count   → non-negative (real: center_entry_count, transition_count, …)
  - *_time    → non-negative (real: open_zone_time, immobility_time, …)
  - *_latency → non-negative (real: light_latency, immobility_latency, …)
  - *_distance → non-negative (real: cumulative_distance, …)
"""

import math
from typing import Any

# Suffixes that imply non-negative values (matching real catalog metric names)
_NON_NEGATIVE_SUFFIXES = ("_count", "_time", "_latency", "_distance")


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
        # Skip bool (isinstance(True, int) is True, so check bool first)
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue

        # NaN/Inf check (name-agnostic — the core AutoResearch safety net)
        if math.isnan(value):
            violations.append({"metric": name, "issue": "NaN", "value": "NaN"})
            continue
        if math.isinf(value):
            violations.append({"metric": name, "issue": "Inf", "value": str(value)})
            continue

        # Percentage range check (naming convention: *_pct, 0–100)
        if name.endswith("_pct") and not (0.0 <= value <= 100.0):
            violations.append({
                "metric": name,
                "issue": "percentage_out_of_range",
                "value": str(value),
            })

        # Ratio range check (naming convention: *_ratio, 0–1)
        if name.endswith("_ratio") and not (0.0 <= value <= 1.0):
            violations.append({
                "metric": name,
                "issue": "ratio_out_of_range",
                "value": str(value),
            })

        # Non-negative check — suffix-based to match real catalog names
        # (center_entry_count, immobility_time, light_latency, cumulative_distance, …)
        if any(name.endswith(suffix) for suffix in _NON_NEGATIVE_SUFFIXES):
            if value < 0:
                violations.append({
                    "metric": name,
                    "issue": "negative_value",
                    "value": str(value),
                })

    return violations
