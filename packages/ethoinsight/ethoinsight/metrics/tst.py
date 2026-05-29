"""Tail Suspension Test 范式指标：不动行为检测（抑郁样行为）。

Sprint 2b: 函数签名加 **kwargs 透传参数（velocity_threshold, velocity_min_duration,
pendulum_* 等），由 compute script 通过 parse_parameters → **parameters_in_use unpack 传入。
"""

from __future__ import annotations
import pandas as pd
from ethoinsight.metrics._common import (
    compute_immobility_time,
    compute_immobility_latency,
    compute_immobility_bout_count,
)

# Let _find_mobility_column auto-detect: handles both combined "activity_state"
# and one-hot "activity_state_immobile" (EthoVision real export).
DEFAULT_MOBILITY_COL: str | None = None


def compute_immobility_time_tst(df: pd.DataFrame, **kwargs) -> float | None:
    """Total immobility time (seconds) for Tail Suspension Test.

    Accepts either the combined ``activity_state`` column or the one-hot
    ``activity_state_immobile`` column.

    kwargs 透传给 compute_immobility_time（Sprint 2b）。
    """
    return compute_immobility_time(df, mobility_col=DEFAULT_MOBILITY_COL, **kwargs)


def compute_immobility_latency_tst(df: pd.DataFrame, **kwargs) -> float | None:
    """Latency to first immobility bout (seconds) for Tail Suspension Test.

    Returns None if the animal was never immobile.
    """
    return compute_immobility_latency(df, mobility_col=DEFAULT_MOBILITY_COL, **kwargs)


def compute_immobility_bout_count_tst(df: pd.DataFrame, **kwargs) -> int | None:
    """Number of immobility bouts (run-length encoding) for Tail Suspension Test."""
    return compute_immobility_bout_count(df, mobility_col=DEFAULT_MOBILITY_COL, **kwargs)
