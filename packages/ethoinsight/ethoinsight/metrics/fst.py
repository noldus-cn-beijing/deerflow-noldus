"""Forced Swim Test 范式指标：不动行为检测（抑郁样行为）。"""

from __future__ import annotations
import pandas as pd
from ethoinsight.metrics._common import (
    compute_immobility_time,
    compute_immobility_latency,
    compute_immobility_bout_count,
)

# Let _find_mobility_column auto-detect: handles both combined "mobility_state"
# and one-hot "mobility_state_immobile" (EthoVision real export).
DEFAULT_MOBILITY_COL: str | None = None


def compute_immobility_time_fst(df: pd.DataFrame) -> float | None:
    """Total immobility time (seconds) for Forced Swim Test.

    Sums the duration of all immobility bouts. Accepts either the combined
    ``mobility_state`` column or the one-hot ``mobility_state_immobile`` column.
    """
    return compute_immobility_time(df, mobility_col=DEFAULT_MOBILITY_COL)


def compute_immobility_latency_fst(df: pd.DataFrame) -> float | None:
    """Latency to first immobility bout (seconds) for Forced Swim Test.

    Returns None if the animal was never immobile.
    """
    return compute_immobility_latency(df, mobility_col=DEFAULT_MOBILITY_COL)


def compute_immobility_bout_count_fst(df: pd.DataFrame) -> int | None:
    """Number of immobility bouts (run-length encoding) for Forced Swim Test."""
    return compute_immobility_bout_count(df, mobility_col=DEFAULT_MOBILITY_COL)
