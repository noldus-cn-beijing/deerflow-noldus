"""Tail Suspension Test 范式指标：不动行为检测（抑郁样行为）。"""

from __future__ import annotations
import pandas as pd
from ethoinsight.metrics._common import (
    compute_immobility_time,
    compute_immobility_latency,
    compute_immobility_bout_count,
)

# TST typically uses Activity_State column
DEFAULT_MOBILITY_COL = "Activity_State"


def compute_immobility_time_tst(df: pd.DataFrame) -> float | None:
    """Total immobility time (seconds) for Tail Suspension Test.

    Sums the duration of all immobility bouts (runs of 0 in Activity_State).
    """
    return compute_immobility_time(df, mobility_col=DEFAULT_MOBILITY_COL)


def compute_immobility_latency_tst(df: pd.DataFrame) -> float | None:
    """Latency to first immobility bout (seconds) for Tail Suspension Test.

    Returns the trial_time value of the first frame where Activity_State==0.
    Returns None if the animal was never immobile.
    """
    return compute_immobility_latency(df, mobility_col=DEFAULT_MOBILITY_COL)


def compute_immobility_bout_count_tst(df: pd.DataFrame) -> int | None:
    """Number of immobility bouts (run-length encoding) for Tail Suspension Test.

    Each consecutive run of 0 values in Activity_State counts as one bout.
    """
    return compute_immobility_bout_count(df, mobility_col=DEFAULT_MOBILITY_COL)
