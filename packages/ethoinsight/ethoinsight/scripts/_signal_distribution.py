"""共享的信号分布提取逻辑，供 FST/TST compute_* 脚本使用。

Phase 2 (seal-robustness): 从 DataFrame 提取 periodicity / velocity 逐帧分布统计量，
随 compute 脚本 payload 输出，供 code-executor 聚合进 per_subject 的 _signal_distributions。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ethoinsight.metrics._common import (
    _compute_distribution_stats,
    _estimate_dt,
    _find_activity_column,
)


def extract_signal_distribution(
    df: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict | None:
    """从 DataFrame 提取 periodicity / velocity 逐帧分布统计量。

    优先 pendulum (activity) 路径，fallback velocity (x/y center) 路径。
    返回 None 表示无法提取（缺列 / 全 NaN）。

    Parameters
    ----------
    df : pd.DataFrame
        解析后的轨迹数据。
    parameters : dict
        脚本参数（含 pendulum_* / velocity_threshold 等）。
    """
    # --- 路径 1: pendulum / activity ---
    activity_col = _find_activity_column(df)
    if activity_col is not None:
        activity = df[activity_col].to_numpy(dtype=float)
        if not np.all(np.isnan(activity)):
            dt = _estimate_dt(df)
            pendulum_kwargs = {k: v for k, v in parameters.items() if k.startswith("pendulum_")}
            from ethoinsight.metrics._pendulum import pendulum_periodicity_series

            periodicity = pendulum_periodicity_series(activity, dt, **pendulum_kwargs)
            return _compute_distribution_stats(periodicity, "periodicity")

    # --- 路径 2: velocity (x/y center) ---
    if "x_center" in df.columns and "y_center" in df.columns:
        from ethoinsight.metrics._common import _resolve_immobile_from_velocity

        vel_kwargs = {}
        for k in ("velocity_threshold", "velocity_min_duration"):
            if k in parameters:
                vel_kwargs[k] = parameters[k]
        result = _resolve_immobile_from_velocity(df, return_signal=True, **vel_kwargs)
        if result is not None:
            _, _, velocity_arr = result
            return _compute_distribution_stats(velocity_arr, "velocity")

    return None
