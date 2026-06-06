"""共享的信号分布提取逻辑，供 FST/TST compute_* 脚本使用。

Phase 2 (seal-robustness): 从 DataFrame 提取 periodicity / velocity 逐帧分布统计量，
随 compute 脚本 payload 输出，供 code-executor 聚合进 per_subject 的 _signal_distributions。

**路径一致性（2026-06-03 修复）**：signal_distribution 必须与 immobility 指标**实际走的
resolution path** 一致，不能各自独立选信号。否则会出现"指标用 mobility_state 算、分布却报
velocity"的错配——data-analyst 拿一个与指标无关的分布去审计从未运行的算法的阈值，陷入死循环。
- mobility_state 路径：immobility 来自 EthoVision XT 自带 Mobility 检测，**我们没有阈值化任何
  连续信号** → 不产出 signal_distribution（None）。
- pendulum 路径：产出 periodicity 分布。
- velocity 路径：产出 velocity 分布。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ethoinsight.metrics._common import (
    MOBILITY_STATE_PATH,
    PENDULUM_PATH,
    VELOCITY_PATH,
    _compute_distribution_stats,
    _estimate_dt,
    _find_activity_column,
    filter_parameters_for_path,
    resolve_immobile_with_path,
)


def resolve_immobility_metadata(
    df: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    mobility_col: str | None = None,
) -> tuple[dict[str, Any], dict | None]:
    """一次性解析 immobility resolution path，产出 compute 脚本要透传的两样元数据。

    供 FST/TST 的 immobility compute 脚本调用，确保 parameters_used 与
    signal_distribution **都**与指标实际走的路径一致：
    - parameters_used 只保留实际路径真正消费的参数（剔除幽灵 pendulum_/velocity_）。
    - signal_distribution 与同一路径同源（mobility_state 路径不报分布）。

    Returns
    -------
    (filtered_parameters, signal_distribution_or_None)

    路径解析失败（无任何可用列）时：参数原样返回、分布为 None（让指标 value 自己
    去返回 None；本函数不二次判错）。
    """
    resolved = resolve_immobile_with_path(df, mobility_col, **parameters)
    if resolved is None:
        # 无可用列：指标 value 也会是 None。参数无从裁剪，原样返回。
        return parameters, None
    path = resolved[2]
    filtered = filter_parameters_for_path(parameters, path)
    sig = extract_signal_distribution(df, parameters, path=path, mobility_col=mobility_col)
    return filtered, sig



def extract_signal_distribution(
    df: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    path: str | None = None,
    mobility_col: str | None = None,
) -> dict | None:
    """从 DataFrame 提取与 immobility 指标实际路径一致的逐帧信号分布统计量。

    返回 None 表示该路径没有"我们阈值化的连续信号"可报（mobility_state 路径），
    或无法提取（缺列 / 全 NaN）。

    Parameters
    ----------
    df : pd.DataFrame
        解析后的轨迹数据。
    parameters : dict
        脚本参数（含 pendulum_* / velocity_threshold 等）。
    path : str | None
        immobility 指标实际走的 resolution path（MOBILITY_STATE_PATH /
        PENDULUM_PATH / VELOCITY_PATH）。compute 脚本应传入它解析指标时拿到的
        同一个 path，确保分布与指标同源。None 时本函数自行解析（向后兼容）。
    mobility_col : str | None
        path=None 时用于自行解析 immobility path 的 mobility 列名。
    """
    if path is None:
        resolved = resolve_immobile_with_path(df, mobility_col, **parameters)
        path = resolved[2] if resolved is not None else None

    # mobility_state 路径：immobility 由 EthoVision 自带列给出，我们没有阈值化
    # 任何连续信号 → 报分布是误导，直接 None。
    if path == MOBILITY_STATE_PATH:
        return None

    if path == PENDULUM_PATH:
        activity_col = _find_activity_column(df)
        if activity_col is None:
            return None
        activity = df[activity_col].to_numpy(dtype=float)
        if np.all(np.isnan(activity)):
            return None
        dt = _estimate_dt(df)
        pendulum_kwargs = {k: v for k, v in parameters.items() if k.startswith("pendulum_")}
        from ethoinsight.metrics._pendulum import pendulum_periodicity_series

        periodicity = pendulum_periodicity_series(activity, dt, **pendulum_kwargs)
        return _compute_distribution_stats(periodicity, "periodicity")

    if path == VELOCITY_PATH:
        if "x_center" not in df.columns or "y_center" not in df.columns:
            return None
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
