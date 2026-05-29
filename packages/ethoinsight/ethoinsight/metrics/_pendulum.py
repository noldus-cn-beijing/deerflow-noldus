"""TST 钟摆运动检测 — 自相关周期性分析。

从 Activity 时间序列中区分"钟摆式摆动"与"真实挣扎"。
核心原理：停止挣扎后的钟摆运动具有显著的周期性（摆动频率稳定、幅度衰减），
真实挣扎表现为不规则的 Activity 波动。

算法来源：同事 tstYoyo 的 tst_pendulum_example.py + tst-pendulum-algorithm.md

Sprint 2b: 删除 9 个模块常量，改为函数签名 kwargs（带 pendulum_ 前缀，
与 catalog YAML key 一致，**parameters_in_use 直接 unpack）。
"""

from __future__ import annotations

import numpy as np


def detect_pendulum(
    activity: np.ndarray,
    dt: float = 0.04,
    *,
    pendulum_smooth_window: int = 1,
    pendulum_analysis_window: int = 25,
    pendulum_period_min: int = 4,
    pendulum_period_max: int = 12,
    pendulum_periodicity_threshold: float = 0.55,
    pendulum_activity_struggle_threshold: float = 2.0,
    pendulum_min_still_activity: float = 0.3,
    pendulum_moderate_activity_threshold: float = 1.0,
    pendulum_min_state_duration: int = 25,
    pendulum_grace_period: int = 20,
) -> list[dict]:
    """对 Activity 序列运行 6 阶段钟摆检测算法。

    Parameters
    ----------
    activity : np.ndarray
        Activity 百分比值序列 (0–100)。NaN 表示缺失帧。
    dt : float
        采样间隔（秒），用于帧率自适应。
    pendulum_* : 可调参数，default 与 catalog _common.yaml shared_parameters 一致。

    Returns
    -------
    list[dict]
        每帧一个 dict: ``state`` (0=静止, 1=挣扎),
        ``periodicity`` (0~1), ``is_pendulum`` (bool).
    """
    scale = 0.04 / dt if dt > 0 and dt != 0.04 else 1.0
    min_dur = max(1, round(pendulum_min_state_duration * scale))
    grace_max = max(0, round(pendulum_grace_period * scale))

    ring_buffer = [0.0] * pendulum_analysis_window
    ring_idx = 0
    smooth_buffer: list[float] = []
    output_state = 0
    pending_state = -1
    pending_count = 0
    grace_counter = 0
    results: list[dict] = []

    for i in range(len(activity)):
        raw = float(activity[i])

        if np.isnan(raw):
            results.append({"state": 0, "periodicity": 0.0, "is_pendulum": False})
            continue

        # Phase 1: 预处理平滑
        smooth_buffer.append(raw)
        if len(smooth_buffer) > pendulum_smooth_window:
            smooth_buffer.pop(0)
        smoothed = sum(smooth_buffer) / len(smooth_buffer)

        # Phase 2: 环形缓冲区
        ring_buffer[ring_idx % pendulum_analysis_window] = smoothed
        ring_idx += 1

        if ring_idx < pendulum_analysis_window:
            results.append({"state": 1, "periodicity": 0.0, "is_pendulum": False})
            continue

        n = pendulum_analysis_window
        mean_act = sum(ring_buffer[(ring_idx - n + j) % n] for j in range(n)) / n

        # Phase 3: 自相关周期性检测
        norm_data = [ring_buffer[(ring_idx - n + j) % n] - mean_act for j in range(n)]
        energy = sum(v * v for v in norm_data)

        max_ac = 0.0
        if energy > 1e-10:
            max_lag = min(pendulum_period_max + 1, n // 2)
            for lag in range(pendulum_period_min, max_lag):
                ac = sum(norm_data[j] * norm_data[j + lag] for j in range(n - lag))
                ac /= energy
                if ac > max_ac:
                    max_ac = ac

        periodicity = max(0.0, min(1.0, max_ac))

        # Phase 4: 宽容期更新
        if periodicity > pendulum_periodicity_threshold:
            grace_counter = grace_max
        elif grace_counter > 0:
            grace_counter -= 1

        # Phase 5: 状态判定
        recent_pendulum = grace_counter > 0
        if mean_act < pendulum_min_still_activity:
            state = 0
        elif periodicity > pendulum_periodicity_threshold:
            state = 0
        elif mean_act > pendulum_activity_struggle_threshold:
            state = 1
        elif mean_act > pendulum_moderate_activity_threshold:
            state = 1
        elif recent_pendulum:
            state = 0
        else:
            state = 1

        is_pendulum = periodicity > pendulum_periodicity_threshold and mean_act >= pendulum_min_still_activity

        # Phase 6: 持续时间过滤
        if state == output_state:
            pending_state = -1
            pending_count = 0
        elif state == pending_state:
            pending_count += 1
            if pending_count >= min_dur:
                output_state = state
                pending_state = -1
                pending_count = 0
        else:
            pending_state = state
            pending_count = 1

        results.append({
            "state": output_state,
            "periodicity": float(periodicity),
            "is_pendulum": bool(is_pendulum),
        })

    return results


def pendulum_immobility_series(
    activity: np.ndarray,
    dt: float = 0.04,
    **pendulum_kwargs,
) -> np.ndarray:
    """从 Activity 序列派生 immobility 二值序列（1=immobile, 0=mobile）。

    ``detect_pendulum`` 输出的 state=0 表示静止 → immobile=1，
    state=1 表示挣扎 → immobile=0。

    pendulum_kwargs 透传给 detect_pendulum（Sprint 2b: 支持 **parameters_in_use 直接 unpack）。

    Returns int ndarray, same length as activity.
    """
    results = detect_pendulum(activity, dt, **pendulum_kwargs)
    return np.array([1 - r["state"] for r in results], dtype=int)
