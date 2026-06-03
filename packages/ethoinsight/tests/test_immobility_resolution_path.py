"""Tests for path-aware parameters_used trimming + signal_distribution alignment.

2026-06-03 fix: FST/TST immobility 指标按可用列走三条互斥 resolution path 之一
（mobility_state / pendulum / velocity）。compute 脚本必须只把**实际走的那条
路径**真正消费的参数填进 parameters_used，并让 signal_distribution 与同一路径
同源——否则 data-analyst 的 step 2.8 会去审计从未运行过的算法的阈值，陷入死循环。

根因实证：真实 Porsolt FST 导出文件含 EthoVision 自带 mobility_state_immobile
列、无 Activity 列 → 走 mobility_state 路径，immobility 由 EthoVision 自带检测
给出，pendulum_*/velocity_* 参数一个都没参与计算，却被全量报进了 parameters_used。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ethoinsight.metrics._common import (
    MOBILITY_STATE_PATH,
    PENDULUM_PATH,
    VELOCITY_PATH,
    filter_parameters_for_path,
    resolve_immobile_with_path,
)
from ethoinsight.scripts._signal_distribution import resolve_immobility_metadata

# 全量参数集（catalog 把 pendulum_* + velocity_* 作为 shared_parameters 一并注入）
_FULL_PARAMS = {
    "pendulum_activity_struggle_threshold": 2.0,
    "pendulum_analysis_window": 25,
    "pendulum_periodicity_threshold": 0.55,
    "velocity_threshold": 30.0,
    "velocity_min_duration": 25,
    "sample_size_underpowered_threshold": 5,  # 路径无关参数
}


# ----------------------------------------------------------------------------
# DataFrame fixtures — 每个对应一条 resolution path
# ----------------------------------------------------------------------------
def _df_mobility_state(n: int = 100) -> pd.DataFrame:
    """含 EthoVision 自带 mobility_state_immobile 列 → mobility_state 路径。"""
    return pd.DataFrame({
        "trial_time": np.arange(n) * 0.04,
        "x_center": np.zeros(n),
        "y_center": np.zeros(n),
        "mobility_state_immobile": [1] * (n // 2) + [0] * (n - n // 2),
    })


def _df_activity(n: int = 200) -> pd.DataFrame:
    """含连续 Activity 列、无 mobility_state → pendulum 路径。"""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "trial_time": np.arange(n) * 0.04,
        "activity_sampleRate_1": rng.uniform(0, 3, n),
    })


def _df_velocity(n: int = 200) -> pd.DataFrame:
    """只有 x/y 中心点、无 mobility / activity 列 → velocity 路径。"""
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "trial_time": np.arange(n) * 0.04,
        "x_center": np.cumsum(rng.uniform(-0.5, 0.5, n)),
        "y_center": np.cumsum(rng.uniform(-0.5, 0.5, n)),
    })


# ============================================================================
# resolve_immobile_with_path — 路径识别正确
# ============================================================================
class TestResolveImmobileWithPath:
    def test_mobility_state_path(self):
        res = resolve_immobile_with_path(_df_mobility_state(), None, **_FULL_PARAMS)
        assert res is not None
        assert res[2] == MOBILITY_STATE_PATH

    def test_pendulum_path(self):
        res = resolve_immobile_with_path(_df_activity(), None, **_FULL_PARAMS)
        assert res is not None
        assert res[2] == PENDULUM_PATH

    def test_velocity_path(self):
        res = resolve_immobile_with_path(_df_velocity(), None, **_FULL_PARAMS)
        assert res is not None
        assert res[2] == VELOCITY_PATH

    def test_no_usable_column_returns_none(self):
        df = pd.DataFrame({"trial_time": np.arange(10) * 0.04})
        assert resolve_immobile_with_path(df, None, **_FULL_PARAMS) is None

    def test_two_tuple_wrapper_stays_backward_compatible(self):
        """_resolve_immobile_series 仍返回 2 元组（现有调用方零波及）。"""
        from ethoinsight.metrics._common import _resolve_immobile_series

        res = _resolve_immobile_series(_df_mobility_state(), None, **_FULL_PARAMS)
        assert res is not None
        assert len(res) == 2


# ============================================================================
# filter_parameters_for_path — 只保留实际路径参数
# ============================================================================
class TestFilterParametersForPath:
    def test_mobility_state_drops_all_threshold_params(self):
        """mobility_state 路径：pendulum_* 与 velocity_* 全部剔除，路径无关参数保留。"""
        out = filter_parameters_for_path(_FULL_PARAMS, MOBILITY_STATE_PATH)
        assert out == {"sample_size_underpowered_threshold": 5}
        assert not any(k.startswith("pendulum_") for k in out)
        assert not any(k.startswith("velocity_") for k in out)

    def test_pendulum_keeps_only_pendulum_params(self):
        out = filter_parameters_for_path(_FULL_PARAMS, PENDULUM_PATH)
        assert all(k.startswith("pendulum_") or k == "sample_size_underpowered_threshold" for k in out)
        assert not any(k.startswith("velocity_") for k in out)
        assert "pendulum_periodicity_threshold" in out

    def test_velocity_keeps_only_velocity_params(self):
        out = filter_parameters_for_path(_FULL_PARAMS, VELOCITY_PATH)
        assert all(k.startswith("velocity_") or k == "sample_size_underpowered_threshold" for k in out)
        assert not any(k.startswith("pendulum_") for k in out)
        assert "velocity_threshold" in out

    def test_path_agnostic_param_always_kept(self):
        for path in (MOBILITY_STATE_PATH, PENDULUM_PATH, VELOCITY_PATH):
            out = filter_parameters_for_path(_FULL_PARAMS, path)
            assert out.get("sample_size_underpowered_threshold") == 5


# ============================================================================
# resolve_immobility_metadata — parameters_used + signal_distribution 同源
# ============================================================================
class TestResolveImmobilityMetadata:
    def test_mobility_state_emits_no_signal_and_empty_params(self):
        """mobility_state 路径：无幽灵参数 + 不报 signal_distribution。"""
        used, sig = resolve_immobility_metadata(_df_mobility_state(), dict(_FULL_PARAMS))
        assert used == {"sample_size_underpowered_threshold": 5}
        assert sig is None  # 我们没阈值化任何连续信号

    def test_pendulum_signal_is_periodicity(self):
        used, sig = resolve_immobility_metadata(_df_activity(), dict(_FULL_PARAMS))
        assert not any(k.startswith("velocity_") for k in used)
        assert sig is not None
        assert sig["signal_key"] == "periodicity"

    def test_velocity_signal_is_velocity(self):
        used, sig = resolve_immobility_metadata(_df_velocity(), dict(_FULL_PARAMS))
        assert not any(k.startswith("pendulum_") for k in used)
        assert sig is not None
        assert sig["signal_key"] == "velocity"


# ============================================================================
# 回归：真实 Porsolt FST 文件不再报幽灵 pendulum/velocity 参数
# ============================================================================
class TestRealPorsoltFstRegression:
    """复刻 2026-06-02 dogfood 死循环的真实数据形态：

    EthoVision FST 导出含 mobility_state_immobile + x/y_center、无 Activity 列。
    旧行为：parameters_used 报 12 个 pendulum/velocity 参数、signal_key=velocity。
    新行为：parameters_used 不含任何 pendulum_/velocity_、无 signal_distribution。
    """

    def _df_real_fst_shape(self, n: int = 3750) -> pd.DataFrame:
        rng = np.random.default_rng(190)
        # 真实文件的列集合（见 parse_trajectory 输出）：mobility_state_immobile 主导
        return pd.DataFrame({
            "trial_time": np.arange(n) * 0.04,
            "x_center": np.cumsum(rng.uniform(-0.5, 0.5, n)),
            "y_center": np.cumsum(rng.uniform(-0.5, 0.5, n)),
            "mobility_continuous": rng.uniform(0, 1, n),
            "mobility_state_immobile": rng.integers(0, 2, n),
        })

    def test_no_ghost_pendulum_params_in_parameters_used(self):
        used, sig = resolve_immobility_metadata(self._df_real_fst_shape(), dict(_FULL_PARAMS))
        # 没有任何 pendulum_/velocity_ 幽灵参数（它们从未参与计算）
        assert not any(k.startswith("pendulum_") for k in used), used
        assert not any(k.startswith("velocity_") for k in used), used
        # 不报与指标无关的 velocity 分布
        assert sig is None
