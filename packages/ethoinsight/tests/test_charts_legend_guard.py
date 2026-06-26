"""spec 2026-06-26: trajectory_plot legend UserWarning 消除（红→绿 TDD）。

根因：``charts.py`` 的 ``trajectory_plot`` 在 ``color_by`` 分支内无条件调
``ax.legend(...)``。matplotlib 收集图例 handle 时会**忽略 label 以下划线开头的
artist**（``ax.get_legend_handles_labels()`` 的既定行为）。当唯一/所有分组的
``str(grp)`` 恰好以下划线开头（如 ``_subject_1`` 这类标识），``legend()`` 调用时
拿不到任何可收集 label，遂发
``UserWarning: No artists with labels found to put in legend.``（gateway.log 一 thread
出现 19 次）。修法 = 调 legend 前用 ``ax.get_legend_handles_labels()`` 守门，仅当确
有 label 时才画图例——matplotlib 对"无 label 仍调 legend"才发此 warning。

图本身不变：legend 本就该在有可收集图例项时才出。
"""

from __future__ import annotations

import warnings

import pandas as pd

from ethoinsight.charts import trajectory_plot


def _df_with_positions() -> pd.DataFrame:
    return pd.DataFrame({"x_center": [0, 1, 2, 3], "y_center": [0, 1, 0, 1]})


def test_trajectory_plot_no_legend_warning_without_labels(tmp_path):
    """单组轨迹（无 color_by 分组）不应触发 matplotlib legend UserWarning。"""
    df = _df_with_positions()
    out = tmp_path / "traj.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        path = trajectory_plot(df, output_path=str(out))
    assert path == str(out)


def test_trajectory_plot_all_nan_color_by_no_legend_warning(tmp_path):
    """color_by 列存在但全 NaN（无可用分组 label）也不应触发 legend warning。

    这正是 gateway.log 里 19 次-warning 的诱因分支：进入 ``color_by in df.columns``
    分支、但 ``ax.get_legend_handles_labels()`` 取不到 label，旧代码仍无条件调
    ``ax.legend(...)`` → warning。
    """
    df = _df_with_positions()
    df["subject"] = pd.NA  # color_by 列存在但无任何有效分组
    out = tmp_path / "traj_nan.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trajectory_plot(df, color_by="subject", output_path=str(out))


def test_trajectory_plot_underscore_group_label_no_legend_warning(tmp_path):
    """分组名以下划线开头时（matplotlib 收集 legend 会忽略）不应触发 warning。

    faithful reproduction：``str(grp)`` 形如 ``_a``，matplotlib 的
    ``get_legend_handles_labels()`` 跳过下划线前缀 label，于是 ``legend()`` 拿不到任
    何 artist → 旧代码此处必发 UserWarning。修法=守门后该 warning 消失。
    """
    df = _df_with_positions()
    df["subject"] = ["_a", "_a", "_a", "_a"]
    out = tmp_path / "traj_underscore.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trajectory_plot(df, color_by="subject", output_path=str(out))
