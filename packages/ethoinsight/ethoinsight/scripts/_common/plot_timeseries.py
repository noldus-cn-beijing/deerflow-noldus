"""通用:时间序列图(单 subject 或多 subject 叠加)。

CLI:
  单文件: python -m ethoinsight.scripts._common.plot_timeseries \
            --input <轨迹文件> --output <png> [--y-col <列名>]
  多文件: python -m ethoinsight.scripts._common.plot_timeseries \
            --inputs <inputs.json> --output <png> [--y-col <列名>]

--y-col 未传时,按 paradigm 默认 y_col 映射(--paradigm 也未传时回退
distance_moved)。如果默认列在数据里不存在(常见于 FST/LDB 真实导出),会再次
回退到第一个数值列,避免生成空图。
"""

from __future__ import annotations

import sys

import pandas as pd
import pandas.api.types as pdt

from ethoinsight.charts import timeseries_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


_DEFAULT_Y_COL_BY_PARADIGM: dict[str, str] = {
    "epm": "open_arm_time_ratio",
    "oft": "center_time_ratio",
    "zero_maze": "open_arm_time_ratio",
    "fst": "mobility_continuous",
    "tst": "mobility_continuous",
    "light_dark_box": "in_zone_light",
    "shoaling": "nearest_neighbor_distance",
}
_GLOBAL_DEFAULT_Y_COL = "distance_moved"

_FALLBACK_SKIP_COLUMNS = frozenset({
    "trial_time", "recording_time", "x_center", "y_center",
    "subject", "subject_index", "group",
})


def _resolve_y_col(y_col: str | None, paradigm: str | None) -> str:
    if y_col:
        return y_col
    if paradigm and paradigm in _DEFAULT_Y_COL_BY_PARADIGM:
        return _DEFAULT_Y_COL_BY_PARADIGM[paradigm]
    return _GLOBAL_DEFAULT_Y_COL


def _pick_fallback_y_col(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for col in df.columns:
        if col in _FALLBACK_SKIP_COLUMNS:
            continue
        if pdt.is_numeric_dtype(df[col]) and not pdt.is_bool_dtype(df[col]):
            return col
    return requested


def main(argv: list[str] | None = None) -> int:
    ap = make_plot_parser(description="时间序列图 CLI 包装", supports_groups=False)
    ap.add_argument("--y-col", default=None, help="y 轴列名(可选,默认按 paradigm 选)")
    ap.add_argument("--paradigm", default=None, help="paradigm 名(用于决定默认 y_col)")
    args = ap.parse_args(argv)

    y_col = _resolve_y_col(args.y_col, args.paradigm)

    if args.input:
        df = parse_trajectory(args.input)
    else:
        paths = read_inputs_json(args.inputs)
        dfs = []
        for p in paths:
            sub_df = parse_trajectory(p)
            subject_attr = sub_df.attrs.get("subject", p)
            sub_df = sub_df.assign(subject=subject_attr)
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)

    y_col = _pick_fallback_y_col(df, y_col)

    output_path = timeseries_plot(df, y_col=y_col, output_path=args.output)
    emit_result({"plot": "timeseries", "path": output_path, "y_col": y_col})
    return 0


if __name__ == "__main__":
    sys.exit(main())
