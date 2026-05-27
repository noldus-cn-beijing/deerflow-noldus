"""通用：热区图（2D hexbin density plot of X/Y positions）。

CLI:
  单文件:  python -m ethoinsight.scripts._common.plot_heatmap \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts._common.plot_heatmap \
             --inputs <inputs.json> --output <png>

输出: PNG 图像文件。
"""

from __future__ import annotations

import sys

import pandas as pd

from ethoinsight.charts import heatmap_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)

    if args.input:
        df = parse_trajectory(args.input)
    else:
        paths = read_inputs_json(args.inputs)
        dfs = []
        for p in paths:
            sub_df = parse_trajectory(p)
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)

    output_path = heatmap_plot(df, output_path=args.output)
    emit_result({"plot": "heatmap", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
