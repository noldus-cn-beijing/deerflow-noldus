"""通用：轨迹图 (X/Y plot of position over time)。

CLI:
  单文件:  python -m ethoinsight.scripts._common.plot_trajectory \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts._common.plot_trajectory \
             --inputs <inputs.json> --output <png>

输出: PNG 图像文件。

inputs.json 格式: ["/path/to/file1.txt", "/path/to/file2.txt", ...]
"""

from __future__ import annotations

import sys

import pandas as pd

from ethoinsight.charts import trajectory_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)

    if args.input:
        df = parse_trajectory(args.input)
    else:
        # Multi-file aggregated trajectory plot
        paths = read_inputs_json(args.inputs)
        dfs = []
        for p in paths:
            sub_df = parse_trajectory(p)
            subject_attr = sub_df.attrs.get("subject", p)
            sub_df = sub_df.assign(subject=subject_attr)
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)

    output_path = trajectory_plot(df, color_by="subject", output_path=args.output)
    emit_result({"plot": "trajectory", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
