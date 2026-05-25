"""FST: 活动强度图（velocity 时序面积图）。

CLI: python -m ethoinsight.scripts.fst.plot_activity_intensity \
       --input <轨迹文件> --output <png>
"""

from __future__ import annotations

import sys

from ethoinsight.charts import activity_intensity_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    if not args.input:
        print("error: plot_activity_intensity requires --input (single file)", file=sys.stderr)
        return 2
    df = parse_trajectory(args.input)
    output_path = activity_intensity_plot(df, output_path=args.output)
    emit_result({"plot": "activity_intensity", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
