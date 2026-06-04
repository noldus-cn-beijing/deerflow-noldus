"""FST: 活动强度图（velocity 时序面积图）。

CLI:
  单文件:  python -m ethoinsight.scripts.fst.plot_activity_intensity \
             --input <轨迹文件> --output <png>
  多文件:  python -m ethoinsight.scripts.fst.plot_activity_intensity \
             --inputs <inputs.json> --output <png>

per-subject plot: reads ``paths[0]`` when given an inputs.json so callers can
use the uniform ``--inputs`` contract regardless of single- vs multi-subject.
"""

from __future__ import annotations

import sys

from ethoinsight.charts import activity_intensity_plot
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import emit_result, make_plot_parser, resolve_per_subject_input


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    try:
        path = resolve_per_subject_input(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    df = parse_trajectory(path)
    output_path = activity_intensity_plot(df, output_path=args.output, smooth_window=10)
    emit_result({"plot": "activity_intensity", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
