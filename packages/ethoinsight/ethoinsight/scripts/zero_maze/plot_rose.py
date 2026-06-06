"""Zero Maze: 头部朝向玫瑰图 (rose plot)。

CLI: python -m ethoinsight.scripts.zero_maze.plot_rose \
       --inputs <inputs.json> --output <png>
"""

from __future__ import annotations

import sys

import numpy as np

from ethoinsight.charts import rose_plot
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
    if not args.inputs:
        print("error: plot_rose requires --inputs (multi-file)", file=sys.stderr)
        return 2

    paths = read_inputs_json(args.inputs)
    parsed = parse_batch(paths)

    directions_rad: list[float] = []
    for _subject_name, df in parsed["subjects"].items():
        if "Direction" in df.columns:
            v = df["Direction"].dropna().to_numpy(dtype=float)
            directions_rad.extend(v.tolist())

    output_path = rose_plot(
        np.array(directions_rad),
        output_path=args.output,
        title="Head direction distribution",
    )
    emit_result({"plot": "rose", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
