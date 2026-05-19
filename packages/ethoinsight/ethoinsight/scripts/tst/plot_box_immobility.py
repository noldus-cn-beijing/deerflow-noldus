"""TST: 不动行为组间对比箱线图。

CLI: python -m ethoinsight.scripts.tst.plot_box_immobility \
       --inputs <inputs.json> --groups <groups.json> --output <png>

inputs.json: ["/path/to/file1.txt", ...]
groups.json: {"control": ["Subject 1", ...], "treatment": ["Subject 4", ...]}

输出: PNG 图像。
"""

from __future__ import annotations

import sys

from ethoinsight.charts import box_plot
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    make_plot_parser,
    read_groups_json,
    read_inputs_json,
)


METRICS_TO_PLOT = ["immobility_time", "immobility_bout_count"]


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)
    if not args.inputs:
        print(
            "error: plot_box_immobility requires --inputs (multi-file)", file=sys.stderr
        )
        return 2

    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups) if args.groups else None

    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(
        parsed, paradigm="tail_suspension", groups=groups
    )

    output_path = box_plot(
        metrics, metrics_to_plot=METRICS_TO_PLOT, output_path=args.output
    )
    emit_result(
        {"plot": "box_immobility", "path": output_path, "metrics": METRICS_TO_PLOT}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
