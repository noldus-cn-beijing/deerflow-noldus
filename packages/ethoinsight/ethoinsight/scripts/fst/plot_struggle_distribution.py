"""FST: 放弃挣扎分布图（immobility bouts eventplot）。

CLI: python -m ethoinsight.scripts.fst.plot_struggle_distribution \
       --inputs <inputs.json> --output <png>

每个 subject 一行横条，展示不动期（放弃挣扎）的时间分布。
"""

from __future__ import annotations

import sys

from ethoinsight.charts import struggle_distribution_plot
from ethoinsight.metrics._common import extract_immobility_bouts
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import emit_result, make_plot_parser, read_inputs_json


def main(argv: list[str] | None = None) -> int:
    args = make_plot_parser(description=__doc__, supports_groups=True).parse_args(argv)
    if not args.inputs:
        print("error: plot_struggle_distribution requires --inputs (multi-file)", file=sys.stderr)
        return 2
    paths = read_inputs_json(args.inputs)
    parsed = parse_batch(paths)

    bouts_by_subject: dict[str, list[tuple[float, float]]] = {}
    for subject_name, df in parsed["subjects"].items():
        bouts_by_subject[subject_name] = extract_immobility_bouts(df)

    output_path = struggle_distribution_plot(bouts_by_subject, output_path=args.output)
    emit_result({"plot": "struggle_distribution", "path": output_path})
    return 0


if __name__ == "__main__":
    sys.exit(main())
