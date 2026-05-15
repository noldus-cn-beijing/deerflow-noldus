"""EPM: 分组统计检验（Shapiro-Wilk 决策树自动选 t-test / Mann-Whitney）。

CLI: python -m ethoinsight.scripts.epm.run_groupwise_stats \
       --inputs <inputs.json> --groups <groups.json> --output <stats.json>

输出 JSON:
  {"paradigm": "epm",
   "comparisons": {metric: [{"group1", "group2", "p_value", ...}]},
   "summary": str, "alpha": float, "correction": str}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    make_stats_parser,
    read_groups_json,
    read_inputs_json,
    save_output_json,
)
from ethoinsight.statistics import compare_groups


METRICS_TO_TEST = [
    "open_arm_time_ratio",
    "open_arm_entry_count",
    "open_arm_entry_ratio",
    "open_arm_time",
    "total_entry_count",
]


def main(argv: list[str] | None = None) -> int:
    args = make_stats_parser(description=__doc__).parse_args(argv)
    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups)

    parsed = parse_batch(paths)
    metrics = compute_paradigm_metrics(parsed, paradigm="epm", groups=groups)
    stats = compare_groups(metrics, metrics_to_test=METRICS_TO_TEST)

    payload = {"paradigm": "epm", **stats}
    save_output_json(args.output, payload)
    emit_result(
        {
            "stats": "epm_groupwise",
            "n_metrics": len(METRICS_TO_TEST),
            "summary": stats.get("summary", ""),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
