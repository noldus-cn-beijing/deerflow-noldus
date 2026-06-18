"""FST: 分组统计检验（Shapiro-Wilk 决策树自动选 t-test / Mann-Whitney）。

CLI: python -m ethoinsight.scripts.fst.run_groupwise_stats \
       --inputs <inputs.json> --groups <groups.json> --output <stats.json>

输出 JSON:
  {"paradigm": "forced_swim",
   "comparisons": {metric: [{"group1", "group2", "p_value", ...}]},
   "summary": str, "alpha": float, "correction": str}
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    bridge_groups_to_subjects,
    build_subject_label_map,
    emit_result,
    make_stats_parser,
    parse_parameters,
    read_groups_json,
    read_inputs_json,
    save_output_json,
)
from ethoinsight.statistics import compare_groups


METRICS_TO_TEST = [
    "immobility_time",
    "immobility_latency",
    "immobility_bout_count",
]


def main(argv: list[str] | None = None) -> int:
    args = make_stats_parser(description=__doc__).parse_args(argv)
    paths = read_inputs_json(args.inputs)
    groups = read_groups_json(args.groups)

    parsed = parse_batch(paths)
    # 第三层 bug 修复（spec 2026-06-16）：按文件桥接 groups→subject key（见 _cli.bridge_groups_to_subjects）。
    subject_label_map = build_subject_label_map(groups, parsed["file_subjects"])
    groups = bridge_groups_to_subjects(groups, parsed["file_subjects"])
    zone_overrides = parse_parameters(args)
    metrics = compute_paradigm_metrics(
        parsed, paradigm="forced_swim", groups=groups, zone_overrides=zone_overrides
    )
    stats = compare_groups(
        metrics, metrics_to_test=METRICS_TO_TEST, subject_label_map=subject_label_map
    )

    payload = {"paradigm": "forced_swim", **stats}
    save_output_json(args.output, payload)
    emit_result(
        {
            "stats": "fst_groupwise",
            "n_metrics": len(METRICS_TO_TEST),
            "summary": stats.get("summary", ""),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
