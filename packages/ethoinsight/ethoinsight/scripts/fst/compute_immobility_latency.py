"""FST: 不动潜伏期 (immobility latency)。

CLI: python -m ethoinsight.scripts.fst.compute_immobility_latency \
       --input <轨迹文件> --output <metric.json>

输出 JSON:
  {"metric": "immobility_latency", "value": <float or null>,
   "signal_distribution": {...} or null}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.fst import compute_immobility_latency_fst
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    parse_parameters,
    save_output_json,
)
from ethoinsight.scripts._signal_distribution import extract_signal_distribution


METRIC_NAME = "immobility_latency"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    parameters = parse_parameters(args)
    value = compute_immobility_latency_fst(df, **parameters)

    payload = {"metric": METRIC_NAME, "value": value, "parameters_used": parameters}

    # Phase 2: 额外提取逐帧信号分布统计量
    sig = extract_signal_distribution(df, parameters)
    if sig is not None and sig.get("n_frames", 0) > 0:
        payload["signal_distribution"] = sig

    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
