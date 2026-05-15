"""Shoaling: 鱼群极性 (group polarity)。

CLI: python -m ethoinsight.scripts.shoaling.compute_group_polarity \
       --inputs <inputs.json> --output <metric.json>

inputs.json 包含多个 trajectory 文件路径的 JSON 数组。

输出 JSON:
  {"metric": "group_polarity", "value": {"mean_polarity": float | null, ...}}

stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations
import sys
from ethoinsight.metrics.shoaling import compute_group_polarity
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import (
    emit_result,
    read_inputs_json,
    save_output_json,
)
import argparse

METRIC_NAME = "group_polarity"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs", required=True, help="Path to JSON array of input file paths"
    )
    ap.add_argument("--output", required=True, help="Path to write metric JSON")
    args = ap.parse_args(argv)

    paths = read_inputs_json(args.inputs)
    parsed = parse_batch(paths)
    result = compute_group_polarity(parsed["subjects"])

    if result is not None:
        summary = {
            "mean_polarity": float(result["polarity"].mean()),
            "std_polarity": float(result["polarity"].std()),
        }
    else:
        summary = None
    payload = {"metric": METRIC_NAME, "value": summary}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
