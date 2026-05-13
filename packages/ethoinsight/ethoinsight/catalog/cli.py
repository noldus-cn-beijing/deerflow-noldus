"""Catalog CLI entry — python -m ethoinsight.catalog.resolve.

参数:
  --paradigm PARADIGM            必填
  --columns-file PATH            必填（dump_headers 产物）
  --raw-files-json PATH          必填（指向 JSON 数组）
  --groups-file PATH             可选
  --workspace-dir PATH           必填
  --include METRIC_ID            可重复
  --exclude METRIC_ID            可重复
  --n-per-group INT              可选
  --n-groups INT                 可选
  --ev19-template TEMPLATE_ID    可选
  --output PATH                  必填，写 plan.json 路径

行为:
  成功: exit 0 + 写 plan.json
  失败: exit 1 + stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.resolve")
    p.add_argument("--paradigm", required=True)
    p.add_argument("--columns-file", required=True)
    p.add_argument("--raw-files-json", required=True)
    p.add_argument("--workspace-dir", required=True)
    p.add_argument("--groups-file", default=None)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--n-per-group", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--ev19-template", default=None)
    p.add_argument("--output", required=True)
    return p


def _emit_error(code: str, message: str, details: dict | None = None) -> int:
    payload = {"code": code, "message": message, "details": details or {}}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Read columns.json
    try:
        columns_data = json.loads(Path(args.columns_file).read_text(encoding="utf-8"))
        columns = columns_data.get("columns", [])
        if not isinstance(columns, list):
            return _emit_error(
                "schema_violation",
                f"columns-file does not contain a 'columns' list: {args.columns_file}",
                {"path": args.columns_file},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read columns-file: {e}",
            {"path": args.columns_file},
        )

    # Read raw-files-json
    try:
        raw_files = json.loads(Path(args.raw_files_json).read_text(encoding="utf-8"))
        if not isinstance(raw_files, list) or not all(isinstance(p, str) for p in raw_files):
            return _emit_error(
                "schema_violation",
                f"raw-files-json must be a JSON array of strings: {args.raw_files_json}",
                {"path": args.raw_files_json},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read raw-files-json: {e}",
            {"path": args.raw_files_json},
        )

    try:
        plan = resolve(
            paradigm=args.paradigm,
            columns=columns,
            raw_files=raw_files,
            workspace_dir=args.workspace_dir,
            include=args.include,
            exclude=args.exclude,
            n_per_group=args.n_per_group,
            n_groups=args.n_groups,
            groups_file=args.groups_file,
            columns_file=args.columns_file,
            ev19_template=args.ev19_template,
        )
    except ResolveError as e:
        return _emit_error(e.code, str(e), e.details)
    except Exception as e:
        return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

    # Write plan.json
    plan_dict = plan_to_dict(plan)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = (
        f"Plan written to {args.output}: paradigm={plan.paradigm}, "
        f"metrics={len(plan.metrics)}, charts={len(plan.charts)}, "
        f"skipped={len(plan.skipped)}, statistics="
        f"{'skip' if (plan.statistics and plan.statistics.skip_reason) else 'run' if plan.statistics else 'none'}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
