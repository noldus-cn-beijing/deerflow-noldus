"""Catalog CLI entry — python -m ethoinsight.catalog.resolve.

参数:
  --paradigm PARADIGM            必填
  --columns-file PATH            必填（dump_headers 产物）
  --raw-files-json PATH          必填（指向 JSON 数组）
  --groups-file PATH             可选
  --workspace-dir PATH           必填（物理路径，用于脚本执行）
  --virtual-workspace-dir PATH   可选（plan.json output 虚拟路径）
                                  fallback 顺序：
                                    1. 显式传入的 --virtual-workspace-dir 值（如果不是物理路径）
                                    2. 环境变量 DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 的 key 反推
                                       （sandbox 注入的标准 env var，详见 deerflow.sandbox.tools._build_path_env）
                                    3. 兜底使用 --workspace-dir（非 sandbox 调试场景）
  --include METRIC_ID            可重复
  --exclude METRIC_ID            可重复
  --n-per-group INT              可选
  --n-groups INT                 可选
  --ev19-template TEMPLATE_ID    可选
  --output PATH                  必填，写 plan.json 路径

行为:
  成功: exit 0 + 写 plan.json
  失败: exit 1 + stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}

G5 回归修复（thread 8ff3be6d）:
  sandbox replace_virtual_paths_in_command 会把命令字符串里所有 /mnt/user-data/...
  字面量翻译成物理路径——包括 --virtual-workspace-dir 参数的值。修复方案是改用
  sandbox 已经注入的 env var 作为虚拟路径来源（env var 的 key 名稳定编码虚拟路径，
  即便 value 是物理路径，key 本身仍可反推回虚拟字符串）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ethoinsight.catalog.loader import load_common_catalog
from ethoinsight.catalog.resolve import ResolveError, plan_charts_to_dict, plan_metrics_to_dict, resolve_charts, resolve_metrics, select_charts_by_priority


# 与 deerflow.sandbox.tools._build_path_env 完全一致的命名规则：
# 虚拟路径 /mnt/user-data/workspace → env key DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
_WORKSPACE_VIRTUAL_PATH = "/mnt/user-data/workspace"
_WORKSPACE_ENV_KEY = "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE"


def _resolve_virtual_workspace_dir(arg_value: str | None, workspace_dir: str) -> str:
    """决定 plan.json output 字段用的"虚拟"路径，三级 fallback：

    1. 如果 arg_value 已经是虚拟路径前缀（/mnt/user-data/...），用 arg_value
       （兼容直接命令行调试时显式传入虚拟字符串的场景）
    2. 否则如果 env var DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE 存在，
       用 _WORKSPACE_VIRTUAL_PATH（env key 反推的稳定虚拟字符串）
    3. 否则兜底用 workspace_dir（非 sandbox 调试场景，行为同 commit 2eb1532a 之前）

    为什么 env var 的 VALUE 不直接当虚拟路径用：
      sandbox 注入的 env var value 是物理路径（用于 Python 脚本运行时读真实文件），
      不是虚拟路径。但 env var 的 KEY 是稳定的——它的存在本身就证明"sandbox 抽象在生效"，
      此时我们可以用代码中硬编码的虚拟路径常量 _WORKSPACE_VIRTUAL_PATH 作为返回值。
    """
    # 1) 显式传入的虚拟路径优先（直接调试场景）
    if arg_value and arg_value.startswith(_WORKSPACE_VIRTUAL_PATH):
        return arg_value

    # 2) sandbox 模式：env var 存在 → 用硬编码虚拟路径
    if _WORKSPACE_ENV_KEY in os.environ:
        return _WORKSPACE_VIRTUAL_PATH

    # 3) 非 sandbox 调试场景：兜底物理 workspace_dir
    return arg_value or workspace_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.resolve")
    p.add_argument("--mode", choices=["metrics", "charts"], default="metrics",
                   help="Output mode: 'metrics' (default, backward-compat) or 'charts'")
    p.add_argument("--paradigm", required=True)
    p.add_argument("--columns-file", required=True)
    p.add_argument("--raw-files-json", required=True)
    p.add_argument("--workspace-dir", required=True)
    p.add_argument("--virtual-workspace-dir", default=None)
    p.add_argument("--groups-file", default=None)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--n-per-group", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--total-subjects", type=int, default=None,
                   help="Total number of subjects; used in charts mode when conditions")
    p.add_argument("--user-intent", default=None,
                   help="User's intent string; stored in plan_charts.json (charts mode only)")
    p.add_argument("--groups-json", default=None,
                   help="Optional path to a JSON file with group assignments. "
                        "Accepted shapes: {arena_key: group_name} (e.g. {\"Arena 1\": \"Treatment\"}) "
                        "or {group_name: [subject_path, ...]}. Threaded to aggregate plots that need "
                        "group labels (chart entries with needs_groups: true).")
    p.add_argument("--ev19-template", default=None)
    p.add_argument("--column-aliases-file", default=None,
                   help="JSON file mapping {raw_or_normalized column → catalog concept} "
                        "(Sprint 1 列语义对齐). Remaps user-named zone columns before resolve.")
    p.add_argument("--overrides-file", default=None,
                   help="Optional JSON file with parameter overrides (Sprint 2b). "
                        "Content must be a JSON dict, e.g. "
                        "{\"velocity_threshold\": 5.0, \"pendulum_periodicity_threshold\": 0.6}.")
    p.add_argument("--chart-budget", type=int, default=None,
                   help="charts mode only (P5): total chart plot budget. aggregate 图（组间对比）"
                        "全画不受限；per_subject 图（个体）用剩余预算按代表性子集取。"
                        "省略 = 不限制（全画）。被截断的 per_subject 图写入 charts_budget_remaining。")
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
        if not isinstance(raw_files, list) or not all(
            isinstance(p, str) for p in raw_files
        ):
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

    # 三级 fallback 决定 plan.json output 用的虚拟路径
    virtual_workspace_dir = _resolve_virtual_workspace_dir(
        args.virtual_workspace_dir, args.workspace_dir
    )

    # Optional groups JSON (charts mode aggregate plots only).
    groups: dict | None = None
    if getattr(args, "groups_json", None):
        try:
            groups_data = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return _emit_error(
                "schema_violation",
                f"Cannot read groups-json: {e}",
                {"path": args.groups_json},
            )
        if not isinstance(groups_data, dict):
            return _emit_error(
                "schema_violation",
                f"groups-json must be a JSON object: {args.groups_json}",
                {"path": args.groups_json},
            )
        groups = groups_data

    # === Sprint 2b: load overrides ===
    overrides: dict[str, float | int | str] = {}
    if args.overrides_file:
        try:
            overrides_data = json.loads(Path(args.overrides_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return _emit_error(
                "schema_violation",
                f"Cannot read overrides-file: {e}",
                {"path": args.overrides_file},
            )
        if not isinstance(overrides_data, dict):
            return _emit_error(
                "schema_violation",
                f"overrides-file content must be a JSON dict, got {type(overrides_data).__name__}",
                {"path": args.overrides_file},
            )
        overrides = overrides_data

    # === Sprint 2b: load common catalog for shared_parameters ===
    common_catalog = None
    try:
        common_catalog = load_common_catalog()
    except Exception:
        pass  # _common.yaml missing → no shared params, that's OK

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional column aliases JSON (Sprint 1 列语义对齐): {raw_or_normalized → catalog concept}.
    column_aliases: dict | None = None
    if getattr(args, "column_aliases_file", None):
        try:
            aliases_data = json.loads(
                Path(args.column_aliases_file).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as e:
            return _emit_error(
                "schema_violation",
                f"Cannot read column-aliases-file: {e}",
                {"path": args.column_aliases_file},
            )
        if not isinstance(aliases_data, dict):
            return _emit_error(
                "schema_violation",
                f"column-aliases-file must be a JSON object: {args.column_aliases_file}",
                {"path": args.column_aliases_file},
            )
        column_aliases = aliases_data

    if args.mode == "charts":
        # charts mode: resolve_charts → plan_charts.json
        try:
            pc = resolve_charts(
                paradigm=args.paradigm,
                columns=columns,
                raw_files=raw_files,
                workspace_dir=args.workspace_dir,
                user_intent=args.user_intent,
                total_subjects=args.total_subjects,
                n_per_group=args.n_per_group,
                n_groups=args.n_groups,
                groups_file=args.groups_file,
                groups=groups,
                columns_file=args.columns_file,
                ev19_template=args.ev19_template,
                virtual_workspace_dir=virtual_workspace_dir,
                column_aliases=column_aliases,
            )
        except ResolveError as e:
            return _emit_error(e.code, str(e), e.details)
        except Exception as e:
            return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

        # P5: 按类型定优先级选图。aggregate 全画优先；per_subject 用剩余预算取代表性子集。
        # 不传 --chart-budget 时全画（select 返回原样，remaining=[]）。
        if args.chart_budget is not None:
            selected, remaining = select_charts_by_priority(pc.charts, budget=args.chart_budget)
            pc.charts = selected
            pc.charts_budget_remaining = remaining
            if remaining:
                # 红线一：预算挤掉产出要留指纹。降级原因写进 notes。
                remaining_ids = sorted({c.id for c in remaining})
                pc.notes.append(
                    f"Chart budget ({args.chart_budget}) cut {len(remaining)} per_subject "
                    f"chart(s) — aggregate plots prioritized; cut chart types: "
                    f"{', '.join(remaining_ids)}. User may re-request more individual plots."
                )

        plan_dict = plan_charts_to_dict(pc)
        out_path.write_text(
            json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = (
            f"PlanCharts written to {args.output}: paradigm={pc.paradigm}, "
            f"charts={len(pc.charts)}, fallback={len(pc.charts_fallback_available)}"
        )
        print(summary)
        return 0

    else:
        # metrics mode (default, backward-compat): resolve_metrics → plan_metrics.json
        try:
            pm = resolve_metrics(
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
                virtual_workspace_dir=virtual_workspace_dir,
                column_aliases=column_aliases,
                overrides=overrides,
                common_catalog=common_catalog,
            )
        except ResolveError as e:
            return _emit_error(e.code, str(e), e.details)
        except Exception as e:
            return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

        plan_dict = plan_metrics_to_dict(pm)
        out_path.write_text(
            json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = (
            f"PlanMetrics written to {args.output}: paradigm={pm.paradigm}, "
            f"metrics={len(pm.metrics)}, skipped={len(pm.skipped)}, statistics="
            f"{'skip' if (pm.statistics and pm.statistics.skip_reason) else 'run' if pm.statistics else 'none'}"
        )
        print(summary)
        return 0


if __name__ == "__main__":
    sys.exit(main())
