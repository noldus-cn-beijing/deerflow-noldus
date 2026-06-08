"""统一的脚本 CLI helper。

所有 ethoinsight.scripts.* 下的脚本通过本模块统一参数解析、I/O 接口。

接口约定见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §3.2
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ============================================================================
# Stdout marker - subagent uses this to extract per-script result
# ============================================================================

RESULT_MARKER = "[result]"


def emit_result(payload: dict[str, Any]) -> None:
    """Print a `[result] {json}` line to stdout for subagent extraction.

    Also runs the L-A safety net (NaN / Inf only) on the payload and prints
    VALIDATION_ERROR lines when violations are found.  Range checks (ratio /
    pct / non-negative) are NOT done here — they are L-B's job, run against
    the catalog's output_unit by ``ethoinsight.validate_catalog`` at the
    code-executor layer.  These are informational — the result is still
    emitted so downstream can decide how to handle partial data.
    """
    from ethoinsight.validate import validate_metrics

    # Validate the result value if it looks like a single-metric payload.
    if "metric" in payload and "value" in payload:
        violations = validate_metrics({payload["metric"]: payload["value"]})
        for v in violations:
            print(
                f"VALIDATION_ERROR: {v['metric']}: {v['issue']} (value={v['value']})"
            )

    print(f"{RESULT_MARKER} {json.dumps(payload, ensure_ascii=False)}")


# ============================================================================
# Virtual → real path resolution（补 JSON 内路径不经 bash 替换的缺口）
# ============================================================================

# 已知的沙箱虚拟路径前缀（稳定契约，见 sandbox/local/local_sandbox_provider.py
# _USER_DATA_VIRTUAL_PREFIX / sandbox tools.py _build_path_env）。
# 排序：从长到短（最长前缀优先匹配，/mnt/user-data/workspace 优先于 /mnt/user-data）。
_KNOWN_SANDBOX_PREFIXES = [
    "/mnt/user-data/workspace",
    "/mnt/user-data/uploads",
    "/mnt/user-data/outputs",
    "/mnt/user-data",
    "/mnt/shared",
    "/mnt/skills",
    "/mnt/acp-workspace",
]


def _sandbox_env_key_for_prefix(prefix: str) -> str:
    """Compute the ``DEERFLOW_PATH_*`` env key for a given container_path prefix.

    生成规则与 ``local_sandbox.py:338`` / ``tools.py:557`` 精确对称：
        ``"DEERFLOW_PATH_" + container_path.strip("/").replace("/", "_").replace("-", "_").upper()``

    因为是固定前缀（不含动态参数），生成是确定性的，无需反向有损还原。
    """
    return "DEERFLOW_PATH_" + prefix.strip("/").replace("/", "_").replace("-", "_").upper()


def resolve_sandbox_path(path: str | Path) -> Path:
    """把 ``/mnt/<x>/...`` 虚拟沙箱路径解析成真实路径。

    设计依据：local sandbox 给脚本进程注入 ``DEERFLOW_PATH_*`` 环境变量
    （见 sandbox/local/local_sandbox.py）。命令行参数里的 /mnt 路径由 bash
    工具替换，但**从 JSON 内部读出的路径字符串**不经替换——本函数补这个解析。

    - 输入是 /mnt/... 且能匹配到 DEERFLOW_PATH_* env → 返回真实路径。
    - 输入已是真实路径（不以 /mnt 开头）→ 原样返回（Path）。
    - 输入是 /mnt 但匹配不到 env（如非沙箱环境/直接跑测试）→ 原样返回
      （fail-safe，行为与修复前一致，不引入新失败模式）。
    """
    p = str(path)
    if not p.startswith("/mnt/"):
        return Path(p)

    # 最长前缀优先匹配
    for prefix in _KNOWN_SANDBOX_PREFIXES:
        if p.startswith(prefix + "/") or p == prefix:
            env_key = _sandbox_env_key_for_prefix(prefix)
            real_base = os.environ.get(env_key)
            if real_base is not None:
                suffix = p[len(prefix):].lstrip("/")
                return Path(real_base) / suffix if suffix else Path(real_base)

    # 匹配不到 → 原样（fail-safe：非沙箱环境/测试直接运行）
    return Path(p)


# ============================================================================
# File I/O
# ============================================================================


def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically (temp file + rename), creating parent dirs.

    Note: tempfile.mkstemp() creates the temp file with 0o600 (owner-only). The
    metric JSONs must be world-readable so the L-B catalog validator
    (``python -m ethoinsight.validate_catalog``) — which may run under a different
    sandbox uid — can read them. So we relax to 0o644 after the atomic rename.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
        os.chmod(p, 0o644)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def read_inputs_json(path: str | Path) -> list[str]:
    """Read a JSON file containing a list of input file paths.

    Format: ``["/path/to/subject1.txt", "/path/to/subject2.txt", ...]``

    Virtual sandbox paths (``/mnt/...``) are resolved to real host paths
    via :func:`resolve_sandbox_path` so callers can ``Path().open()`` them
    directly.  Already-real paths pass through unchanged.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must be a JSON array of file paths, got {type(data).__name__}"
        )
    return [str(resolve_sandbox_path(item)) for item in data]


def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    """Read a JSON file containing the groups mapping.

    Format: ``{"group_name": ["subject_path_1", "subject_path_2"], ...}``

    Subject paths (inside the per-group lists) are resolved via
    :func:`resolve_sandbox_path` — virtual ``/mnt/...`` paths become real
    host paths; already-real paths pass through unchanged.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object mapping group names to subject lists"
        )
    return {k: [str(resolve_sandbox_path(s)) for s in v] for k, v in data.items()}


def resolve_per_subject_input(args: argparse.Namespace) -> str:
    """For per-subject plots: return the single trajectory path from args.

    Accepts either:
      - ``--input <path>`` (legacy single-file form)
      - ``--inputs <json>`` (uniform form; reads the FIRST path from the JSON array)

    Raises ValueError if neither is provided or inputs.json is empty.
    """
    if getattr(args, "input", None):
        return args.input
    if getattr(args, "inputs", None):
        paths = read_inputs_json(args.inputs)
        if not paths:
            raise ValueError(f"{args.inputs} is an empty JSON array")
        return paths[0]
    raise ValueError("plot script requires --input or --inputs")


# ============================================================================
# Standard argparse builders for the three script types
# ============================================================================


def make_compute_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``compute_*.py``: --input + --output + --parameters-json."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--input", required=True, help="Path to a single EthoVision trajectory file"
    )
    ap.add_argument("--output", required=True, help="Path to write the metric JSON")
    # === Sprint 2b ===
    ap.add_argument(
        "--parameters-json",
        default="{}",
        help=(
            "JSON dict of parameters (e.g. velocity_threshold, pendulum_*). "
            "Empty dict means use script-side defaults. "
            "Populated by catalog.resolve from PlanMetric.parameters_in_use."
        ),
    )
    return ap


def parse_parameters(args_namespace: argparse.Namespace) -> dict[str, float | int | str]:
    """Parse args.parameters_json into dict. Returns {} on empty/invalid (with stderr warning).

    Sprint 2b: shared by all compute_*.py scripts for uniform parameter parsing.
    """
    raw = getattr(args_namespace, "parameters_json", "") or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[warning] --parameters-json parse failed ({e}); falling back to script defaults", file=sys.stderr)
        return {}
    if not isinstance(result, dict):
        print(f"[warning] --parameters-json must be a JSON object, got {type(result).__name__}", file=sys.stderr)
        return {}
    return result


def make_plot_parser(
    description: str, *, supports_groups: bool = False
) -> argparse.ArgumentParser:
    """Argparse for ``plot_*.py``.

    Single-file plots use ``--input``; aggregated plots use ``--inputs`` + optional ``--groups``.
    """
    ap = argparse.ArgumentParser(description=description)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", help="Path to a single trajectory file (single-file plots)"
    )
    group.add_argument(
        "--inputs",
        help="Path to a JSON file containing a list of trajectory file paths",
    )
    if supports_groups:
        ap.add_argument(
            "--groups",
            help="Path to a JSON file mapping group_name -> [subject_name, ...]",
        )
    ap.add_argument("--output", required=True, help="Path to write the PNG plot")
    return ap


def make_stats_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``run_*_stats.py``: --inputs --groups --output."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--inputs",
        required=True,
        help="Path to a JSON file containing a list of trajectory file paths",
    )
    ap.add_argument(
        "--groups",
        required=True,
        help="Path to a JSON file mapping group_name -> [subject_name, ...]",
    )
    ap.add_argument("--output", required=True, help="Path to write the stats JSON")
    return ap
