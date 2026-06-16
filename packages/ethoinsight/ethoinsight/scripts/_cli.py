"""统一的脚本 CLI helper。

所有 ethoinsight.scripts.* 下的脚本通过本模块统一参数解析、I/O 接口。

接口约定见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §3.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


def resolve_sandbox_path(
    path: str | Path, workspace_base: str | Path | None = None
) -> Path:
    """把 ``/mnt/<x>/...`` 虚拟沙箱路径解析成真实路径。

    设计依据：local sandbox 给脚本进程注入 ``DEERFLOW_PATH_*`` 环境变量
    （见 sandbox/local/local_sandbox.py）。命令行参数里的 /mnt 路径由 bash
    工具替换，但**从 JSON 内部读出的路径字符串**不经替换——本函数补这个解析。

    解析优先级：
      1. ``DEERFLOW_PATH_*`` env（沙箱子进程/进程池 worker 内：env 已设，原路径）。
      2. ``workspace_base`` 兜底（**仅对 workspace 前缀**）——当 ethoinsight 函数被
         harness **进程内直接调用**（不经沙箱子进程、未设 env）时，调用方传入真实
         workspace 物理路径即可可靠解析，不再依赖 env。这是 2026-06-16 故障族根治
         （#5/#6a 是该族两个已点修实例）：让"进程内调 ethoinsight 读 workspace /mnt
         文件"不再依赖调用方记得设 env。其他前缀（uploads/outputs）不在此兜底——
         它们的兜底由 prep 已做的 ``replace_virtual_path`` 在传入前解决。
      3. 都没有 → 原样返回（fail-safe，与历史等价）。

    ⚠️ 隐性契约显性化：harness 进程内直接调 ethoinsight 读 /mnt 文件时，**必须传
    ``workspace_base``（或先用 ``replace_virtual_path`` 预解析），不能依赖 env**——
    否则本函数 fail-safe 原样返回 /mnt，下游读到的是物理上不存在的路径、静默退化。

    - 输入已是真实路径（不以 /mnt 开头）→ 原样返回（Path），``workspace_base`` 无影响。
    """
    p = str(path)
    if not p.startswith("/mnt/"):
        return Path(p)

    # 最长前缀优先匹配 env（保持历史循环语义：某前缀无 env 不阻断，继续找更短前缀
    # 的 env——如 workspace env 缺失时退化到 user-data env）。
    for prefix in _KNOWN_SANDBOX_PREFIXES:
        if p.startswith(prefix + "/") or p == prefix:
            env_key = _sandbox_env_key_for_prefix(prefix)
            real_base = os.environ.get(env_key)
            if real_base is not None:
                suffix = p[len(prefix):].lstrip("/")
                return Path(real_base) / suffix if suffix else Path(real_base)

    # 所有前缀都无 env：对 workspace 前缀用 workspace_base 兜底（进程内被 harness
    # 直接调、未设 env 时：用真实 workspace 物理路径拼后缀）。**在 env 全扫完之后**
    # 才兜底——保证 env 始终优先于 workspace_base（env 优先路径字节不变）。
    if workspace_base and (
        p.startswith("/mnt/user-data/workspace/")
        or p == "/mnt/user-data/workspace"
    ):
        suffix = p[len("/mnt/user-data/workspace"):].lstrip("/")
        return Path(workspace_base) / suffix if suffix else Path(workspace_base)

    # 都没有 → 原样（fail-safe：非沙箱环境/测试直接运行）
    # 可观测信号：收到 /mnt 路径却既无 env 又无 workspace_base 兜底，原样返回时记
    # 一行 debug（非 warning——正常沙箱外测试也走这条，不该刷 warning）。给未来排查
    # "读不到 /mnt 文件"提供 grep 锚点（守 feedback_isolate_root_cause_*：合法静默
    # 路径不响亮，但留痕）。
    logger.debug(
        "resolve_sandbox_path: 虚拟路径未解析（无 env/无兜底），原样返回 %s", p
    )
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
    # resolve /mnt 虚拟沙箱路径（run_metric_plan 进程内执行无 bash mount 时必需；
    # fail-safe 幂等——真实路径/bash-mounted 路径原样返回，零行为变化）。与
    # read_inputs_json/read_groups_json 在同一 I/O 边界对称收口（2026-06-15 spec #2）。
    p = Path(resolve_sandbox_path(path))
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

    Both the JSON file itself and the listed file paths are resolved to real
    host paths via :func:`resolve_sandbox_path` — virtual ``/mnt/...`` paths
    become real so callers can ``Path().open()`` them directly; already-real
    paths pass through unchanged.

    Spec 2026-06-16 缺陷 1a：``path`` 参数本身也要 resolve（statistics 链读
    workspace 下的 inputs.json，其 /mnt 虚拟路径此前无人 resolve → FileNotFoundError）。
    与 ``save_output_json`` 在同一 I/O 边界对称（#2 当时漏了这两个读函数）。
    """
    p = Path(resolve_sandbox_path(path))
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must be a JSON array of file paths, got {type(data).__name__}"
        )
    return [str(resolve_sandbox_path(item)) for item in data]


def read_groups_json(path: str | Path) -> dict[str, list[str]]:
    """Read groups.json and return ``{group_name: [subject_path, ...]}``.

    SSOT 文件格式（``prep_metric_plan`` 写、``metric_aggregation`` 主读）：
    ``{subject_file: group_name}``（flat map）。本函数读入后**反转**成下游
    ``compute_paradigm_metrics`` / ``compare_groups`` 期望的
    ``{group_name: [subject_path, ...]}``。subject 路径经 ``resolve_sandbox_path``
    解析（虚拟 ``/mnt/...`` → 真实）。

    兼容两种输入（用首个 value 的类型判别，避免破坏遗留形态）：
      - ``{file: group}``（str value）→ SSOT flat map，反转。
      - ``{group: [files]}``（list value）→ 已是目标形态，直通。

    Spec 2026-06-16 缺陷 1a+1b：①``path`` 参数也要 resolve（statistics 链读
    workspace 下 groups.json 的 /mnt 虚拟路径此前 FileNotFoundError）；②旧的 docstring
    认为文件就是 ``{group: [files]}``，与 SSOT ``{file: group}`` 不符——透传会把
    字符串组名当可迭代拆字符。修法是函数内反转（派生视图，非双存——守 SSOT 铁律，
    格式只存一份，写方/主读方都不动）。
    """
    p = Path(resolve_sandbox_path(path))
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object, got {type(data).__name__}"
        )

    inverted: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            # 遗留形态 {group: [files]}：直通 + resolve。
            inverted[k] = [str(resolve_sandbox_path(s)) for s in v]
        else:
            # SSOT 形态 {file: group}：反转成 {group: [file]}。
            group = str(v)
            inverted.setdefault(group, []).append(str(resolve_sandbox_path(k)))
    return inverted


def bridge_groups_to_subjects(
    groups: dict[str, list[str]],
    file_subjects: dict[str, str],
) -> dict[str, list[str]]:
    """Translate file-path groups into subject-key groups for the dispatcher.

    ``read_groups_json`` yields ``{group: [file_path, ...]}`` (SSOT keys files),
    but ``compute_paradigm_metrics`` matches each group member against
    ``parse_batch()["subjects"]`` **keys** — which are EV19 "对象名称" values
    (frequently blank → ``''`` / ``'_1'`` / …), NOT file paths. Without a bridge
    the match set is always empty and ``comparisons`` is always empty on real
    data (spec 2026-06-16 第三层 bug).

    ``parse_batch`` now returns ``file_subjects`` (``{file_path: subject_key}``).
    This helper rewrites each group's file paths into the corresponding subject
    keys **by file**, not by positional index — so a file silently dropped by
    ``parse_batch`` (non-existent / non-EthoVision) simply drops out of its
    group instead of shifting every subsequent group member onto the wrong file.

    Both sides resolve paths via :func:`resolve_sandbox_path` upstream
    (``read_groups_json`` and ``read_inputs_json``), so the path strings match
    directly. A path with no parsed subject (filtered out) is skipped; a
    ``stderr`` note is emitted so the drop is observable, never silent.
    """
    bridged: dict[str, list[str]] = {}
    for group, file_paths in groups.items():
        subject_keys: list[str] = []
        for fp in file_paths:
            key = file_subjects.get(fp)
            if key is None:
                print(
                    f"[warning] groups file has no parsed subject (filtered "
                    f"out?), dropping from group '{group}': {fp}",
                    file=sys.stderr,
                )
                continue
            subject_keys.append(key)
        bridged[group] = subject_keys
    return bridged


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
