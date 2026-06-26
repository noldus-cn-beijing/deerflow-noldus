from pathlib import Path
from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.sandbox.tools import replace_virtual_path
from deerflow.tools.types import Runtime

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
PLAN_CHARTS_DEFAULT_VIRTUAL = f"{VIRTUAL_PATH_PREFIX}/workspace/plan_charts.json"
HANDOFF_CHART_MAKER_VIRTUAL = f"{VIRTUAL_PATH_PREFIX}/workspace/handoff_chart_maker.json"

# 缩略图目标边长（spec §3.1.6：Pillow Image.thumbnail((400,400)) → 几十 KB）。
THUMBNAIL_MAX_SIZE = (400, 400)


def _get_thread_id(runtime: Runtime) -> str | None:
    """Resolve the current thread id from runtime context or RunnableConfig."""
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id:
        return thread_id

    runtime_config = getattr(runtime, "config", None) or {}
    thread_id = runtime_config.get("configurable", {}).get("thread_id")
    if thread_id:
        return thread_id

    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None


def _get_run_id(runtime: Runtime) -> str | None:
    """Resolve the current run id from runtime context (spec 2026-06-26 §任务2).

    run worker（runtime/runs/worker.py）把 ``run_id`` 注入 runtime.context，
    task_tool 透传到 subagent。present_files 在 lead / subagent 调用时从 context
    取当前 run，stamp 进 chart 元数据，让 merge_artifacts 按 (run_id, path) 去重。
    非 run 上下文（如直接 tool 调用 / 旧入口）→ None，meta 不写 run_id 字段。
    """
    if runtime.context:
        run_id = runtime.context.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return None


def _normalize_presented_filepath(
    runtime: Runtime,
    filepath: str,
) -> str:
    """Normalize a presented file path to the `/mnt/user-data/outputs/*` contract.

    Accepts either:
    - A virtual sandbox path such as `/mnt/user-data/outputs/report.md`
    - A host-side thread outputs path such as
      `/app/backend/.deer-flow/threads/<thread>/user-data/outputs/report.md`

    Returns:
        The normalized virtual path.

    Raises:
        ValueError: If runtime metadata is missing or the path is outside the
            current thread's outputs directory.
    """
    if runtime.state is None:
        raise ValueError("Thread runtime state is not available")

    thread_id = _get_thread_id(runtime)
    if not thread_id:
        raise ValueError("Thread ID is not available in runtime context or runtime config")

    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("Thread outputs path is not available in runtime state")

    outputs_dir = Path(outputs_path).resolve()
    stripped = filepath.lstrip("/")
    virtual_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")

    if stripped == virtual_prefix or stripped.startswith(virtual_prefix + "/"):
        try:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath, user_id=get_effective_user_id())
        except TypeError:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath)
    else:
        actual_path = Path(filepath).expanduser().resolve()

    try:
        relative_path = actual_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Only files in {OUTPUTS_VIRTUAL_PREFIX} can be presented: {filepath}") from exc

    if not actual_path.exists():
        raise ValueError(f"File does not exist: {filepath}")

    return f"{OUTPUTS_VIRTUAL_PREFIX}/{relative_path.as_posix()}"


# ============================================================================
# chart 元数据接出（spec 2026-06-24-frontend-phase0-3-artifact-gallery，决策1=路 A）
# plan_charts.json 是 chart 元数据 SSOT；present_file 只「关联接出」不重算分类。
# ============================================================================


def _derive_chart_type(chart_id: str, script: str) -> str | None:
    """从 chart_id/script 确定性推导 chart_type（box/bar/trajectory/...）。

    spec §3.1 SSOT 守则：chart_type 若 plan_charts.json 没直接给，由后端一处推导，
    不在前端各猜一遍。命名约定 ``plot_<...>_<type>``（trajectory/timeseries/box/bar...）。
    """
    text = f"{chart_id} {script}".lower()
    for token in ("trajectory", "timeseries", "time_series", "box", "bar", "heatmap", "violin", "scatter", "line"):
        if token in text:
            return "timeseries" if token in ("timeseries", "time_series") else token
    return None


def _load_plan_charts(thread_data: ThreadDataState | None) -> dict[str, Any] | None:
    """读 plan_charts.json，返回 {paradigm, by_output: {output_virtual: entry}}。

    plan 默认在 /mnt/user-data/workspace/plan_charts.json（与 run_chart_plan 默认值一致）。
    缺失/不可解析 → None（present_file 仍按裸 path 写，向后兼容）。
    """
    if not thread_data:
        return None
    try:
        plan_real = replace_virtual_path(PLAN_CHARTS_DEFAULT_VIRTUAL, thread_data)
        if not Path(plan_real).exists():
            return None
        import json

        plan = json.loads(Path(plan_real).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — plan 缺失/坏掉不影响 present 主路径
        return None

    by_output: dict[str, dict[str, Any]] = {}
    for entry in plan.get("charts", []) or []:
        output = entry.get("output")
        if isinstance(output, str) and output:
            by_output[output] = entry
    return {"paradigm": plan.get("paradigm", ""), "by_output": by_output}


def _load_chart_maker_handoff(thread_data: ThreadDataState | None) -> dict[str, Any] | None:
    """读 handoff_chart_maker.json 取 failed_charts/remaining_charts（spec §四 Step 5）。

    run_chart_plan 封存的 handoff 含 chart_files / failed_charts / remaining_charts。
    前端原本拿不到（不在 state）；present_file 一并接出摘要进 charts_status。
    缺失 → None。
    """
    if not thread_data:
        return None
    try:
        handoff_real = replace_virtual_path(HANDOFF_CHART_MAKER_VIRTUAL, thread_data)
        if not Path(handoff_real).exists():
            return None
        import json

        return json.loads(Path(handoff_real).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _real_to_outputs_virtual(real_path: Path, thread_data: ThreadDataState | None) -> str | None:
    """把 outputs/ 下的真实路径还原成 /mnt/user-data/outputs/<name> 虚拟路径。"""
    if not thread_data:
        return None
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        return None
    try:
        rel = real_path.relative_to(Path(outputs_path).resolve())
    except ValueError:
        return None
    return f"{OUTPUTS_VIRTUAL_PREFIX}/{rel.as_posix()}"


def _generate_thumbnail(source_real: Path, thread_data: ThreadDataState | None) -> str | None:
    """用 Pillow 为一张 chart png 生成缩略图，返回 thumb 的虚拟路径（spec §3.1.6）。

    trajectory 实测 1–2.7MB/张是画廊主成本；缩略图把网络/解码成本 ① 砍 1-2 个数量级。
    生成失败（无 Pillow/读图失败/写盘失败）→ 返回 None，前端退化原 path + decoding=async。
    thumb 文件名：``<stem>.thumb.webp``，与原图同目录（outputs/）。
    """
    try:
        from PIL import Image
    except ImportError:
        return None  # Pillow 缺失不阻塞

    try:
        thumb_real = source_real.with_name(f"{source_real.stem}.thumb.webp")
        if not thumb_real.exists():
            with Image.open(source_real) as img:
                img.thumbnail(THUMBNAIL_MAX_SIZE)
                img.convert("RGB").save(thumb_real, format="WEBP", quality=80)
        return _real_to_outputs_virtual(thumb_real, thread_data)
    except Exception:  # noqa: BLE001 — 缩略图是增强项，任何失败都退化原图
        return None


def _build_artifact_meta(
    normalized_virtual: str,
    plan: dict[str, Any] | None,
    thread_data: ThreadDataState | None,
    generate_thumb: bool,
    *,
    run_id: str | None = None,
) -> str | dict[str, Any]:
    """为一张已 normalize 的虚拟路径构造 ArtifactMeta（或退回裸 string）。

    路径命中 plan_charts.json.output → chart dict（带元数据 + thumb_path）；
    否则 → 裸 string（报告/技能等无 chart 元数据产物，向后兼容）。

    ``run_id``（spec 2026-06-26 §任务2 路 A）：chart 元数据带 run 维度，让
    merge_artifacts 按 (run_id, path) 去重——同 thread 多 run 产同名 chart 不互覆盖。
    None（非 run 上下文）→ 不写字段，与旧 meta 字节兼容。
    """
    if not plan:
        return normalized_virtual

    entry = plan.get("by_output", {}).get(normalized_virtual)
    if not entry:
        return normalized_virtual

    chart_id = str(entry.get("id") or entry.get("output") or "")
    meta: dict[str, Any] = {
        "path": normalized_virtual,
        "kind": "chart",
        "chart_id": chart_id or None,
        "output_mode": entry.get("output_mode") or "per_subject",
        "paradigm": plan.get("paradigm") or None,
        "metric": entry.get("metric"),
        "subject": entry.get("subject"),
        "group": entry.get("group"),
        "chart_type": _derive_chart_type(chart_id, str(entry.get("script") or "")),
    }
    if run_id:
        meta["run_id"] = run_id

    if generate_thumb and thread_data:
        real_path = Path(replace_virtual_path(normalized_virtual, thread_data))
        if real_path.exists():
            thumb_virtual = _generate_thumbnail(real_path, thread_data)
            if thumb_virtual:
                meta["thumb_path"] = thumb_virtual
    return meta


def _build_charts_status(thread_data: ThreadDataState | None) -> dict[str, Any] | None:
    """从 handoff_chart_maker.json 构造 charts_status 摘要（failed/remaining）。

    只在有失败/截断时带出（全成功 → None，不污染 state）。前端 inline 据此显式
    呈现「N 张未生成 + 原因」（守「无声截断」铁律）。
    """
    handoff = _load_chart_maker_handoff(thread_data)
    if not handoff:
        return None
    failed = handoff.get("failed_charts") or []
    remaining = handoff.get("remaining_charts") or []
    rendered = handoff.get("chart_files") or []
    if not failed and not remaining:
        return None
    return {
        "n_rendered": len(rendered),
        "failed": [
            {"chart_id": str(f.get("chart_id") or ""), "reason": str(f.get("reason") or "")}
            for f in failed
            if isinstance(f, dict)
        ],
        "remaining": [
            {"chart_id": str(r.get("chart_id") or ""), "reason": str(r.get("reason") or "")}
            for r in remaining
            if isinstance(r, dict)
        ],
    }


@tool("present_files", parse_docstring=True)
def present_file_tool(
    runtime: Runtime,
    filepaths: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Make files visible to the user for viewing and rendering in the client interface.

    When to use the present_files tool:

    - Making any file available for the user to view, download, or interact with
    - Presenting multiple related files at once
    - After creating files that should be presented to the user

    When NOT to use the present_files tool:
    - When you only need to read file contents for your own processing
    - For temporary or intermediate files not meant for user viewing

    Notes:
    - You should call this tool after creating files and moving them to the `/mnt/user-data/outputs` directory.
    - This tool can be safely called in parallel with other tools. State updates are handled by a reducer to prevent conflicts.

    Args:
        filepaths: List of absolute file paths to present to the user. **Only** files in `/mnt/user-data/outputs` can be presented.
    """
    try:
        normalized_paths = [_normalize_presented_filepath(runtime, filepath) for filepath in filepaths]
    except ValueError as exc:
        return Command(
            update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]},
        )

    thread_data = runtime.state.get("thread_data") if runtime.state else None
    # 接出 chart 元数据（spec phase0-3 决策1=路 A）：plan_charts.json 是 SSOT。
    # 任一文件命中 plan 才生成缩略图（避免给报告/技能等非 chart 产物白跑 Pillow）。
    plan = _load_plan_charts(thread_data)
    has_any_chart = plan is not None and any(p in plan.get("by_output", {}) for p in normalized_paths)
    # run_id 维度（spec 2026-06-26 §任务2 路 A）：chart meta 带 run_id，merge_artifacts
    # 按 (run_id, path) 去重——同 thread 多 run 同名 chart 不互覆盖。
    run_id = _get_run_id(runtime)
    artifact_metas: list[str | dict[str, Any]] = [
        _build_artifact_meta(p, plan, thread_data, generate_thumb=has_any_chart, run_id=run_id)
        for p in normalized_paths
    ]

    # 失败/截断摘要接进 state（spec §四 Step 5：前端拿不到 handoff，present 时一并带出）。
    update: dict[str, Any] = {
        "artifacts": artifact_metas,
        "messages": [ToolMessage("Successfully presented files", tool_call_id=tool_call_id)],
    }
    charts_status = _build_charts_status(thread_data)
    if charts_status is not None:
        update["charts_status"] = charts_status

    # The merge_artifacts reducer will handle merging and deduplication (by path).
    return Command(update=update)
