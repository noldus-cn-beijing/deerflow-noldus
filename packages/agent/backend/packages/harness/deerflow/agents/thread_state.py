from typing import Annotated, NotRequired, TypedDict

from langchain.agents import AgentState


class SandboxState(TypedDict):
    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]
    shared_path: NotRequired[str | None]


class ViewedImageData(TypedDict):
    base64: str
    mime_type: str


def merge_sandbox(existing: SandboxState | None, new: SandboxState | None) -> SandboxState | None:
    """Reducer for sandbox state - accepts idempotent writes only.

    Multiple sandbox tools can initialize lazily in the same graph step and
    emit the same sandbox_id via Command(update=...). LangGraph needs an
    explicit reducer for that shared state key. Different sandbox ids in the
    same thread indicate a lifecycle/isolation bug, so fail closed instead of
    choosing one silently.
    """
    if new is None:
        return existing
    if existing is None:
        return new

    existing_id = existing.get("sandbox_id")
    new_id = new.get("sandbox_id")
    if existing_id == new_id:
        return existing
    raise ValueError(f"Conflicting sandbox state updates: {existing_id!r} != {new_id!r}")


SandboxStateField = Annotated[NotRequired[SandboxState | None], merge_sandbox]


def _artifact_path(artifact: str | dict) -> str:
    """提取 artifact 的去重键 path。

    artifacts 契约（spec 2026-06-24-frontend-phase0-3-artifact-gallery）：可以是
    裸 string（旧数据/报告/技能等无元数据产物）或 ArtifactMeta dict（chart 产物
    带 chart_id/output_mode/paradigm/metric/subject/group/chart_type/thumb_path）。
    去重一律按 ``path`` —— 整对象相等会把「同 path 不同轮次补全的元数据」当两条。
    """
    if isinstance(artifact, dict):
        path = artifact.get("path")
        if isinstance(path, str) and path:
            return path
        return ""  # 无 path 的 dict（异常形态）退化为空串
    return str(artifact)


def _artifact_run_id(artifact: str | dict) -> str:
    """提取 artifact 的 run 维度键（spec 2026-06-26 §任务2 路 A）。

    chart 元数据带 ``run_id`` 时按 (run_id, path) 去重——同一 thread 内多 run 若产
    同名 chart（chart_id 来自 catalog 固定、与 run 无关），跨 run 不再互覆盖；同 run
    同 path 仍覆盖（追问轮补全元数据）。裸 string / 无 run_id 的 dict 退化为空串，
    与历史「按 path 去重」行为完全一致（向后兼容旧产物）。
    """
    if isinstance(artifact, dict):
        rid = artifact.get("run_id")
        if isinstance(rid, str) and rid:
            return rid
    return ""


def merge_artifacts(existing: list | None, new: list | None) -> list:
    """Reducer for artifacts list - merges and deduplicates artifacts.

    兼容两种形态（spec §3.1）：裸 string（向后兼容锚点）与 ArtifactMeta dict。
    去重键是 ``(run_id, path)``（spec 2026-06-26 §任务2 路 A）：同 (run_id, path)
    的产物新值覆盖旧值；跨 run 同 path 不互覆盖（"113 图只显示 1 张"家族病根之一）。
    无 run_id 的产物退化为 ("", path) —— 等价于旧的「按 path 去重」语义。
    """
    if existing is None:
        return new or []
    if new is None:
        return existing

    # 按 (run_id, path) 去重；新值覆盖旧值（new 在后写入覆盖 existing）。
    merged: dict[tuple[str, str], str | dict] = {}
    for artifact in [*existing, *new]:
        key = (_artifact_run_id(artifact), _artifact_path(artifact))
        # 无 path 的项（空串键）各自保留——用 id 做兜底键避免互吞。
        if key[1] == "":
            key = ("", f"__no_path_{id(artifact)}")
        merged[key] = artifact
    return list(merged.values())


def merge_charts_status(existing: dict | None, new: dict | None) -> dict | None:
    """Reducer for charts_status (failed/remaining chart summary).

    chart-maker 的 ``handoff_chart_maker.json`` 里的 failed_charts/remaining_charts
    不在 thread state（前端拿不到，spec §四 Step 5「拿不到记后端小补丁」）。present_file
    写 chart 产物时一并把这份摘要接进 state，前端 inline 摘要行据此显式呈现「N 张未生成」，
    守「无声截断」铁律。

    语义：new 是 None → 保留 existing（节点没碰）；new 非空（含空 dict）→ 覆盖
    （present_file 每次以最新 plan 的 failed/remaining 为准，累积无意义）。
    """
    if new is None:
        return existing
    return new


def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """Reducer for viewed_images dict - merges image dictionaries.

    Special case: If new is an empty dict {}, it clears the existing images.
    This allows middlewares to clear the viewed_images state after processing.
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # Special case: empty dict means clear all viewed images
    if len(new) == 0:
        return {}
    # Merge dictionaries, new values override existing ones for same keys
    return {**existing, **new}


def merge_todos(existing: list | None, new: list | None) -> list | None:
    """Reducer for todos list - keeps the last non-None value.

    Semantics:
    - If `new` is None (node didn't touch todos), preserve `existing`.
    - If `new` is provided (even empty list), it represents an explicit
      update and wins over `existing`.
    """
    if new is None:
        return existing
    return new


class PromotedTools(TypedDict):
    catalog_hash: str
    names: list[str]


def merge_promoted(existing: PromotedTools | None, new: PromotedTools | None) -> PromotedTools | None:
    """Reducer for deferred-tool promotions, scoped by catalog hash.

    - new None/empty -> preserve existing (node didn't touch promotions).
    - catalog_hash changed -> replace wholesale, dropping stale names (prevents a
      persisted bare name from exposing a different tool after catalog drift).
    - same catalog_hash -> union names, dedupe, preserve order.
    """
    if not new:
        return existing
    if existing is None or existing.get("catalog_hash") != new["catalog_hash"]:
        return {
            "catalog_hash": new["catalog_hash"],
            "names": list(dict.fromkeys(new["names"])),
        }
    return {
        "catalog_hash": existing["catalog_hash"],
        "names": list(dict.fromkeys(existing["names"] + new["names"])),
    }


class ThreadState(AgentState):
    sandbox: SandboxStateField
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list, merge_artifacts]
    todos: Annotated[list | None, merge_todos]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # image_path -> {base64, mime_type}
    promoted: Annotated[PromotedTools | None, merge_promoted]
    charts_status: Annotated[NotRequired[dict | None], merge_charts_status]  # failed/remaining chart summary (spec phase0-3)
