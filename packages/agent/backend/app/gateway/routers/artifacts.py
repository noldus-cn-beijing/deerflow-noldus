import json
import logging
import mimetypes
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse

from app.gateway.authz import require_permission
from app.gateway.path_utils import resolve_thread_virtual_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["artifacts"])

ACTIVE_CONTENT_MIME_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "image/svg+xml",
}

MAX_SKILL_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024
_SKILL_ARCHIVE_READ_CHUNK_SIZE = 64 * 1024

# 产物画廊（spec 2026-06-24-frontend-phase0-3-artifact-gallery §3.1.7）：流式 zip 打包时
# 排除后端生成的缩略图（<name>.thumb.webp）——它是渲染优化衍生物，不该进「下载全部图」。
_THUMB_SUFFIX = ".thumb.webp"
# 流式 zip 单次读取块大小（不全量进内存）。
_ZIP_STREAM_CHUNK = 64 * 1024


def _build_content_disposition(disposition_type: str, filename: str) -> str:
    """Build an RFC 5987 encoded Content-Disposition header value."""
    return f"{disposition_type}; filename*=UTF-8''{quote(filename)}"


def _build_attachment_headers(filename: str, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Content-Disposition": _build_content_disposition("attachment", filename)}
    if extra_headers:
        headers.update(extra_headers)
    return headers


def is_text_file_by_content(path: Path, sample_size: int = 8192) -> bool:
    """Check if file is text by examining content for null bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
            # Text files shouldn't contain null bytes
            return b"\x00" not in chunk
    except Exception:
        return False


def _read_skill_archive_member(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read a .skill archive member while enforcing an uncompressed size cap."""
    if info.file_size > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
        raise HTTPException(status_code=413, detail="Skill archive member is too large to preview")

    chunks: list[bytes] = []
    total_read = 0
    with zip_ref.open(info, "r") as src:
        while chunk := src.read(_SKILL_ARCHIVE_READ_CHUNK_SIZE):
            total_read += len(chunk)
            if total_read > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
                raise HTTPException(status_code=413, detail="Skill archive member is too large to preview")
            chunks.append(chunk)
    return b"".join(chunks)


def _extract_file_from_skill_archive(zip_path: Path, internal_path: str) -> bytes | None:
    """Extract a file from a .skill ZIP archive.

    Args:
        zip_path: Path to the .skill file (ZIP archive).
        internal_path: Path to the file inside the archive (e.g., "SKILL.md").

    Returns:
        The file content as bytes, or None if not found.
    """
    if not zipfile.is_zipfile(zip_path):
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # List all files in the archive
            infos_by_name = {info.filename: info for info in zip_ref.infolist()}

            # Try direct path first
            if internal_path in infos_by_name:
                return _read_skill_archive_member(zip_ref, infos_by_name[internal_path])

            # Try with any top-level directory prefix (e.g., "skill-name/SKILL.md")
            for name, info in infos_by_name.items():
                if name.endswith("/" + internal_path) or name == internal_path:
                    return _read_skill_archive_member(zip_ref, info)

            # Not found
            return None
    except (zipfile.BadZipFile, KeyError):
        return None


@router.get(
    "/threads/{thread_id}/artifacts/archive",
    summary="Download All Artifacts (ZIP)",
    description="Stream a ZIP of all artifact files in the thread's outputs directory (zero-render bulk download).",
)
@require_permission("threads", "read", owner_check=True)
async def archive_artifacts(thread_id: str, request: Request) -> StreamingResponse:
    """第 1 层主路径（spec §3.1.7）：一次点击带走全部图，零渲染。

    浏览器直接下载，不经任何缩略图渲染。流式 zip：边读边压、不全量进内存。
    排除后端缩略图（``*.thumb.webp``）——它们是渲染衍生物，不该进「全部图」。
    路径必须在 ``{path:path}`` catch-all 之前注册（FastAPI 按声明顺序匹配）。
    """
    outputs_dir = resolve_thread_virtual_path(thread_id, "/mnt/user-data/outputs")
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise HTTPException(status_code=404, detail="No artifacts to archive")

    files: list[Path] = [p for p in sorted(outputs_dir.rglob("*")) if p.is_file() and not p.name.endswith(_THUMB_SUFFIX)]
    if not files:
        raise HTTPException(status_code=404, detail="No artifacts to archive")

    def _stream_zip():
        """流式生成 zip：用 ZipFile 写到 buffer，逐文件 flush。

        每个文件读 _ZIP_STREAM_CHUNK 字节写入 zip 流；buffer 在 yield 后被消费，
        整体内存占用 ≈ 单文件 chunk 而非全部图之和（trajectory 实测 1–2.7MB/张，
        100+ 张全量进内存会爆）。``zipfile`` 不支持真正的 pipe 流式，所以用 SpooledTemporaryFile
        滚动落盘 + 分块读回 yield 的折中（内存有界，超大图集自动溢出磁盘）。
        """
        import io
        import tempfile

        buf = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, suffix=".zip")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fpath in files:
                arcname = fpath.relative_to(outputs_dir).as_posix()
                zinfo = zipfile.ZipInfo.from_file(fpath, arcname)
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                with zf.open(zinfo, mode="w") as dst, open(fpath, "rb") as src:
                    while True:
                        chunk = src.read(_ZIP_STREAM_CHUNK)
                        if not chunk:
                            break
                        dst.write(chunk)
        buf.seek(0)
        try:
            while True:
                chunk = buf.read(_ZIP_STREAM_CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            buf.close()

    filename = f"artifacts-{thread_id[:8]}.zip"
    return StreamingResponse(
        _stream_zip(),
        media_type="application/zip",
        headers={"Content-Disposition": _build_content_disposition("attachment", filename)},
    )


@router.get(
    "/threads/{thread_id}/artifacts/data-table",
    summary="Export Data Table (CSV placeholder)",
    description="Placeholder for data-table CSV export (spec Step 5: full implementation deferred; returns the first .csv artifact or 404).",
)
@require_permission("threads", "read", owner_check=True)
async def export_data_table(thread_id: str, request: Request) -> Response:
    """数据表导出占位（spec §四 Step 5：CSV 完整实现顺延，不阻塞画廊主体）。

    Phase 0：返回 outputs/ 里第一个 ``.csv`` 文件（metric 计算结果若有 CSV 产物即取）。
    无 CSV → 404，前端按钮置灰。完整数据表实现（按指标/组别重组）顺延。
    """
    outputs_dir = resolve_thread_virtual_path(thread_id, "/mnt/user-data/outputs")
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise HTTPException(status_code=404, detail="No data table available")

    csv_files = sorted(p for p in outputs_dir.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise HTTPException(status_code=404, detail="No data table available")

    csv_path = csv_files[0]
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        headers={"Content-Disposition": _build_content_disposition("attachment", csv_path.name)},
    )


# 画廊全量图端点（spec 2026-06-26-artifact-bubbling-report-display-gallery-return-fix §1.1）。
# 磁盘是唯一真相：subagent→lead 边界会丢 state.artifacts（run_chart_plan 写进 subagent
# state 不上行），但图都确定性落盘在 outputs/。画廊直接按磁盘 + plan_charts.json 取图，
# 不再依赖 LangGraph state 冒泡。元数据 join 逻辑与 present_file_tool._build_artifact_meta
# 同构（chart_type 推导、by_output 命中、thumb_path 退化），但走 resolve_thread_virtual_path
# （需 thread_id，无 thread_data），端点侧自洽。
_OUTPUTS_VIRTUAL_PREFIX = "/mnt/user-data/outputs"
_PLAN_CHARTS_VIRTUAL = "/mnt/user-data/workspace/plan_charts.json"


def _derive_chart_type(chart_id: str, script: str) -> str | None:
    """从 chart_id/script 确定性推导 chart_type（与 present_file_tool 同构，SSOT 一处推导）。"""
    text = f"{chart_id} {script}".lower()
    for token in ("trajectory", "timeseries", "time_series", "box", "bar", "heatmap", "violin", "scatter", "line"):
        if token in text:
            return "timeseries" if token in ("timeseries", "time_series") else token
    return None


def _build_chart_by_output(plan: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], str]:
    """把 plan_charts.json 的 charts[] 折叠成 {output_virtual: entry}，便于按磁盘路径反查。

    返回 (by_output, paradigm)。plan 为 None/缺 charts → ({}, "")。
    """
    if not plan:
        return {}, ""
    by_output: dict[str, dict[str, Any]] = {}
    for entry in plan.get("charts", []) or []:
        output = entry.get("output")
        if isinstance(output, str) and output:
            by_output[output] = entry
    return by_output, plan.get("paradigm", "") or ""


def _load_plan_charts(plan_real: Path) -> dict[str, Any] | None:
    """读 plan_charts.json；缺失/坏掉 → None（plan 是增强项，不是前提）。"""
    try:
        if not plan_real.exists():
            return None
        return json.loads(plan_real.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — plan 坏掉不阻塞画廊按磁盘列图
        return None


def list_chart_artifacts(thread_id: str, request: Request | None) -> list[dict[str, Any]]:
    """按磁盘 + plan_charts.json 构造画廊全量图的 ArtifactMeta[]。

    磁盘真实文件即真相：outputs/ 下所有 .png（排除 .thumb.webp 缩略图）。
    每张图路径命中 plan_charts.json.output → 升级成带 chart 元数据的 ArtifactMeta；
    未命中（极少，如 plan 缺失/坏掉/多出孤儿图）→ 退裸 {path}，不丢图。
    缩略图：同 stem 的 <stem>.thumb.webp 若存在则带 thumb_path，否则前端退化原图。
    """
    outputs_dir = resolve_thread_virtual_path(thread_id, _OUTPUTS_VIRTUAL_PREFIX)
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return []

    plan_real = resolve_thread_virtual_path(thread_id, _PLAN_CHARTS_VIRTUAL)
    plan = _load_plan_charts(plan_real)
    by_output, paradigm = _build_chart_by_output(plan)

    metas: list[dict[str, Any]] = []
    outputs_virtual_prefix = _OUTPUTS_VIRTUAL_PREFIX
    for png in sorted(p for p in outputs_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"):
        if png.name.endswith(_THUMB_SUFFIX):
            continue
        rel = png.relative_to(outputs_dir).as_posix()
        virtual = f"{outputs_virtual_prefix}/{rel}"
        entry = by_output.get(virtual)
        if entry:
            chart_id = str(entry.get("id") or entry.get("output") or "")
            meta: dict[str, Any] = {
                "path": virtual,
                "kind": "chart",
                "chart_id": chart_id or None,
                "output_mode": entry.get("output_mode") or "per_subject",
                "paradigm": paradigm or None,
                "metric": entry.get("metric"),
                "subject": entry.get("subject"),
                "group": entry.get("group"),
                "chart_type": _derive_chart_type(chart_id, str(entry.get("script") or "")),
            }
        else:
            meta = {"path": virtual}
        # 缩略图只读不重生成（chart 生成时 Pillow 已产出 <stem>.thumb.webp）。
        thumb = png.with_name(f"{png.stem}.thumb.webp")
        if thumb.exists():
            meta["thumb_path"] = f"{outputs_virtual_prefix}/{thumb.relative_to(outputs_dir).as_posix()}"
        metas.append(meta)
    return metas


@router.get(
    "/threads/{thread_id}/artifacts/charts",
    summary="List Chart Artifacts (disk + plan_charts.json)",
    description="List all generated chart images for the thread, joined with plan_charts.json metadata. Disk is the source of truth (independent of LangGraph state bubbling).",
)
@require_permission("threads", "read", owner_check=True)
async def get_chart_artifacts(thread_id: str, request: Request) -> Response:
    """画廊全量图数据源（spec §1.1）：磁盘 .png + plan_charts.json 元数据 → ArtifactMeta[]。

    注册在 catch-all ``/artifacts/{path:path}`` 之前，避免被吞。
    """
    return Response(
        content=json.dumps(list_chart_artifacts(thread_id, request)),
        media_type="application/json",
    )


# 报告/文档产物清单端点（thread 资产面板，磁盘为真相）。
# report.md / *.html 等文档产物此前只在 LangGraph state.artifacts 里（lead present_files 才有，
# 不确定且 subagent 边界丢失）。资产面板要稳定显示报告，需直接按磁盘列 outputs/ 下的文档产物，
# 与 charts 端点对称。返回虚拟路径，前端用 urlOfArtifact + catch-all 端点取内容。
_REPORT_EXTENSIONS = (".md", ".html", ".htm")


def list_report_artifacts(thread_id: str) -> list[dict[str, Any]]:
    """按磁盘列 outputs/ 下的文档产物（.md/.html），与 list_chart_artifacts 对称。

    纯磁盘扫描，不依赖 state；缺目录返回 []。返回 [{path, kind:"report", filename, ext}]。
    """
    outputs_dir = resolve_thread_virtual_path(thread_id, _OUTPUTS_VIRTUAL_PREFIX)
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for f in sorted(p for p in outputs_dir.rglob("*") if p.is_file() and p.suffix.lower() in _REPORT_EXTENSIONS):
        rel = f.relative_to(outputs_dir).as_posix()
        reports.append(
            {
                "path": f"{_OUTPUTS_VIRTUAL_PREFIX}/{rel}",
                "kind": "report",
                "filename": f.name,
                "ext": f.suffix.lower().lstrip("."),
            }
        )
    return reports


@router.get(
    "/threads/{thread_id}/artifacts/reports",
    summary="List Report Artifacts (disk)",
    description="List all report/document artifacts (.md/.html) on disk for the thread. Disk is the source of truth (independent of LangGraph state bubbling).",
)
@require_permission("threads", "read", owner_check=True)
async def get_report_artifacts(thread_id: str, request: Request) -> Response:
    """报告产物数据源：磁盘 outputs/ 下 .md/.html → [{path, kind, filename, ext}]。

    注册在 catch-all ``/artifacts/{path:path}`` 之前，避免被吞。
    """
    return Response(
        content=json.dumps(list_report_artifacts(thread_id)),
        media_type="application/json",
    )


@router.get(
    "/threads/{thread_id}/artifacts/{path:path}",
    summary="Get Artifact File",
    description="Retrieve an artifact file generated by the AI agent. Text and binary files can be viewed inline, while active web content is always downloaded.",
)
@require_permission("threads", "read", owner_check=True)
async def get_artifact(thread_id: str, path: str, request: Request, download: bool = False) -> Response:
    """Get an artifact file by its path.

    The endpoint automatically detects file types and returns appropriate content types.
    Use the `download` query parameter to force file download for non-active content.

    Args:
        thread_id: The thread ID.
        path: The artifact path with virtual prefix (e.g., mnt/user-data/outputs/file.txt).
        request: FastAPI request object (automatically injected).

    Returns:
        The file content as a FileResponse with appropriate content type:
        - Active content (HTML/XHTML/SVG): Served as download attachment
        - Text files: Plain text with proper MIME type
        - Binary files: Inline display with download option

    Raises:
        HTTPException:
            - 400 if path is invalid or not a file
            - 403 if access denied (path traversal detected)
            - 404 if file not found

    Query Parameters:
        download (bool): If true, forces attachment download for file types that are
            otherwise returned inline or as plain text. Active HTML/XHTML/SVG content
            is always downloaded regardless of this flag.

    Example:
        - Get text file inline: `/api/threads/abc123/artifacts/mnt/user-data/outputs/notes.txt`
        - Download file: `/api/threads/abc123/artifacts/mnt/user-data/outputs/data.csv?download=true`
        - Active web content such as `.html`, `.xhtml`, and `.svg` artifacts is always downloaded
    """
    # Check if this is a request for a file inside a .skill archive (e.g., xxx.skill/SKILL.md)
    if ".skill/" in path:
        # Split the path at ".skill/" to get the ZIP file path and internal path
        skill_marker = ".skill/"
        marker_pos = path.find(skill_marker)
        skill_file_path = path[: marker_pos + len(".skill")]  # e.g., "mnt/user-data/outputs/my-skill.skill"
        internal_path = path[marker_pos + len(skill_marker) :]  # e.g., "SKILL.md"

        actual_skill_path = resolve_thread_virtual_path(thread_id, skill_file_path)

        if not actual_skill_path.exists():
            raise HTTPException(status_code=404, detail=f"Skill file not found: {skill_file_path}")

        if not actual_skill_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {skill_file_path}")

        # Extract the file from the .skill archive
        content = _extract_file_from_skill_archive(actual_skill_path, internal_path)
        if content is None:
            raise HTTPException(status_code=404, detail=f"File '{internal_path}' not found in skill archive")

        # Determine MIME type based on the internal file
        mime_type, _ = mimetypes.guess_type(internal_path)
        # Add cache headers to avoid repeated ZIP extraction (cache for 5 minutes)
        cache_headers = {"Cache-Control": "private, max-age=300"}
        download_name = Path(internal_path).name or actual_skill_path.stem
        if download or mime_type in ACTIVE_CONTENT_MIME_TYPES:
            return Response(content=content, media_type=mime_type or "application/octet-stream", headers=_build_attachment_headers(download_name, cache_headers))

        if mime_type and mime_type.startswith("text/"):
            return PlainTextResponse(content=content.decode("utf-8"), media_type=mime_type, headers=cache_headers)

        # Default to plain text for unknown types that look like text
        try:
            return PlainTextResponse(content=content.decode("utf-8"), media_type="text/plain", headers=cache_headers)
        except UnicodeDecodeError:
            return Response(content=content, media_type=mime_type or "application/octet-stream", headers=cache_headers)

    actual_path = resolve_thread_virtual_path(thread_id, path)

    logger.info(f"Resolving artifact path: thread_id={thread_id}, requested_path={path}, actual_path={actual_path}")

    if not actual_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")

    if not actual_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")

    mime_type, _ = mimetypes.guess_type(actual_path)

    if download:
        return FileResponse(path=actual_path, filename=actual_path.name, media_type=mime_type, headers=_build_attachment_headers(actual_path.name))

    # Always force download for active content types to prevent script execution
    # in the application origin when users open generated artifacts.
    if mime_type in ACTIVE_CONTENT_MIME_TYPES:
        return FileResponse(path=actual_path, filename=actual_path.name, media_type=mime_type, headers=_build_attachment_headers(actual_path.name))

    if mime_type and mime_type.startswith("text/"):
        return PlainTextResponse(content=actual_path.read_text(encoding="utf-8"), media_type=mime_type)

    if is_text_file_by_content(actual_path):
        return PlainTextResponse(content=actual_path.read_text(encoding="utf-8"), media_type=mime_type)

    # Fix 2026-05-28: 改用 FileResponse 而非 Response(read_bytes()) — 后者会同步阻塞 async event loop，
    # 在 SSE stream 高并发期间(如 chart-maker 完成 → 4 个图片同时 GET + suggestions POST + lead final
    # delivery 还在 stream)导致 gateway 整体阻塞，下游请求 504 Gateway Time-out。
    # FileResponse 走 starlette anyio threadpool, 与 async event loop 解耦。
    return FileResponse(
        path=actual_path,
        media_type=mime_type,
        headers={"Content-Disposition": _build_content_disposition("inline", actual_path.name)},
    )
