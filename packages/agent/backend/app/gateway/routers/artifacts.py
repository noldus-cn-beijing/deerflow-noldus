import logging
import mimetypes
import zipfile
from pathlib import Path
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
