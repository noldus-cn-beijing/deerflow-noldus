"""报告多格式导出转换器（spec 2026-06-29-report-export-formats-impl）。

把已 seal 的 report.html 转成 PDF / docx / tex。纯函数 + 惰性 import（守导入环铁律：
weasyprint / pypandoc 的 import 全放函数体内，模块顶层不引——否则可能闭成
app.gateway → report_exporter → weasyprint → ... 的导入环，生产启动崩）。

选型据 spike：PDF=WeasyPrint（纯 Python、CSS 分页好）、docx/tex=pandoc（一个依赖
覆盖两种）。LaTeX 分支必须预处理 base64 data-uri——pandoc 的 latex writer 不支持
data-uri（``\\includegraphics`` 要文件路径，且不会像 docx writer 那样自动落盘），
裸跑静默丢图。spike 实证：preprocess 后才产 ``\\includegraphics``。
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path

# 已 seal 的 report.html 里图已是 data:image/png;base64,... 形态（seal 时占位符已解析）。
# 这里只处理 data-uri；与 seal_handoff_tools 的 {{img:}} 占位符正则不同源（不同阶段）。
_DATA_URI_RE = re.compile(r"data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)")

# 三种导出格式
EXPORT_FORMATS = ("pdf", "docx", "tex")

# 格式 → (mime_type, 扩展名)
_FORMAT_META = {
    "pdf": ("application/pdf", ".pdf"),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "tex": ("application/x-tex", ".tex"),
}


class ExportError(Exception):
    """转换失败（依赖缺失 / pandoc 报错 / 源 HTML 不可读 / 格式非法）。"""


def export_report(report_html_path: Path, fmt: str, *, base_filename: str = "report") -> tuple[bytes, str, str]:
    """转 report.html → 指定格式。

    Args:
        report_html_path: 已 seal 的 report.html 物理路径（含 base64 内联图）。
        fmt: "pdf" | "docx" | "tex"。
        base_filename: 下载文件名主干（不含扩展名），默认 "report"。

    Returns:
        (content_bytes, mime_type, filename) —— 供端点构造 Response + Content-Disposition。

    Raises:
        ExportError: 源文件不存在 / 格式非法 / 转换器报错。
    """
    if fmt not in EXPORT_FORMATS:
        raise ExportError(f"unsupported export format: {fmt!r} (allowed: {EXPORT_FORMATS})")
    if not report_html_path.is_file():
        raise ExportError(f"report.html not found: {report_html_path}")

    if fmt == "pdf":
        content = _to_pdf(report_html_path)
    elif fmt == "docx":
        content = _to_docx(report_html_path)
    else:  # tex
        content = _to_tex(report_html_path)

    mime, ext = _FORMAT_META[fmt]
    return content, mime, f"{base_filename}{ext}"


def _read_html(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise ExportError(f"report.html unreadable: {e}") from e


def _to_pdf(report_html_path: Path) -> bytes:
    """PDF via WeasyPrint（spike 选型）。惰性 import。"""
    try:
        from weasyprint import HTML  # 惰性：守导入环 + 缺依赖时清晰报错
    except ImportError as e:
        raise ExportError("weasyprint not installed (pip install weasyprint)") from e
    try:
        return HTML(filename=str(report_html_path)).write_pdf()
    except Exception as e:
        raise ExportError(f"weasyprint PDF conversion failed: {e}") from e


def _to_docx(report_html_path: Path) -> bytes:
    """docx via pandoc。data-uri pandoc 原生处理（落 word/media），无需预处理。惰性 import。"""
    try:
        import pypandoc
    except ImportError as e:
        raise ExportError("pypandoc not installed (pip install pypandoc_binary)") from e
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.docx"
        try:
            pypandoc.convert_file(str(report_html_path), "docx", outputfile=str(out))
        except Exception as e:
            raise ExportError(f"pandoc docx conversion failed: {e}") from e
        return out.read_bytes()


def _to_tex(report_html_path: Path) -> bytes:
    """tex via pandoc。**必须预处理 base64 data-uri**（spike：latex writer 不支持 data-uri，
    裸跑静默丢图）。把每个 data-uri 解码成临时图片、src 改相对路径、传 --resource-path。"""
    try:
        import pypandoc
    except ImportError as e:
        raise ExportError("pypandoc not installed (pip install pypandoc_binary)") from e
    html = _read_html(report_html_path)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # 预处理：data-uri → 临时图片 + 相对 src。用递增 index 作文件名，避免 hash 碰撞覆盖。
        counter = {"i": 0}

        def _extract(m: re.Match[str]) -> str:
            ext, b64 = m.group(1), m.group(2)
            ext = "jpg" if ext == "jpeg" else ext
            name = f"img_{counter['i']}.{ext}"
            counter["i"] += 1
            (td_path / name).write_bytes(base64.b64decode(b64))
            return name

        preprocessed = _DATA_URI_RE.sub(_extract, html)
        src_in_td = td_path / "report.html"
        src_in_td.write_text(preprocessed, encoding="utf-8")
        out = td_path / "report.tex"
        try:
            pypandoc.convert_file(
                str(src_in_td),
                "latex",
                outputfile=str(out),
                extra_args=[f"--resource-path={td}"],
            )
        except Exception as e:
            raise ExportError(f"pandoc tex conversion failed: {e}") from e
        return out.read_bytes()
