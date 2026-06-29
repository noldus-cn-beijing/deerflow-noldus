"""report_exporter 单测（spec 2026-06-29-report-export-formats-impl §4.1）。

转换器看**真产物**不看 mock（spec §六.5：data-uri 在各格式的行为只能真跑 pandoc/weasyprint
才知，spike 教训）。这些测试真跑 weasyprint + pandoc（pypandoc_binary 自带 pandoc 3.x）。

fixture 与 test_seal_html_report 同款：tmp_path + outputs + 真 PNG + 构造 report.html
（含 base64 内联图 + 表格），模拟已 seal 的 report.html 形态。
"""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

import pytest

from deerflow.tools.builtins.report_exporter import ExportError, export_report

# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------


def _real_png_bytes() -> bytes:
    """生成一张合法的最小 PNG（weasyprint/pandoc 都能嵌）。Pillow 经 weasyprint 间接可用。"""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _make_report_html(outputs_dir: Path, *, with_image: bool = True) -> Path:
    """构造已 seal 形态的 report.html：含 base64 内联图 + 表格。"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    img_tag = ""
    if with_image:
        b64 = base64.b64encode(_real_png_bytes()).decode("ascii")
        img_tag = f'<figure><img src="data:image/png;base64,{b64}" alt="代表图"/></figure>'
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>报告</title></head><body>"
        "<h1>实验概况</h1>"
        f"{img_tag}"
        "<table><thead><tr><th>组别</th><th>n</th></tr></thead>"
        "<tbody><tr><td>Control</td><td>6</td></tr>"
        "<tr><td>Treatment</td><td>6</td></tr></tbody></table>"
        "</body></html>"
    )
    report = outputs_dir / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# §1 PDF（WeasyPrint）
# ---------------------------------------------------------------------------


class TestExportPdf:
    def test_export_pdf_returns_bytes_and_image_present(self, tmp_path):
        """export_report(report, "pdf") 返回合法 PDF（%PDF 头）且非空字节。"""
        report = _make_report_html(tmp_path / "outputs")
        content, mime, filename = export_report(report, "pdf")

        assert mime == "application/pdf"
        assert filename == "report.pdf"
        assert content[:5] == b"%PDF-"  # 合法 PDF 魔数
        assert len(content) > 1000  # 含图 + 表格，体积有下限


# ---------------------------------------------------------------------------
# §2 docx（pandoc）
# ---------------------------------------------------------------------------


class TestExportDocx:
    def test_export_docx_returns_valid_zip_with_media(self, tmp_path):
        """docx 是合法 zip，含 word/media/ 且至少 1 个 png（data-uri 落 word/media）。"""
        report = _make_report_html(tmp_path / "outputs")
        content, mime, filename = export_report(report, "docx")

        assert mime.startswith("application/vnd.openxmlformats")  # docx mime
        assert filename == "report.docx"
        assert content[:2] == b"PK"  # zip 魔数
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            media = [n for n in names if n.startswith("word/media/")]
            assert media, f"docx 应含 word/media/（data-uri 落盘），实际 names: {names[:10]}"
            png_media = [n for n in media if n.endswith(".png")]
            assert png_media, "至少 1 个 png 在 word/media/"


# ---------------------------------------------------------------------------
# §3 tex（pandoc + data-uri 预处理）—— LaTeX 坑的回归门
# ---------------------------------------------------------------------------


class TestExportTex:
    def test_export_tex_contains_includegraphics(self, tmp_path):
        """tex 文本含 \\includegraphics（预处理后正确生成）。

        这是 LaTeX 坑的回归门——裸跑（不预处理 data-uri）会产空 figure、无 \\includegraphics，
        测试会红。spike 实证。
        """
        report = _make_report_html(tmp_path / "outputs")
        content, mime, filename = export_report(report, "tex")

        assert mime == "application/x-tex"
        assert filename == "report.tex"
        text = content.decode("utf-8", errors="replace")
        assert "\\includegraphics" in text, "tex 应含 \\includegraphics（data-uri 已预处理），实际丢了图"

    def test_export_tex_without_image_still_works(self, tmp_path):
        """无图 report 转 tex 不崩（边界）。"""
        report = _make_report_html(tmp_path / "outputs", with_image=False)
        content, mime, filename = export_report(report, "tex")

        assert mime == "application/x-tex"
        assert filename == "report.tex"
        assert len(content) > 0
        text = content.decode("utf-8", errors="replace")
        # 无图就不该有 includegraphics
        assert "\\includegraphics" not in text


# ---------------------------------------------------------------------------
# §4 错误边界
# ---------------------------------------------------------------------------


class TestExportErrors:
    def test_unsupported_format_raises(self, tmp_path):
        """export_report(report, "xlsx") 抛 ExportError。"""
        report = _make_report_html(tmp_path / "outputs")
        with pytest.raises(ExportError, match="unsupported export format"):
            export_report(report, "xlsx")

    def test_missing_report_raises(self, tmp_path):
        """源文件不存在抛 ExportError（不抛 FileNotFoundError）。"""
        missing = tmp_path / "outputs" / "report.html"
        with pytest.raises(ExportError, match="report.html not found"):
            export_report(missing, "pdf")
