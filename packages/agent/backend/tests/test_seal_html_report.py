"""seal_report_writer_handoff 的 HTML 模式：占位符→base64 内联 + XSS 消毒。

spec: docs/superpowers/specs/2026-06-29-html-report-format.md

报告从 Markdown 改 HTML 后，seal 时新增两层确定性处理：
1. ``{{img:<basename>}}`` 占位符 → 读对应 .png → base64 → 内联
   ``<img src="data:image/png;base64,...">``（少量代表性图，自包含、离线可看）。
2. HTML 消毒：剥 ``<script>`` / ``on*`` 内联事件 / ``<iframe>`` 等，防 LLM 产出注入。

验收看真 report.html（reward hacking：不看 handoff 自述）。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from deerflow.tools.builtins.seal_handoff_tools import (
    _resolve_html_report_image_placeholders,
    sanitize_report_html,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_chart_handoff(workspace: Path, chart_files: list[str]) -> None:
    """写 handoff_chart_maker.json（chart_files 是 /mnt/user-data/outputs/ 虚拟路径）。"""
    data = {
        "status": "completed",
        "paradigm": "epm",
        "summary": "charts done",
        "chart_files": chart_files,
    }
    (workspace / "handoff_chart_maker.json").write_text(json.dumps(data), encoding="utf-8")


def _make_png(outputs_dir: Path, name: str, payload: bytes = b"\x89PNG\r\n\x1a\nFAKE") -> Path:
    """写一个最小 PNG 文件到 outputs/。"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    p = outputs_dir / name
    p.write_bytes(payload)
    return p


# ---------------------------------------------------------------------------
# §1 占位符 → base64 内联
# ---------------------------------------------------------------------------


class TestHtmlPlaceholderBase64Inline:
    def test_placeholder_replaced_with_data_uri(self, tmp_path):
        """{{img:x.png}} → <img src="data:image/png;base64,...">，base64 与磁盘内容一致。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"image-data" * 4
        _make_png(outputs, "plot_box.png", png_bytes)
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box.png"])

        report = outputs / "report.html"
        report.write_text('<figure><img src="{{img:plot_box.png}}"></figure>', encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        expected_b64 = base64.b64encode(png_bytes).decode("ascii")
        assert f"data:image/png;base64,{expected_b64}" in out
        # 占位符全消
        assert "{{img:" not in out

    def test_all_placeholders_consumed(self, tmp_path):
        """多个占位符一次性全部内联，无残留。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        _make_png(outputs, "a.png")
        _make_png(outputs, "b.png")
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/a.png", "/mnt/user-data/outputs/b.png"])

        report = outputs / "report.html"
        report.write_text(
            '<p>a <img src="{{img:a.png}}"> b <img src="{{img:b.png}}"></p>',
            encoding="utf-8",
        )

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        assert out.count("data:image/png;base64,") == 2
        assert "{{img:" not in out

    def test_unmatched_placeholder_visible_stub(self, tmp_path):
        """占位符 basename 不在 chart_files → 替换成可见错误 stub（不留裸占位符）。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        _make_png(outputs, "real.png")
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/real.png"])

        report = outputs / "report.html"
        report.write_text('<img src="{{img:phantom.png}}">', encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        assert "{{img:" not in out
        assert "phantom.png" in out  # 可见 stub 列出未找到的文件

    def test_missing_file_falls_back_to_textual_note(self, tmp_path):
        """chart_files 声明了 basename 但磁盘 .png 不存在 → 不内联 data URI，留可见提示。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        # 不写 plot_box.png（chart_files 声称有，磁盘没有）
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box.png"])

        report = outputs / "report.html"
        report.write_text('<img src="{{img:plot_box.png}}">', encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        # 绝不产出一个指向不存在 data 的伪 data URI；留可见提示让用户诊断
        assert "data:image/png;base64," not in out
        assert "plot_box.png" in out

    def test_no_placeholders_noop(self, tmp_path):
        """无占位符的报告原样不变。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        report = tmp_path / "report.html"
        original = "<h1>报告</h1><p>无图</p>"
        report.write_text(original, encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        assert report.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# §2 HTML 消毒（<script> / on* / <iframe>）
# ---------------------------------------------------------------------------


class TestHtmlSanitization:
    def test_script_tag_stripped(self):
        html = "<p>ok</p><script>alert(1)</script><p>after</p>"
        clean = sanitize_report_html(html)
        assert "<script" not in clean.lower()
        assert "alert(1)" not in clean
        assert "ok" in clean and "after" in clean

    def test_inline_event_handlers_removed(self):
        html = '<img src="x.png" onerror="alert(1)" onload="evil()">'
        clean = sanitize_report_html(html)
        low = clean.lower()
        assert "onerror" not in low
        assert "onload" not in low
        assert "alert" not in clean

    def test_iframe_removed(self):
        html = '<iframe src="https://evil"></iframe><p>keep</p>'
        clean = sanitize_report_html(html)
        assert "<iframe" not in clean.lower()
        assert "keep" in clean

    def test_inline_event_case_insensitive_removed(self):
        """大小写混淆的内联事件（OnErRoR / OnLoad）也要剥——浏览器对事件名大小写不敏感。"""
        html = '<p OnErRoR="x" OnLoad="y">t</p>'
        clean = sanitize_report_html(html)
        low = clean.lower()
        assert "onerror" not in low
        assert "onload" not in low

    def test_safe_content_preserved(self):
        """正常结构化 HTML（标题/表格/段落/列表/data URI 图）原样保留。"""
        html = (
            "<!DOCTYPE html><html><body>"
            "<h1>实验概况</h1>"
            '<table><tr><th>指标</th><td>1.0</td></tr></table>'
            '<ul><li>项 A</li></ul>'
            '<img src="data:image/png;base64,QUJD">'
            "</body></html>"
        )
        clean = sanitize_report_html(html)
        for fragment in (
            "<h1>实验概况</h1>",
            "<table>",
            "<ul><li>项 A</li></ul>",
            "data:image/png;base64,QUJD",
        ):
            assert fragment in clean
