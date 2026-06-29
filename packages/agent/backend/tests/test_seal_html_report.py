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

    def test_unmatched_placeholder_degrades_to_empty_src(self, tmp_path):
        """占位符 basename 不在 chart_files → 降级为空 src（不破坏 HTML、不留裸占位符）。

        spec 2026-06-29 方案 A：占位符在 ``<img src="...">`` 的值位置，缺图时返回
        空串使 ``<img src="">`` 失效、前端显示 alt；诊断信息进日志，不塞进 src
        （文本 stub 进 src 会变成坏 src 且经 sanitize 残留裸文本）。
        """
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        _make_png(outputs, "real.png")
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/real.png"])

        report = outputs / "report.html"
        report.write_text('<img src="{{img:phantom.png}}" alt="phantom">', encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        assert "{{img:" not in out  # 占位符已消除
        assert "data:image/png;base64," not in out  # 绝不产伪 data URI
        assert "data:image png" not in out  # ``/`` 未被拆
        assert "&lt;img" not in out  # 无嵌套

    def test_missing_file_falls_back_to_empty_src(self, tmp_path):
        """chart_files 声明了 basename 但磁盘 .png 不存在 → 不内联 data URI，降级空 src。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        # 不写 plot_box.png（chart_files 声称有，磁盘没有）
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box.png"])

        report = outputs / "report.html"
        report.write_text('<img src="{{img:plot_box.png}}" alt="plot_box">', encoding="utf-8")

        _resolve_html_report_image_placeholders(report, ws)

        out = report.read_text(encoding="utf-8")
        # 绝不产出一个指向不存在 data 的伪 data URI；降级为空 src（alt 显示）
        assert "data:image/png;base64," not in out
        assert "{{img:" not in out
        assert "&lt;img" not in out

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
# §1.5 占位符→内联后产物必须能被 HTML 渲染器正确解析（治嵌套 <img> 回归）
#
# spec: docs/superpowers/specs/2026-06-29-fix-html-report-inline-img.md
#
# prompt 让 LLM 把占位符写进 ``<img src="{{img:X}}">``（占位符是 src 的值），
# 因此 seal 的替换必须只产 data URI 值，不能产整个 ``<img>`` 元素——否则得到
# 嵌套 ``<img src="<img ...">">``，经 sanitize_report_html 后 ``/`` 被拆成空格、
# 图全废。下方断言看**真产物能不能被渲染**（不只看占位符是否被替换），
# 正是 #234 漏测导致带病合入的那条契约。
# ---------------------------------------------------------------------------


class TestHtmlPlaceholderNoNestingContract:
    def test_resolved_img_is_value_not_nested_element(self, tmp_path):
        """``<img src="{{img:X}}">`` 替换后是合法、无嵌套的 data URI src 值。

        这条断言若存在于 #234，带病的「产整个 <img>」版本会立刻红：
        嵌套产物经 sanitize 后出现 ``&lt;img``、``data:image png``（``/`` 被拆），
        规范子串 ``data:image/png;base64,`` 被破坏。
        """
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"image-data" * 4
        _make_png(outputs, "plot_box.png", png_bytes)
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box.png"])

        report = outputs / "report.html"
        report.write_text(
            '<figure><img src="{{img:plot_box.png}}"><figcaption>cap</figcaption></figure>',
            encoding="utf-8",
        )

        _resolve_html_report_image_placeholders(report, ws)
        out = report.read_text(encoding="utf-8")

        # 规范 data URI 子串完整存在（``/`` 未被拆）
        assert "data:image/png;base64," in out
        # 不应出现嵌套 <img>（替换前 prompt 已写了外层 <img src="...">，
        # 替换只应填值，不得再产第二个 <img）
        assert out.count("<img") == 1
        assert "&lt;img" not in out  # 未经 sanitize 也不应有转义嵌套 img

    def test_end_to_end_through_sanitizer_parseable(self, tmp_path):
        """resolve + sanitize 全链后产物可被 HTMLParser 无错解析、图可见。

        模拟前端实际消费路径：report.html 先过占位符解析、再过确定性消毒。
        断言最终 HTML 里 ``data:image/png;base64,`` 数 == 占位符数、``/`` 未被拆、
        无 ``&lt;img``，且可被 HTMLParser 干净重解析。
        """
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        png_a = b"\x89PNG\r\n\x1a\n" + b"AAAA" * 4
        png_b = b"\x89PNG\r\n\x1a\n" + b"BBBB" * 4
        _make_png(outputs, "a.png", png_a)
        _make_png(outputs, "b.png", png_b)
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/a.png", "/mnt/user-data/outputs/b.png"])

        report = outputs / "report.html"
        report.write_text(
            '<h1>R</h1><p>a <img src="{{img:a.png}}"> b <img src="{{img:b.png}}"></p>',
            encoding="utf-8",
        )

        _resolve_html_report_image_placeholders(report, ws)
        clean = sanitize_report_html(report.read_text(encoding="utf-8"))

        # 嵌套被拆的特征必须缺席
        assert "&lt;img" not in clean
        assert "data:image png" not in clean
        # 规范 data URI 子串 == 占位符数（每张图一条完整 data URI）
        assert clean.count("data:image/png;base64,") == 2
        # 产物可被标准 HTMLParser 干净重解析（结构稳定，重跑不丢图）
        reparsed = sanitize_report_html(clean)
        assert reparsed.count("data:image/png;base64,") == 2

    def test_missing_image_degrades_without_breaking_html(self, tmp_path):
        """缺图时占位符降级为不破坏 HTML 的形态，整篇仍可被 HTMLParser 解析。

        spec 方案 A：缺图返回空串（``<img src="">``），前端加载失败显示 alt。
        关键：缺图分支绝不能产文本 stub 塞进 src（那会变成坏 src 且残留裸文本）。
        """
        ws = tmp_path / "workspace"
        ws.mkdir()
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        # chart_files 声称有，磁盘没有
        _make_chart_handoff(ws, ["/mnt/user-data/outputs/ghost.png"])

        report = outputs / "report.html"
        report.write_text(
            '<p>ok</p><img src="{{img:ghost.png}}" alt="ghost"><p>after</p>',
            encoding="utf-8",
        )

        _resolve_html_report_image_placeholders(report, ws)
        clean = sanitize_report_html(report.read_text(encoding="utf-8"))

        # 占位符已消除，不产伪 data URI，整篇结构完整可重解析
        assert "{{img:" not in clean
        assert "data:image/png;base64," not in clean
        assert "ok" in clean and "after" in clean
        # 重解析稳定（HTMLParser 不抛、结构文本保留）
        reparsed = sanitize_report_html(clean)
        assert "ok" in reparsed and "after" in reparsed


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

    def test_title_dropped_with_content(self):
        """``<title>`` 内容必须连同标签一起删除，不能裸露。

        dogfood thread 73b41dc3 的 report.html 证实：LLM 产 ``<head><title>EPM…报告</title></head>``，
        旧实现把 title 放进 ``_STRIP_TAGS_KEEP_CONTENT``（剥标签留内容）→ 标题文字裸露在
        ``<head>`` 里 → 前端 DOMPurify 解析时该孤儿文本落入 body → 报告顶部冒出一行裸标题。
        治本：title 进 ``_DROP_TAGS_WITH_CONTENT``，整段删除（标题文字本就不该进正文渲染区）。
        """
        html = (
            '<html lang="zh"><head><title>EPM 高架十字迷宫行为学研究报告</title></head>'
            "<body><h2>1. 实验概况</h2><blockquote>正文</blockquote></body></html>"
        )
        clean = sanitize_report_html(html)
        assert "<title" not in clean.lower()
        assert "EPM 高架十字迷宫行为学研究报告" not in clean
        # 正文内容不受影响
        assert "<h2>1. 实验概况</h2>" in clean
        assert "<blockquote>正文</blockquote>" in clean
