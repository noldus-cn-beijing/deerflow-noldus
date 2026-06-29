"""GET /api/threads/{tid}/artifacts/report/export 端点单测（spec §4.2）。

复用 test_artifacts_reports_endpoint 的模式：monkeypatch resolve_thread_virtual_path
把 /outputs 指到 tmp_path，用 _router_auth_helpers.make_authed_test_app 过 @require_permission。
端点测试验**路由/权限/响应头/状态码**（转换器内部已由 test_report_exporter 真产物覆盖，
spec §六.5：端点测试可 mock 转换器边界；这里真跑转换器，因 fixture 小且转换 <1s）。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient

import app.gateway.routers.artifacts as artifacts_router


def _resolve_factory(outputs_dir: Path):
    """替身 resolve_thread_virtual_path：/mnt/user-data/outputs → outputs_dir。"""

    def _resolve(_thread_id: str, virtual_path: str) -> Path:
        vp = virtual_path.rstrip("/")
        if vp.endswith("/outputs"):
            return outputs_dir
        return outputs_dir / "_other"

    return _resolve


def _real_png_b64() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (50, 100, 200)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_report(outputs_dir: Path) -> None:
    """写一份含 base64 内联图 + 表格的 report.html 到 outputs/。"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    b64 = _real_png_b64()
    html = f'<!DOCTYPE html><html><body><h1>报告</h1><img src="data:image/png;base64,{b64}"/><table><tr><td>Control</td><td>6</td></tr></table></body></html>'
    (outputs_dir / "report.html").write_text(html, encoding="utf-8")


def _client(outputs_dir: Path) -> TestClient:
    """构造已 patch 路径解析 + stub auth 的 TestClient app。"""
    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    # patch 必须在 client 发请求前生效； TestClient context manager 装载 app。
    return app


# ---------------------------------------------------------------------------
# §1 各格式返回合法 attachment
# ---------------------------------------------------------------------------


class TestExportEndpointFormats:
    def test_export_endpoint_pdf_returns_attachment(self, tmp_path, monkeypatch):
        """?format=pdf → 200、Content-Disposition attachment filename*=...report.pdf、body 是 PDF。"""
        outputs_dir = tmp_path / "outputs"
        _write_report(outputs_dir)
        monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

        app = _client(outputs_dir)
        with TestClient(app) as client:
            resp = client.get("/api/threads/thread-1/artifacts/report/export", params={"format": "pdf"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/pdf")
        cd = resp.headers["content-disposition"]
        assert "attachment" in cd
        assert "report.pdf" in cd  # RFC 5987 filename*=UTF-8''report.pdf
        assert resp.content[:5] == b"%PDF-"


# ---------------------------------------------------------------------------
# §2 错误状态码
# ---------------------------------------------------------------------------


class TestExportEndpointErrors:
    def test_export_endpoint_bad_format_400(self, tmp_path, monkeypatch):
        """?format=xlsx → 400。"""
        outputs_dir = tmp_path / "outputs"
        _write_report(outputs_dir)
        monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

        app = _client(outputs_dir)
        with TestClient(app) as client:
            resp = client.get("/api/threads/thread-1/artifacts/report/export", params={"format": "xlsx"})

        assert resp.status_code == 400
        assert "unsupported format" in resp.json()["detail"]

    def test_export_endpoint_missing_report_404(self, tmp_path, monkeypatch):
        """线程 outputs/ 无 report.html → 404。"""
        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)  # 有目录但无 report.html
        monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

        app = _client(outputs_dir)
        with TestClient(app) as client:
            resp = client.get("/api/threads/thread-1/artifacts/report/export", params={"format": "pdf"})

        assert resp.status_code == 404
        assert "report.html not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# §3 路由顺序门：export 端点不被 catch-all /artifacts/{path:path} 吞
# ---------------------------------------------------------------------------


def test_export_endpoint_registered_before_catchall(tmp_path, monkeypatch):
    """/artifacts/report/export 不被 catch-all /artifacts/{path:path} 吞——真路径可达且 200。

    若注册顺序错（catch-all 在前），此路径会被 get_artifact 接管、按 {path:path} 解析
    "report/export" 当文件名 → 404。这里 200 = export 端点命中，证明顺序对。
    """
    outputs_dir = tmp_path / "outputs"
    _write_report(outputs_dir)
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    app = _client(outputs_dir)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/report/export", params={"format": "pdf"})

    # 200 + application/pdf = export_report_artifact 命中（catch-all 会返 404 或非 pdf）
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")
