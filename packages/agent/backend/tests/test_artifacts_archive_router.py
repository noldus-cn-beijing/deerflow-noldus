"""ZIP archive + CSV data-table export endpoints (spec 2026-06-24-frontend-phase0-3-artifact-gallery §3.1.7/Step5).

第 1 层主路径「下载全部 ZIP」流式打包；CSV 数据表导出占位。
"""

import io
import zipfile

from fastapi.testclient import TestClient
from _router_auth_helpers import make_authed_test_app

import app.gateway.routers.artifacts as artifacts_router


def _stub_outputs_dir(monkeypatch, outputs_dir):
    """把 resolve_thread_virtual_path 指向测试 outputs 目录（/mnt/user-data/outputs 时）。"""
    monkeypatch.setattr(
        artifacts_router,
        "resolve_thread_virtual_path",
        lambda _thread_id, _path: outputs_dir,
    )


def _client():
    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    return TestClient(app)


def test_archive_streams_zip_of_all_outputs(tmp_path, monkeypatch):
    """ZIP 端点：打包 outputs/ 下全部文件，Content-Disposition attachment。"""
    outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "a.png").write_bytes(b"png-a")
    (outputs_dir / "b.png").write_bytes(b"png-b")
    (outputs_dir / "report.md").write_text("r")
    _stub_outputs_dir(monkeypatch, outputs_dir)

    with _client() as client:
        resp = client.get("/api/threads/thread-1/artifacts/archive")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert names == {"a.png", "b.png", "report.md"}
    assert zf.read("a.png") == b"png-a"


def test_archive_excludes_thumbnails(tmp_path, monkeypatch):
    """ZIP 必须排除后端缩略图（*.thumb.webp）——它们是渲染衍生物。"""
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "traj.png").write_bytes(b"png")
    (outputs_dir / "traj.thumb.webp").write_bytes(b"thumb")
    _stub_outputs_dir(monkeypatch, outputs_dir)

    with _client() as client:
        resp = client.get("/api/threads/thread-1/artifacts/archive")

    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "traj.png" in zf.namelist()
    assert all(not n.endswith(".thumb.webp") for n in zf.namelist())


def test_archive_404_when_no_outputs(tmp_path, monkeypatch):
    _stub_outputs_dir(monkeypatch, tmp_path / "empty")  # 不存在
    with _client() as client:
        resp = client.get("/api/threads/thread-1/artifacts/archive")
    assert resp.status_code == 404


def test_export_data_table_returns_first_csv(tmp_path, monkeypatch):
    """CSV 占位：返回 outputs/ 里第一个 .csv。"""
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "metrics.csv").write_text("a,b\n1,2\n")
    _stub_outputs_dir(monkeypatch, outputs_dir)

    with _client() as client:
        resp = client.get("/api/threads/thread-1/artifacts/data-table")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment" in resp.headers["content-disposition"]


def test_export_data_table_404_when_no_csv(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "a.png").write_bytes(b"png")
    _stub_outputs_dir(monkeypatch, outputs_dir)

    with _client() as client:
        resp = client.get("/api/threads/thread-1/artifacts/data-table")
    assert resp.status_code == 404


def test_archive_and_datatable_routes_take_precedence_over_catchall(tmp_path, monkeypatch):
    """archive / data-table 是固定段，必须在 {path:path} catch-all 之前匹配。

    防回归：若有人把 catch-all 移到前面，/archive 会被当成 path='archive' 的文件请求 → 404。
    """
    outputs_dir = tmp_path / "out"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "a.png").write_bytes(b"png")
    _stub_outputs_dir(monkeypatch, outputs_dir)

    with _client() as client:
        # 这两个固定段路径必须命中各自端点而非 get_artifact 的 {path:path}。
        z = client.get("/api/threads/thread-1/artifacts/archive")
        assert z.status_code == 200
        assert z.headers["content-type"] == "application/zip"
