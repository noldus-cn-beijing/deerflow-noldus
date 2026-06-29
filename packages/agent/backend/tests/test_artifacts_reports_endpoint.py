"""新端点 GET /api/threads/{thread_id}/artifacts/reports 的单测。

plan: thread 级资产面板需稳定显示报告（report.md / *.html）。此前报告只在 LangGraph
state.artifacts（lead present_files 才有、subagent 边界丢失、不确定）。本端点按磁盘
直取 outputs/ 下 .md/.html 文档产物，与 charts 端点对称——磁盘是唯一真相。
"""

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


def test_list_report_artifacts_returns_md_and_html(tmp_path, monkeypatch) -> None:
    """磁盘有 report.md + summary.html → 返回 2 条 report 产物，带 filename/ext。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "report.md").write_text("# 报告", encoding="utf-8")
    (outputs_dir / "summary.html").write_text("<h1>x</h1>", encoding="utf-8")
    # 干扰项：图不算报告
    (outputs_dir / "plot_box.png").write_bytes(b"\x89PNG")

    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    result = artifacts_router.list_report_artifacts("thread-1")

    assert isinstance(result, list)
    assert len(result) == 2
    by_name = {m["filename"]: m for m in result}

    assert by_name["report.md"]["path"] == "/mnt/user-data/outputs/report.md"
    assert by_name["report.md"]["kind"] == "report"
    assert by_name["report.md"]["ext"] == "md"
    assert by_name["summary.html"]["ext"] == "html"


def test_list_report_artifacts_excludes_non_report_files(tmp_path, monkeypatch) -> None:
    """只列 .md/.html；.png/.csv/.json 等不算报告。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "plot.png").write_bytes(b"\x89PNG")
    (outputs_dir / "data.csv").write_text("a,b", encoding="utf-8")
    (outputs_dir / "plan_metrics.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    result = artifacts_router.list_report_artifacts("thread-1")

    assert result == []


def test_list_report_artifacts_empty_when_no_outputs_dir(tmp_path, monkeypatch) -> None:
    """0 产物 thread：返回空列表，不报错。"""
    outputs_dir = tmp_path / "outputs"  # 不 mkdir
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    result = artifacts_router.list_report_artifacts("thread-1")

    assert result == []


def test_list_report_artifacts_route_registered_before_catchall(tmp_path, monkeypatch) -> None:
    """端点经 TestClient 真路径可达：/reports 注册在 catch-all /{path:path} 之前不被吞。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "report.md").write_text("# 报告", encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/reports")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["filename"] == "report.md"
    assert body[0]["kind"] == "report"
