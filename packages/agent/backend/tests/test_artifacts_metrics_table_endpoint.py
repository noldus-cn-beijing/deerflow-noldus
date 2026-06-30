"""spec 2026-06-30 C1 模块2 — Tests for the metrics-table endpoints.

Covers:
- GET /artifacts/data-table → serves the real metrics_table.csv (attachment), 404 if absent.
- list_data_artifacts → surfaces metrics_table.json on disk.
- GET /artifacts/metrics-table → returns the clean parsed JSON, 404 if absent.
- /data + /metrics-table routes registered before the catch-all (TestClient 200, not swallowed).

Pattern mirrors test_artifacts_reports_endpoint.py: monkeypatch
``resolve_thread_virtual_path`` to a tmp outputs_dir, then call the internal
fns directly + a TestClient route test via ``make_authed_test_app``.
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


_METRICS_CSV = "subject,group,open_arm_time_ratio\nctrl,Control,0.3\ndrug,Treatment,0.55\n"

_METRICS_JSON = (
    '{"paradigm": "epm", "metric_names": ["open_arm_time_ratio"], '
    '"groups": [{"group": "Control", "n": 1, "metrics": {"open_arm_time_ratio": {"mean": 0.3, "std": null, "n": 1}}}], '
    '"per_subject": [{"subject": "ctrl", "group": "Control", '
    '"values": {"open_arm_time_ratio": 0.3}, "outlier_flags": {"open_arm_time_ratio": false}}]}'
)


# ---------------------------------------------------------------------------
# data-table CSV download
# ---------------------------------------------------------------------------


def test_export_data_table_returns_real_metrics_table_csv(tmp_path, monkeypatch) -> None:
    """outputs/ 有 metrics_table.csv → GET /data-table 返回真 CSV（attachment）。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "metrics_table.csv").write_text(_METRICS_CSV, encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/data-table")

    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert "metrics_table.csv" in resp.headers.get("content-disposition", "")
    assert resp.text == _METRICS_CSV


def test_export_data_table_404_when_absent(tmp_path, monkeypatch) -> None:
    """无 metrics_table.csv → 404。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/data-table")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# list_data_artifacts
# ---------------------------------------------------------------------------


def test_list_data_artifacts_surfaces_metrics_table(tmp_path, monkeypatch) -> None:
    """磁盘有 metrics_table.json → 返回 1 条 data 产物，带 filename/ext/kind=data。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "metrics_table.json").write_text(_METRICS_JSON, encoding="utf-8")
    # 干扰项：图/报告不算 data 类。
    (outputs_dir / "plot.png").write_bytes(b"\x89PNG")
    (outputs_dir / "report.md").write_text("# x", encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    result = artifacts_router.list_data_artifacts("thread-1")

    assert len(result) == 1
    item = result[0]
    assert item["path"] == "/mnt/user-data/outputs/metrics_table.json"
    assert item["kind"] == "data"
    assert item["filename"] == "metrics_table.json"
    assert item["ext"] == "json"


def test_list_data_artifacts_empty_when_no_outputs(tmp_path, monkeypatch) -> None:
    """0 产物 / 无目录：返回空列表，不报错。"""
    outputs_dir = tmp_path / "outputs"  # 不 mkdir
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    result = artifacts_router.list_data_artifacts("thread-1")
    assert result == []


# ---------------------------------------------------------------------------
# metrics-table JSON
# ---------------------------------------------------------------------------


def test_get_metrics_table_returns_clean_json(tmp_path, monkeypatch) -> None:
    """GET /metrics-table → 200，返回含 groups/per_subject 的干净 JSON，无内脏键。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "metrics_table.json").write_text(_METRICS_JSON, encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/metrics-table")

    assert resp.status_code == 200
    body = resp.json()
    assert body["paradigm"] == "epm"
    assert body["groups"][0]["group"] == "Control"
    assert body["per_subject"][0]["subject"] == "ctrl"
    # 反 vacuous：不含 handoff 内脏键。
    viscera = {"gate_signals", "handoff", "assessment", "statistics", "confidence", "inputs", "sealed_by"}
    assert viscera.isdisjoint(body.keys())


def test_get_metrics_table_404_when_absent(tmp_path, monkeypatch) -> None:
    """无 metrics_table.json → 404。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        resp = client.get("/api/threads/thread-1/artifacts/metrics-table")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Route registration before catch-all
# ---------------------------------------------------------------------------


def test_data_and_metrics_table_routes_registered_before_catchall(tmp_path, monkeypatch) -> None:
    """/data 与 /metrics-table 经 TestClient 真 200（注册在 catch-all /{path:path} 前不被吞）。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "metrics_table.json").write_text(_METRICS_JSON, encoding="utf-8")
    (outputs_dir / "metrics_table.csv").write_text(_METRICS_CSV, encoding="utf-8")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        # /data 若被 catch-all 吞，会走 get_artifact 当文件解析（非 JSON list）→ 断言失败。
        data_resp = client.get("/api/threads/thread-1/artifacts/data")
        json_resp = client.get("/api/threads/thread-1/artifacts/metrics-table")

    assert data_resp.status_code == 200
    assert isinstance(data_resp.json(), list)
    assert data_resp.json()[0]["kind"] == "data"
    assert json_resp.status_code == 200
    assert "groups" in json_resp.json()
