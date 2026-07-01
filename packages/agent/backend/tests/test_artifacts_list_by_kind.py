from app.gateway.routers import artifacts as art


def test_dispatch_chart_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/x.png", "kind": "chart"}]
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: sentinel)
    assert art.list_artifacts_by_kind("chart", "t1", None) == sentinel


def test_dispatch_report_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/r.md", "kind": "report"}]
    monkeypatch.setattr(art, "list_report_artifacts", lambda tid: sentinel)
    assert art.list_artifacts_by_kind("report", "t1", None) == sentinel


def test_dispatch_data_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/d.json", "kind": "data"}]
    monkeypatch.setattr(art, "list_data_artifacts", lambda tid: sentinel)
    assert art.list_artifacts_by_kind("data", "t1", None) == sentinel


def test_dispatch_unknown_kind_raises():
    import pytest
    with pytest.raises(KeyError):
        art.list_artifacts_by_kind("nope", "t1", None)
