"""C2b acceptance: registry dispatch is byte-identical to direct builder calls,
and the equivalence assertion is non-vacuous (fails if a builder is mis-wired)."""
import json

import pytest

from app.gateway.routers import artifacts as art


@pytest.fixture
def fake_builders(monkeypatch):
    """Distinct, content-rich outputs per kind so a mis-wire is detectable."""
    chart = [{"path": "/mnt/user-data/outputs/p.png", "kind": "chart", "chart_type": "box"}]
    report = [{"path": "/mnt/user-data/outputs/r.html", "kind": "report", "filename": "r.html"}]
    data = [{"path": "/mnt/user-data/outputs/metrics_table.json", "kind": "data", "filename": "metrics_table.json"}]
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: chart)
    monkeypatch.setattr(art, "list_report_artifacts", lambda tid: report)
    monkeypatch.setattr(art, "list_data_artifacts", lambda tid: data)
    return {"chart": chart, "report": report, "data": data}


@pytest.mark.parametrize("kind", ["chart", "report", "data"])
def test_dispatch_byte_identical_to_direct_builder(kind, fake_builders):
    via_registry = art.list_artifacts_by_kind(kind, "t1", None)
    direct = fake_builders[kind]
    # Byte-level equivalence: serialize both, compare exact bytes.
    assert json.dumps(via_registry) == json.dumps(direct)


def test_equivalence_is_non_vacuous(monkeypatch, fake_builders):
    """Anti-vacuous probe: mis-wire the chart builder to the report output →
    the byte-equivalence assertion for 'chart' MUST now fail."""
    # Point chart's builder at the report content (simulate a registry mis-wire).
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: fake_builders["report"])
    via_registry = art.list_artifacts_by_kind("chart", "t1", None)
    assert json.dumps(via_registry) != json.dumps(fake_builders["chart"])
