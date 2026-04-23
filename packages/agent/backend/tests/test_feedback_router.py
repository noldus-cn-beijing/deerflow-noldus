import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.gateway.routers import feedback as feedback_mod

    monkeypatch.setattr(feedback_mod, "_base_dir", lambda: tmp_path)

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(feedback_mod.router)
    return TestClient(app), tmp_path


def test_post_feedback_correct_verdict(client):
    c, tmp_path = client
    resp = c.post(
        "/api/threads/thread-42/feedback",
        json={"message_id": "m-1", "verdict": "correct"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True}

    out = tmp_path / "training-data" / "feedback" / "thread-42.jsonl"
    record = json.loads(out.read_text().strip().splitlines()[0])
    assert record["message_id"] == "m-1"
    assert record["verdict"] == "correct"


def test_post_feedback_needs_fix_with_revision(client):
    c, tmp_path = client
    resp = c.post(
        "/api/threads/t/feedback",
        json={
            "message_id": "m-2",
            "verdict": "needs_fix",
            "revised_text": "正确的版本",
            "note": "第二段指标解读方向反了",
        },
    )
    assert resp.status_code == 200
    record = json.loads((tmp_path / "training-data" / "feedback" / "t.jsonl").read_text().strip())
    assert record["revised_text"] == "正确的版本"
    assert record["note"] == "第二段指标解读方向反了"


def test_post_feedback_rejects_invalid_verdict(client):
    c, _ = client
    resp = c.post(
        "/api/threads/t/feedback",
        json={"message_id": "m-1", "verdict": "maybe"},
    )
    assert resp.status_code == 422


def test_post_feedback_disk_failure_returns_500_not_crash(client, monkeypatch):
    """Robustness: a write failure should return an HTTP error, never crash Gateway."""
    c, tmp_path = client
    from app.gateway.routers import feedback as feedback_mod

    def boom(*args, **kwargs):
        raise OSError("simulated disk full")

    # Force file write to fail
    import pathlib
    monkeypatch.setattr(pathlib.Path, "open", boom)

    resp = c.post(
        "/api/threads/t/feedback",
        json={"message_id": "m-1", "verdict": "correct"},
    )
    # Should respond with an HTTP error, not leak the exception.
    assert resp.status_code >= 500


def test_feedback_router_mounted_on_gateway_app():
    from app.gateway.app import app as gateway_app

    paths = {route.path for route in gateway_app.routes if hasattr(route, "path")}
    assert "/api/threads/{thread_id}/feedback" in paths


def test_get_feedback_list(client):
    c, tmp_path = client
    c.post("/api/threads/t-list/feedback", json={"message_id": "m-1", "verdict": "correct"})
    c.post("/api/threads/t-list/feedback", json={"message_id": "m-2", "verdict": "wrong", "revised_text": "R"})

    resp = c.get("/api/threads/t-list/feedback")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    verdicts = {i["message_id"]: i["verdict"] for i in items}
    assert verdicts == {"m-1": "correct", "m-2": "wrong"}


def test_get_feedback_empty_thread_returns_empty_list(client):
    c, _ = client
    resp = c.get("/api/threads/never-touched/feedback")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}
