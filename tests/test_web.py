"""Smoke tests for the web UI server.

All tests use FastAPI's TestClient — no real server bind, no API calls.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from lab.web.server import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers_with_env_keys" in body
    assert "task_count" in body


def test_index_html_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Strategy Lab" in r.text
    assert "<title>Strategy Lab</title>" in r.text


def test_static_assets_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "startBacktest" in r.text
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "font-family" in r.text


def test_runs_list_endpoint(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert "runs" in r.json()
    assert isinstance(r.json()["runs"], list)


def test_task_not_found(client):
    r = client.get("/api/tasks/nonexistent")
    assert r.status_code == 404


def test_run_endpoint_rejects_missing_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/api/run", json={"prompt": "test"})
    assert r.status_code == 400
    assert "No API key" in r.json()["detail"]


def test_run_endpoint_accepts_inline_key_validates_async(client, monkeypatch):
    """With an inline key the request is accepted (200) — the actual API
    call happens async in a thread and may fail later, but the synchronous
    validation passes."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/api/run", json={
        "prompt": "test prompt",
        "anthropic_api_key": "sk-ant-fake-for-test",
    })
    assert r.status_code == 200
    body = r.json()
    assert "task_id" in body
    assert body["status"] in ("pending", "running", "error")


def test_reference_endpoint_schema(client):
    """POST /api/reference accepts a ReferenceRequest and returns a task_id.
    Doesn't wait for completion."""
    r = client.post("/api/reference", json={
        "strategy": "60_40",
        "universe": ["SPY", "TLT"],
    })
    assert r.status_code == 200
    body = r.json()
    assert "task_id" in body


def test_reference_endpoint_unknown_strategy(client):
    """Unknown strategy name → task fails with a clear error message."""
    r = client.post("/api/reference", json={
        "strategy": "nope",
        "universe": ["SPY"],
    })
    # The request is accepted (the task itself raises later).
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    # Wait briefly for the background task to finish.
    for _ in range(30):
        time.sleep(0.1)
        s = client.get(f"/api/tasks/{task_id}").json()
        if s["status"] in ("done", "error"):
            break
    assert s["status"] == "error"
    assert "unknown reference" in s["error"]


def test_create_app_returns_fastapi_instance():
    """create_app() is the exported factory."""
    from fastapi import FastAPI
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Strategy Lab"
