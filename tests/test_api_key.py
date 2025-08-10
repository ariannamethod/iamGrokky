import importlib
import sys
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_KEY", "SECRET")
    for mod in ["aiogram", "aiogram.types", "aiogram.enums", "aiogram.filters", "aiogram.exceptions"]:
        sys.modules.pop(mod, None)
    import server
    importlib.reload(server)

    async def dummy_handle(cmd):
        return {"response": "ok", "pulse": 0}

    monkeypatch.setattr(server, "handle", dummy_handle)
    return server.app


def test_api_key_required(app):
    client = TestClient(app)
    response = client.post("/42", json={"cmd": "42"})
    assert response.status_code == 401


def test_api_key_valid(app):
    client = TestClient(app)
    response = client.post(
        "/42", headers={"X-API-Key": "SECRET"}, json={"cmd": "42"}
    )
    assert response.status_code == 200


def test_feedback_requires_api_key(app):
    client = TestClient(app)
    response = client.post("/feedback", json={})
    assert response.status_code == 401


def test_ui_requires_api_key(app):
    client = TestClient(app)
    assert client.get("/ui").status_code == 401
    resp = client.get("/ui", headers={"X-API-Key": "SECRET"})
    assert resp.status_code == 200


def test_command_requires_api_key(app):
    client = TestClient(app)
    assert client.post("/command/help").status_code == 401
    resp = client.post("/command/help", headers={"X-API-Key": "SECRET"})
    assert resp.status_code == 200
