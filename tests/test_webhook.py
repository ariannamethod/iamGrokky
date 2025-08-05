import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "SECRET")
    import server
    importlib.reload(server)

    async def dummy_feed_update(bot, update):
        return None

    monkeypatch.setattr(server.dp, "feed_update", dummy_feed_update)

    class DummyUpdate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(server.types, "Update", DummyUpdate)
    return server.app


def test_webhook_forbidden(app):
    client = TestClient(app)
    response = client.post(
        "/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "WRONG"},
        json={"update_id": 1},
    )
    assert response.status_code == 403


def test_webhook_ok(app):
    client = TestClient(app)
    response = client.post(
        "/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "SECRET"},
        json={"update_id": 1},
    )
    assert response.status_code == 200
    assert response.text == "OK"
