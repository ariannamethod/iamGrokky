import importlib
import sys

from fastapi.testclient import TestClient


def test_metrics_exposed_for_command(monkeypatch):
    for mod in [
        "aiogram",
        "aiogram.types",
        "aiogram.enums",
        "aiogram.filters",
        "aiogram.exceptions",
    ]:
        sys.modules.pop(mod, None)
    import server
    importlib.reload(server)

    async def dummy_handle(cmd):
        return {"response": "ok", "pulse": 0}

    async def dummy_startup():
        pass

    async def dummy_shutdown():
        pass

    monkeypatch.setattr(server, "handle", dummy_handle)
    monkeypatch.setattr(server, "on_startup", dummy_startup)
    monkeypatch.setattr(server, "on_shutdown", dummy_shutdown)

    client = TestClient(server.app)
    resp = client.post("/42", json={"cmd": "42"})
    assert resp.status_code == 200

    metrics = client.get("/metrics").text
    assert 'commands_total{command="42"}' in metrics
    assert 'data_transferred_bytes_total{direction="in"}' in metrics
