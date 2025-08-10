import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "SECRET")
    monkeypatch.setenv("MAX_WEBHOOK_BODY_SIZE", "50")

    aiogram_pkg = types.ModuleType("aiogram")
    aiogram_pkg.__path__ = []
    aiogram_pkg.Bot = type(
        "Bot",
        (),
        {
            "__init__": lambda self, *a, **k: None,
            "send_message": lambda *a, **k: None,
            "get_file": lambda *a, **k: None,
        },
    )

    class Dispatcher:
        def __init__(self, *a, **k):
            pass

        def message(self, *a, **k):
            def decorator(func):
                return func

            return decorator

        def callback_query(self, *a, **k):
            def decorator(func):
                return func

            return decorator

        async def feed_update(self, *a, **k):
            return None

    aiogram_pkg.Dispatcher = Dispatcher
    monkeypatch.setitem(sys.modules, "aiogram", aiogram_pkg)
    monkeypatch.setitem(
        sys.modules,
        "aiogram.types",
        types.SimpleNamespace(
            Message=type("Message", (), {"reply": lambda self, *a, **k: None}),
            CallbackQuery=type("CallbackQuery", (), {"data": ""}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.enums",
        types.SimpleNamespace(ChatAction=type("ChatAction", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.filters",
        types.SimpleNamespace(Command=type("Command", (), {"__init__": lambda self, *a, **k: None})),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.exceptions",
        types.SimpleNamespace(TelegramAPIError=Exception),
    )
    plugins_pkg = types.ModuleType("utils.plugins")
    plugins_pkg.__path__ = []
    plugins_pkg.load_plugins = lambda: []
    monkeypatch.setitem(sys.modules, "utils.plugins", plugins_pkg)
    monkeypatch.setitem(
        sys.modules,
        "utils.plugins.coder",
        types.SimpleNamespace(interpret_code=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.plugins.42",
        types.SimpleNamespace(handle=lambda *a, **k: None),
    )
    async def _parse_and_store_file(*a, **k):
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "utils.context_neural_processor",
        types.SimpleNamespace(parse_and_store_file=_parse_and_store_file),
    )

    import server
    importlib.reload(server)

    async def dummy_feed_update(bot, update):
        return None

    monkeypatch.setattr(server.dp, "feed_update", dummy_feed_update)
    class DummyUpdate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(server.types, "Update", DummyUpdate, raising=False)
    return server.app


def test_webhook_rate_limit_ip(app):
    client = TestClient(app)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "SECRET"}
    for i in range(30):
        uid = i % 2
        resp = client.post(
            "/webhook",
            headers=headers,
            json={"update_id": i, "message": {"from": {"id": uid}}},
        )
        assert resp.status_code == 200

    resp = client.post(
        "/webhook",
        headers=headers,
        json={"update_id": 31, "message": {"from": {"id": 0}}},
    )
    assert resp.status_code == 429


def test_webhook_rate_limit_user(app):
    client = TestClient(app)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "SECRET"}
    for i in range(20):
        resp = client.post(
            "/webhook",
            headers=headers,
            json={"update_id": i, "message": {"from": {"id": 99}}},
        )
        assert resp.status_code == 200

    resp = client.post(
        "/webhook",
        headers=headers,
        json={"update_id": 21, "message": {"from": {"id": 99}}},
    )
    assert resp.status_code == 429


def test_file_rate_limit(app):
    client = TestClient(app)
    for _ in range(10):
        resp = client.post("/file", files={"file": ("a.txt", b"data")})
        assert resp.status_code == 200

    resp = client.post("/file", files={"file": ("a.txt", b"data")})
    assert resp.status_code == 429

