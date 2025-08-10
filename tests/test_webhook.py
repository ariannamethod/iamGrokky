import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "SECRET")
    monkeypatch.setenv("MAX_WEBHOOK_BODY_SIZE", "50")
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []  # mark as package
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)

    def add_utils_submodule(name, attrs):
        mod = types.ModuleType(f"utils.{name}")
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, f"utils.{name}", mod)
        setattr(utils_pkg, name, mod)

    add_utils_submodule("dayandnight", {"day_and_night_task": lambda *a, **k: None})
    add_utils_submodule("mirror", {"mirror_task": lambda *a, **k: None})
    add_utils_submodule(
        "prompt",
        {"get_chaos_response": lambda *a, **k: None, "build_system_prompt": lambda **k: ""},
    )
    add_utils_submodule("repo_monitor", {"monitor_repository": lambda *a, **k: None})
    add_utils_submodule("imagine", {"imagine": lambda *a, **k: None})
    add_utils_submodule("vision", {"analyze_image": lambda *a, **k: None})
    add_utils_submodule("coder", {"interpret_code": lambda *a, **k: None})
    add_utils_submodule(
        "context_neural_processor",
        {"parse_and_store_file": lambda *a, **k: None},
    )
    add_utils_submodule("vector_engine", {"VectorGrokkyEngine": type("VectorGrokkyEngine", (), {})})
    add_utils_submodule("hybrid_engine", {"HybridGrokkyEngine": type("HybridGrokkyEngine", (), {})})
    add_utils_submodule("plugins", {"load_plugins": lambda: []})

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
        types.SimpleNamespace(
            Command=type("Command", (), {"__init__": lambda self, *a, **k: None})
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiogram.exceptions",
        types.SimpleNamespace(TelegramAPIError=Exception),
    )
    sub_module_42 = types.ModuleType("utils.42")
    sub_module_42.handle = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "utils.42", sub_module_42)
    utils_pkg.handle = sub_module_42.handle

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


def test_webhook_payload_too_large(app):
    client = TestClient(app)
    response = client.post(
        "/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "SECRET"},
        json={"update_id": 1, "text": "x" * 100},
    )
    assert response.status_code == 413
    assert response.text == "payload too large"
