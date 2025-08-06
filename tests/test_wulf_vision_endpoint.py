import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_app(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "SECRET")
    monkeypatch.setenv("API_KEY", "KEY")
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    sys.modules["utils"] = utils_pkg

    def add_utils_submodule(name, attrs):
        mod = types.ModuleType(f"utils.{name}")
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[f"utils.{name}"] = mod
        setattr(utils_pkg, name, mod)

    add_utils_submodule("dayandnight", {"day_and_night_task": lambda *a, **k: None})
    add_utils_submodule("mirror", {"mirror_task": lambda *a, **k: None})
    add_utils_submodule("prompt", {"get_chaos_response": lambda *a, **k: None})
    add_utils_submodule("repo_monitor", {"monitor_repository": lambda *a, **k: None})
    add_utils_submodule("imagine", {"imagine": lambda *a, **k: None})
    add_utils_submodule("vision", {"analyze_image": lambda *a, **k: ""})
    add_utils_submodule("coder", {"interpret_code": lambda *a, **k: None})
    add_utils_submodule(
        "context_neural_processor", {"parse_and_store_file": lambda *a, **k: None}
    )
    add_utils_submodule("vector_engine", {"VectorGrokkyEngine": type("VectorGrokkyEngine", (), {})})
    add_utils_submodule("hybrid_engine", {"HybridGrokkyEngine": type("HybridGrokkyEngine", (), {})})
    add_utils_submodule("audio", {"transcribe_audio": lambda *a, **k: ""})
    sub_module_42 = types.ModuleType("utils.42")
    sub_module_42.handle = lambda *a, **k: None
    sys.modules["utils.42"] = sub_module_42
    utils_pkg.handle = sub_module_42.handle

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
    sys.modules["aiogram"] = aiogram_pkg
    sys.modules["aiogram.types"] = types.SimpleNamespace(
        Message=type("Message", (), {"reply": lambda self, *a, **k: None}),
        CallbackQuery=type("CallbackQuery", (), {"data": ""}),
    )
    sys.modules["aiogram.enums"] = types.SimpleNamespace(
        ChatAction=type("ChatAction", (), {})
    )
    sys.modules["aiogram.filters"] = types.SimpleNamespace(
        Command=type("Command", (), {"__init__": lambda self, *a, **k: None})
    )
    sys.modules["aiogram.exceptions"] = types.SimpleNamespace(
        TelegramAPIError=Exception
    )

    sl_pkg = types.ModuleType("SLNCX")
    sl_pkg.__path__ = []
    sys.modules["SLNCX"] = sl_pkg
    wi_module = types.ModuleType("SLNCX.wulf_integration")
    wi_module.generate_response = lambda *a, **k: None
    sys.modules["SLNCX.wulf_integration"] = wi_module
    setattr(sl_pkg, "wulf_integration", wi_module)

    import server
    importlib.reload(server)
    return server


def test_wulf_vision_url(server_app, monkeypatch):
    server = server_app
    client = TestClient(server.app)

    def fake_analyze(url):
        return "desc"

    monkeypatch.setattr(server, "analyze_image", fake_analyze)
    resp = client.post("/wulf/vision?url=http://img", headers={"X-API-Key": "KEY"})
    assert resp.status_code == 200
    assert resp.json() == {"description": "desc"}


def test_wulf_vision_file_upload(server_app, monkeypatch):
    server = server_app
    client = TestClient(server.app)

    captured = {}

    def fake_analyze(url):
        captured["url"] = url
        return "desc"

    monkeypatch.setattr(server, "analyze_image", fake_analyze)
    resp = client.post(
        "/wulf/vision",
        headers={"X-API-Key": "KEY"},
        files={"file": ("a.png", b"123", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"description": "desc"}
    assert captured["url"].startswith("data:image/png;base64,")
