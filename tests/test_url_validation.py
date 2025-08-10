import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import types

sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))
sys.modules.setdefault("requests", types.ModuleType("requests"))

aiogram_pkg = types.ModuleType("aiogram")
aiogram_pkg.__path__ = []

class Bot:
    def __init__(self, *a, **k):
        pass
    async def send_message(self, *a, **k):
        return None
    async def get_file(self, *a, **k):
        return None

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

aiogram_pkg.Bot = Bot
aiogram_pkg.Dispatcher = Dispatcher

types_mod = types.ModuleType("aiogram.types")
class Message:
    async def reply(self, *a, **k):
        return None
types_mod.Message = Message
class CallbackQuery:
    def __init__(self, data=""):
        self.data = data
types_mod.CallbackQuery = CallbackQuery
aiogram_pkg.types = types_mod
sys.modules["aiogram.types"] = types_mod

enums_mod = types.ModuleType("aiogram.enums")
enums_mod.ChatAction = type("ChatAction", (), {})
aiogram_pkg.enums = enums_mod
sys.modules["aiogram.enums"] = enums_mod

filters_mod = types.ModuleType("aiogram.filters")
class Command:
    def __init__(self, *a, **k):
        pass
filters_mod.Command = Command
aiogram_pkg.filters = filters_mod
sys.modules["aiogram.filters"] = filters_mod

exceptions_mod = types.ModuleType("aiogram.exceptions")
class TelegramAPIError(Exception):
    pass
exceptions_mod.TelegramAPIError = TelegramAPIError
aiogram_pkg.exceptions = exceptions_mod
sys.modules["aiogram.exceptions"] = exceptions_mod

sys.modules["aiogram"] = aiogram_pkg

utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = []

def _stub_async(*args, **kwargs):
    async def _inner(*a, **k):
        return None
    return _inner

sys.modules["utils"] = utils_pkg

plugins_pkg = types.ModuleType("utils.plugins")
plugins_pkg.__path__ = []
plugins_pkg.load_plugins = lambda: {}
# keep stubs
sys.modules["utils.plugins"] = plugins_pkg

utils_pkg.handle = _stub_async()
sys.modules["utils"] = utils_pkg

def _add_stub(module_name, **attrs):
    mod = types.ModuleType(module_name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[module_name] = mod

_add_stub("utils.dayandnight", day_and_night_task=_stub_async())
_add_stub("utils.mirror", mirror_task=_stub_async())
_add_stub(
    "utils.prompt", get_chaos_response=lambda: "", build_system_prompt=lambda **k: ""
)
_add_stub("utils.repo_monitor", monitor_repository=_stub_async())
_add_stub("utils.plugins.imagine", imagine=lambda prompt: "")
_add_stub("utils.vision", analyze_image=_stub_async())
_add_stub("utils.plugins.coder", interpret_code=_stub_async())

eng_stub = type("Engine", (), {})
_add_stub("utils.vector_engine", VectorGrokkyEngine=eng_stub)
_add_stub("utils.hybrid_engine", HybridGrokkyEngine=eng_stub)
_add_stub("utils.context_neural_processor", parse_and_store_file=_stub_async())

import server  # noqa: E402
sys.modules.pop("utils.plugins", None)
sys.modules.pop("utils", None)
sys.modules.pop("aiogram", None)
sys.modules.pop("aiogram.types", None)
sys.modules.pop("aiogram.enums", None)
sys.modules.pop("aiogram.filters", None)
sys.modules.pop("aiogram.exceptions", None)
import importlib  # noqa: E402
importlib.invalidate_caches()


class DummyMessage:
    def __init__(self):
        self.chat = type("Chat", (), {"id": 1, "type": "private"})()
        self.from_user = type("User", (), {"id": 42})()
        self.reply_to_message = None


class DummyEngine:
    async def add_memory(self, *args, **kwargs):
        pass


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://127.0.0.1",
        "http://10.0.0.1",
    ],
)
async def test_handle_text_rejects_private_urls(monkeypatch, url):
    outputs = []
    called = []

    async def fake_reply_split(message, text):
        outputs.append(text)

    async def fake_summarize_link(link):
        called.append(link)
        return "summary"

    server.engine = DummyEngine()
    monkeypatch.setattr(server, "reply_split", fake_reply_split)
    monkeypatch.setattr(server, "summarize_link", fake_summarize_link)

    msg = DummyMessage()
    await server.handle_text(msg, url)

    assert outputs == ["🚫 Некорректная ссылка."]
    assert called == []
