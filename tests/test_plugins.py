import importlib
import sys
import pytest


def _load():
    sys.modules.pop("utils", None)
    for name in list(sys.modules):
        if name.startswith("utils.plugins"):
            sys.modules.pop(name)
    importlib.invalidate_caches()
    plugins_pkg = importlib.import_module("utils.plugins")
    return plugins_pkg.load_plugins()


def test_plugin_discovery() -> None:
    plugins = _load()
    assert any("search" in p.commands for p in plugins)


@pytest.mark.asyncio
async def test_command_routing(monkeypatch) -> None:
    for mod in ["aiogram", "aiogram.types", "aiogram.enums", "aiogram.filters", "aiogram.exceptions"]:
        sys.modules.pop(mod, None)
    import server
    importlib.reload(server)
    outputs = []

    async def fake_reply_split(message, text):
        outputs.append(text)

    monkeypatch.setattr(server, "reply_split", fake_reply_split)
    msg = type("Msg", (), {"text": "/search grokky"})()
    for handler in server.dp.message.handlers:
        for f in handler.filters:
            commands = getattr(f.callback, "commands", [])
            if "search" in commands:
                await handler.callback(msg)
                assert outputs == ["Searching the web for: grokky"]
                return
    pytest.fail("search handler not registered")
