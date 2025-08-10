import importlib
import sys
from pathlib import Path

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


def test_unknown_plugin_ignored() -> None:
    plugin_dir = Path(__file__).resolve().parents[1] / "utils" / "plugins"
    fake = plugin_dir / "malicious.py"
    fake.write_text(
        "from utils.plugins import BasePlugin\n"
        "class Malicious(BasePlugin):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.commands['evil'] = lambda _: 'bad'\n"
    )
    try:
        plugins = _load()
        assert all("evil" not in p.commands for p in plugins)
    finally:
        fake.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_command_routing(monkeypatch) -> None:
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
