import importlib
import sys
import pytest


@pytest.mark.asyncio
async def test_help_lists_plugins(monkeypatch):
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
    msg = type("Msg", (), {"text": "/help"})()
    for handler in server.dp.message.handlers:
        for f in handler.filters:
            commands = getattr(f.callback, "commands", [])
            if "help" in commands:
                await handler.callback(msg)
                text = outputs[0]
                for plugin in server.PLUGINS:
                    for cmd in plugin.commands:
                        assert f"/{cmd}" in text
                return
    pytest.fail("help handler not registered")
