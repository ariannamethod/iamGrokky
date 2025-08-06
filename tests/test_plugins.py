import pytest

from utils.plugins import iter_plugins


class MockMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:  # pragma: no cover - simple storage
        self.replies.append(text)


class MockMessageRouter:
    def __init__(self) -> None:
        self.handlers: dict[str, callable] = {}

    def register(self, handler, command_filter) -> None:  # pragma: no cover - simple storage
        self.handlers[command_filter.commands[0]] = handler


class MockDispatcher:
    def __init__(self) -> None:
        self.message = MockMessageRouter()


def test_plugin_discovery() -> None:
    names = {plugin.__class__.__name__ for plugin in iter_plugins()}
    assert "WebSearch" in names


@pytest.mark.asyncio
async def test_command_routing() -> None:
    dp = MockDispatcher()
    for plugin in iter_plugins():
        plugin.register(dp)
    assert "search" in dp.message.handlers
    handler = dp.message.handlers["search"]
    msg = MockMessage("/search grokky")
    await handler(msg)
    assert msg.replies == ["Searching the web for: grokky"]
