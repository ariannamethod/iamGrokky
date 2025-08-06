"""Example plugin providing a /search command."""
from __future__ import annotations

from typing import Any

from . import BasePlugin

try:  # pragma: no cover - used only with aiogram installed
    from aiogram.types import Message  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    Message = Any  # type: ignore


class WebSearch(BasePlugin):
    """Simple example plugin that echoes search queries."""

    def __init__(self) -> None:
        super().__init__()
        self.commands["search"] = self.handle_search

    async def handle_search(self, message: Message) -> None:
        text = getattr(message, "text", "") or ""
        parts = text.split(maxsplit=1)
        query = parts[1] if len(parts) > 1 else ""
        await message.reply(f"Searching the web for: {query}")
