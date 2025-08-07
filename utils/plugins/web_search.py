"""Example plugin providing a /search command."""
from __future__ import annotations

from . import BasePlugin


class WebSearch(BasePlugin):
    """Simple example plugin that echoes search queries."""

    def __init__(self) -> None:
        super().__init__()
        self.commands["search"] = self.search

    async def search(self, args: str) -> str:
        query = args.strip()
        return f"Searching the web for: {query}"
