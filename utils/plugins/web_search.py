"""Example plugin providing a /search command."""
from __future__ import annotations

from .base import BasePlugin


class WebSearch(BasePlugin):
    """Simple example plugin that echoes search queries."""

    name = "search"
    description = "search the web"

    async def run(self, args: str) -> str:
        query = args.strip()
        return f"Searching the web for: {query}"
