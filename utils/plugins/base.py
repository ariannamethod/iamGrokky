from __future__ import annotations

from abc import ABC, abstractmethod


class BasePlugin(ABC):
    """Minimal interface for Grokky plugins."""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def run(self, args: str) -> str:
        """Execute the plugin and return a textual result."""
        raise NotImplementedError
