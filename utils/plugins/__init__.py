"""Plugin system base and discovery."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Awaitable, Callable, Dict, List

CommandHandler = Callable[[str], Awaitable[str]]


class BasePlugin:
    """Base class for plugins providing command handlers."""

    commands: Dict[str, CommandHandler]

    def __init__(self) -> None:
        self.commands = {}


def load_plugins() -> List[BasePlugin]:
    """Discover plugin classes and instantiate them."""
    plugins: List[BasePlugin] = []
    package = __name__
    base_path = Path(__file__).resolve().parent
    for path in base_path.glob("[!_]*.py"):
        if path.name == "__init__.py":
            continue
        module = import_module(f"{package}.{path.stem}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
            ):
                instance = obj()
                if instance.commands:
                    plugins.append(instance)
    return plugins


__all__ = ["BasePlugin", "load_plugins", "CommandHandler"]
