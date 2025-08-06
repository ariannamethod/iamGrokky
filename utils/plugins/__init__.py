"""Plugin system for Grokky."""
from __future__ import annotations

from importlib import import_module
import pkgutil
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Type

try:  # pragma: no cover - used only with aiogram installed
    from aiogram.filters import Command  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    class Command:  # type: ignore
        """Minimal stand-in for aiogram's Command filter."""

        def __init__(self, command: str):
            self.commands = [command]


Handler = Callable[[Any], Awaitable[None]]


class BasePlugin:
    """Base class for Grokky plugins.

    Subclasses should populate :pyattr:`commands` with a mapping from command
    names to async handler callables. The :py:meth:`register` method wires the
    handlers into an aiogram :class:`Dispatcher`.
    """

    commands: Dict[str, Handler]

    def __init__(self) -> None:
        if not hasattr(self, "commands"):
            self.commands = {}

    def register(self, dispatcher: Any) -> None:
        """Register handlers with the provided dispatcher."""
        message_router = getattr(dispatcher, "message", None)
        if message_router is None:
            return
        register = getattr(message_router, "register", None)
        if register is None:
            return
        for command, handler in self.commands.items():
            register(handler, Command(command))


def iter_plugins() -> Iterable[BasePlugin]:
    """Yield plugin instances discovered in this package."""
    package = __name__
    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = module_info.name
        if name.startswith("_"):
            continue
        module = import_module(f"{package}.{name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BasePlugin) and obj is not BasePlugin:
                yield obj()
