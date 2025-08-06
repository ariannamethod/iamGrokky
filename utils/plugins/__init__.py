"""Plugin discovery utilities."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Dict

from .base import BasePlugin


def load_plugins() -> Dict[str, BasePlugin]:
    """Discover and instantiate available plugins."""
    plugins: Dict[str, BasePlugin] = {}
    package = __name__
    base_path = Path(__file__).resolve().parent
    for path in base_path.glob("[!_]*.py"):
        name = path.stem
        if name == "base":
            continue
        module = import_module(f"{package}.{name}")
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BasePlugin)
                and obj is not BasePlugin
            ):
                instance = obj()
                plugins[instance.name] = instance
    return plugins

__all__ = ["BasePlugin", "load_plugins"]
