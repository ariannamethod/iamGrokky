import importlib
import sys
import pytest


def _load() -> dict:
    sys.modules.pop("utils", None)
    for name in list(sys.modules):
        if name.startswith("utils.plugins"):
            sys.modules.pop(name)
    importlib.invalidate_caches()
    plugins_pkg = importlib.import_module("utils.plugins")
    return plugins_pkg.load_plugins()


def test_plugin_discovery() -> None:
    plugins = _load()
    assert "search" in plugins


@pytest.mark.asyncio
async def test_plugin_run() -> None:
    plugins = _load()
    result = await plugins["search"].run("grokky")
    assert result == "Searching the web for: grokky"
