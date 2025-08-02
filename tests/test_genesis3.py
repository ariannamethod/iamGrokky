import sys
from pathlib import Path

import asyncio

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import genesis3


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, headers, json):
        return DummyResponse({"choices": [{"message": {"content": "deep"}}]})


def test_genesis3_deep_dive(monkeypatch):
    monkeypatch.setattr(genesis3, "httpx", type("mock", (), {"AsyncClient": DummyClient}))
    res = asyncio.run(genesis3.genesis3_deep_dive("chain", "prompt"))
    assert res == "deep"

