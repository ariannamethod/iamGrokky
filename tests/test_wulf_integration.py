import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from SLNCX import wulf_integration

import pytest  # noqa: E402


class FakePiece:
    def __init__(self, text):
        self.type = "output_text"
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakePiece(text)]


class FakeResponse:
    def __init__(self, text):
        self.output = [FakeMessage(text)]


@pytest.mark.anyio
async def test_generate_response_handles_response_objects(monkeypatch):
    def fake_get_dynamic(prompt, api_key=None):
        return FakeResponse("ok")

    monkeypatch.setattr(wulf_integration, "get_dynamic_knowledge", fake_get_dynamic)
    result = []
    async for token in wulf_integration.generate_response("hi", "grok3"):
        result.append(token)
    assert "".join(result) == "ok"
@pytest.mark.anyio
async def test_generate_response_uses_wulf_model(monkeypatch):
    async def fake_generate(prompt, ckpt_path="out/ckpt.pt", api_key=None):
        yield "wolf"

    monkeypatch.setattr(wulf_integration, "run_wulf", fake_generate)
    result = []
    async for token in wulf_integration.generate_response("hi", "wulf"):
        result.append(token)
    assert "".join(result).strip() == "wolf"
