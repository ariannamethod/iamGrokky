import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import server


class DummyMessage:
    def __init__(self, text):
        self.text = text
        self.chat = type("Chat", (), {"id": 1, "type": "private"})()
        self.from_user = type("User", (), {"id": 42})()

    async def reply(self, text):
        pass


@pytest.mark.anyio
async def test_handle_text_slncx_error(monkeypatch, anyio_backend):
    if anyio_backend != "asyncio":
        pytest.skip("asyncio only")
    outputs = []

    async def fake_reply_split(message, text):
        outputs.append(text)

    def fake_slncx_generate(prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "slncx_generate", fake_slncx_generate)
    monkeypatch.setattr(server, "reply_split", fake_reply_split)
    server.SLNCX_MODE[1] = True

    msg = DummyMessage("hi")
    await server.handle_text(msg, "hi")

    assert any("SLNCX error" in o and "boom" in o for o in outputs)
    server.SLNCX_MODE.clear()


@pytest.mark.anyio
async def test_cmd_slncx_error(monkeypatch, anyio_backend):
    if anyio_backend != "asyncio":
        pytest.skip("asyncio only")
    outputs = []

    async def fake_reply_split(message, text):
        outputs.append(text)

    def fake_slncx_generate(prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "slncx_generate", fake_slncx_generate)
    monkeypatch.setattr(server, "reply_split", fake_reply_split)

    msg = DummyMessage("/slncx hi")
    await server.cmd_slncx(msg)

    assert outputs and "SLNCX error" in outputs[0]
