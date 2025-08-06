import pytest
from types import SimpleNamespace

from SLNCX import wulf_inference
import server


@pytest.mark.anyio
async def test_wulf_inference_streams_tokens():
    gen = wulf_inference.generate("hello")
    tokens = []
    async for t in gen:
        tokens.append(t)
    # Expect tokens join to one of predefined responses
    assert "".join(tokens).strip().startswith("⚡ SLNCX")


class DummyBot:
    def __init__(self):
        self.actions = []
        self.edits = []

    async def send_chat_action(self, chat_id, action):
        self.actions.append((chat_id, action))

    async def edit_message_text(self, text, chat_id, message_id):
        self.edits.append(text)

    async def send_message(self, chat_id, text):
        self.sent = text
        return SimpleNamespace(message_id=1)

    async def get_file(self, *a, **k):
        return None


class DummyMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=1)
        self.replied = []

    async def reply(self, text):
        self.replied.append(text)
        return SimpleNamespace(message_id=1)


@pytest.mark.anyio
async def test_server_streams_to_bot(monkeypatch):
    async def fake_generate(prompt, mode):
        for tok in ["foo ", "bar"]:
            yield tok

    monkeypatch.setattr(server, "generate_response", fake_generate)
    dummy_bot = DummyBot()
    monkeypatch.setattr(server, "bot", dummy_bot)
    monkeypatch.setattr(server, "ChatAction", SimpleNamespace(TYPING="typing"))
    msg = DummyMessage()

    await server.stream_wulf_response(msg, "hi")

    assert msg.replied == ["foo "]
    assert dummy_bot.edits == ["foo bar"]
    assert dummy_bot.actions  # typing actions sent
