import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import server

class DummyMsg:
    def __init__(self):
        self.replies = []
        self.chat = type("c", (), {"id": 1})

    async def reply(self, text):
        self.replies.append(text)

class DummyBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))

@pytest.mark.asyncio
async def test_reply_split_two_parts(monkeypatch):
    msg = DummyMsg()
    bot = DummyBot()
    monkeypatch.setattr(server, "bot", bot)

    long_text = "a" * 5000
    await server.reply_split(msg, long_text)

    assert len(msg.replies) == 1
    assert len(bot.sent) == 1
    assert "Часть 1/2" in msg.replies[0]
    assert "Часть 2/2" in bot.sent[0][1]
