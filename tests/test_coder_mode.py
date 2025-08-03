import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import server


class DummyMessage:
    def __init__(self):
        self.chat = type('Chat', (), {'id': 1, 'type': 'private'})()
        self.from_user = type('User', (), {'id': 42})()


@pytest.mark.asyncio
async def test_handle_coder_prompt_without_engine(monkeypatch):
    outputs = []

    async def fake_interpret_code(text):
        return f"code: {text}"

    async def fake_reply_split(message, text):
        outputs.append(text)

    monkeypatch.setattr(server, 'interpret_code', fake_interpret_code)
    monkeypatch.setattr(server, 'reply_split', fake_reply_split)
    server.engine = None

    msg = DummyMessage()
    await server.handle_coder_prompt(msg, 'print(1)')

    assert outputs == ["code: print(1)"]
