import sys
from pathlib import Path
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import server


class DummyMessage:
    def __init__(self):
        self.replies: list[str] = []
    async def reply(self, text):
        self.replies.append(text)


class DummyDocument:
    def __init__(self, file_id):
        self.file_id = file_id


class DummyClient:
    def __init__(self, content: bytes):
        self._content = content
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def get(self, url, timeout):
        return types.SimpleNamespace(content=self._content)


@pytest.mark.anyio("asyncio")
async def test_process_document_allows_txt(monkeypatch):
    outputs = []

    async def fake_parse(path):
        return "PARSED"

    async def fake_reply_split(message, text):
        outputs.append(text)

    async def fake_get_file(file_id):
        return types.SimpleNamespace(file_path="doc/test.txt")

    monkeypatch.setattr(server.bot, "get_file", fake_get_file)
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda *a, **k: DummyClient(b"hi"))
    monkeypatch.setattr(server, "parse_and_store_file", fake_parse)
    monkeypatch.setattr(server, "reply_split", fake_reply_split)

    msg = DummyMessage()
    doc = DummyDocument("1")
    await server._process_document(msg, doc)

    assert outputs == ["PARSED"]
    assert msg.replies == []


@pytest.mark.anyio("asyncio")
async def test_process_document_rejects_exe(monkeypatch):
    outputs = []
    called = []

    async def fake_parse(path):
        called.append(path)
        return "PARSED"

    async def fake_reply_split(message, text):
        outputs.append(text)

    async def fake_get_file(file_id):
        return types.SimpleNamespace(file_path="doc/test.exe")

    monkeypatch.setattr(server.bot, "get_file", fake_get_file)
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda *a, **k: DummyClient(b"hi"))
    monkeypatch.setattr(server, "parse_and_store_file", fake_parse)
    monkeypatch.setattr(server, "reply_split", fake_reply_split)

    msg = DummyMessage()
    doc = DummyDocument("1")
    await server._process_document(msg, doc)

    assert outputs == []
    assert called == []
    assert msg.replies and "Unsupported file type" in msg.replies[0]
