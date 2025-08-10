import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_module(monkeypatch):
    monkeypatch.setenv("API_KEY", "SECRET")
    for mod in [
        "aiogram",
        "aiogram.types",
        "aiogram.enums",
        "aiogram.filters",
        "aiogram.exceptions",
    ]:
        sys.modules.pop(mod, None)
    import server
    importlib.reload(server)
    return server


def _client(server_module):
    return TestClient(server_module.app)


def test_file_upload_ok(server_module, monkeypatch):
    async def fake_parse(path):
        return "PARSED"

    monkeypatch.setattr(server_module, "parse_and_store_file", fake_parse)
    client = _client(server_module)
    resp = client.post(
        "/file",
        headers={"X-API-Key": "SECRET"},
        files={"file": ("test.txt", b"hi", "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"result": "PARSED"}


def test_file_upload_bad_extension(server_module, monkeypatch):
    called = []

    async def fake_parse(path):
        called.append(path)
        return "PARSED"

    monkeypatch.setattr(server_module, "parse_and_store_file", fake_parse)
    client = _client(server_module)
    resp = client.post(
        "/file",
        headers={"X-API-Key": "SECRET"},
        files={"file": ("test.exe", b"hi", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["error"]
    assert called == []


def test_file_upload_too_large(server_module, monkeypatch):
    called = []

    async def fake_parse(path):
        called.append(path)
        return "PARSED"

    monkeypatch.setattr(server_module, "parse_and_store_file", fake_parse)
    monkeypatch.setattr(server_module, "MAX_FILE_SIZE", 5)
    client = _client(server_module)
    resp = client.post(
        "/file",
        headers={"X-API-Key": "SECRET"},
        files={"file": ("test.txt", b"123456", "text/plain")},
    )
    assert resp.status_code == 400
    assert "File too large" in resp.json()["error"]
    assert called == []
