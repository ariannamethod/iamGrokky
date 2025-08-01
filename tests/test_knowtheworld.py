import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.knowtheworld as ktw

class DummyEngine:
    def __init__(self):
        self.mem = []

    async def add_memory(self, user_id, content, role="user"):
        self.mem.append((user_id, content, role))

    async def generate_with_xai(self, messages, context=""):
        return "summary"

    async def get_recent_memory(self, user_id, limit=10):
        return "recent"

@pytest.mark.asyncio
async def test_know_the_world_once(monkeypatch):
    async def fake_fetch(topic):
        return f"news {topic}"
    monkeypatch.setattr(ktw, "fetch_news", fake_fetch)

    engine = DummyEngine()
    await ktw.know_the_world_task(engine, interval=0, iterations=1)

    assert engine.mem
    assert engine.mem[0][0] == "journal"
    assert "#knowtheworld" in engine.mem[0][1]
