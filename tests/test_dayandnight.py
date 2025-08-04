import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.dayandnight import day_and_night_task

class DummyEngine:
    def __init__(self):
        self.mem = []

    async def generate_with_xai(self, messages, context=""):
        return "reflection"

    async def add_memory(self, user_id, content, role="user"):
        self.mem.append((user_id, content, role))

@pytest.mark.anyio("asyncio")
async def test_day_and_night_once():
    engine = DummyEngine()
    await day_and_night_task(engine, interval=0, iterations=1)
    assert engine.mem
    user_id, content, role = engine.mem[0]
    assert user_id == "journal"
    assert role == "journal"
    assert "reflection" in content
