import pytest

from utils.plugins.coder import GrokkyCoder


class DummyClient:
    class responses:
        @staticmethod
        def create(*args, **kwargs):
            return type("R", (), {"output_text": "ok"})()


@pytest.mark.anyio("asyncio")
async def test_history_trim(monkeypatch):
    monkeypatch.setattr("utils.plugins.coder.client", DummyClient())
    coder = GrokkyCoder()
    total_calls = GrokkyCoder.MAX_HISTORY // 2 + 5
    for i in range(total_calls):
        await coder.chat(f"prompt {i}")
    assert len(coder.history) == GrokkyCoder.MAX_HISTORY
    expected_first = total_calls - GrokkyCoder.MAX_HISTORY // 2
    assert coder.history[0] == f"prompt {expected_first}"
