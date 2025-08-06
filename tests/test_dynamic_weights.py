import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncio
import time
import pytest

from utils.dynamic_weights import DynamicWeights, aget_dynamic_knowledge


def test_weights_shift_with_pulse(monkeypatch):
    dw = DynamicWeights([1.0, 1.0, 1.0])
    monkeypatch.setattr(
        DynamicWeights, "pulse_from_prompt", lambda self, prompt, api_key=None: 0.5
    )
    weights = dw.weights_for_prompt("hi")
    assert weights[1] == max(weights)

    monkeypatch.setattr(
        DynamicWeights, "pulse_from_prompt", lambda self, prompt, api_key=None: 1.0
    )
    weights = dw.weights_for_prompt("hi")
    assert weights[2] == max(weights)


@pytest.mark.asyncio
async def test_get_dynamic_knowledge_parallel(monkeypatch):
    async def fake_grok(prompt, api_key=None):
        await asyncio.sleep(0.1)
        return "Grok-3 offline"

    async def fake_gpt(prompt, api_key=None):
        await asyncio.sleep(0.1)
        return "ok"

    import utils.dynamic_weights as dw

    monkeypatch.setattr(dw, "query_grok3", fake_grok)
    monkeypatch.setattr(dw, "query_gpt4", fake_gpt)

    start = time.perf_counter()
    result = await aget_dynamic_knowledge("hi")
    duration = time.perf_counter() - start

    assert result == "ok"
    assert duration < 0.15
