import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from SLNCX import wulf_inference
from utils.dynamic_weights import DynamicWeights

import pytest


@pytest.mark.anyio
async def test_generate_russian_response(monkeypatch):
    def fake_weights(self, prompt, api_key=None):
        return [1.0] + [0.0] * 4

    monkeypatch.setattr(DynamicWeights, "weights_for_prompt", fake_weights)
    tokens = []
    async for t in wulf_inference.generate("привет"):
        tokens.append(t)
    response = "".join(tokens)
    assert "Задача" in response

