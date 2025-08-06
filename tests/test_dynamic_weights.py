import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.dynamic_weights import DynamicWeights


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
