import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from SLNCX import wulf_inference  # noqa: E402
from utils.dynamic_weights import DynamicWeights  # noqa: E402


def test_generate_russian_response(monkeypatch):
    def fake_generate(self, prompt, api_key=None):
        return "ответ"

    monkeypatch.setattr(DynamicWeights, "generate_response", fake_generate)
    response = wulf_inference.generate("привет")
    assert response == "ответ"
