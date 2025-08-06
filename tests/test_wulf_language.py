import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from SLNCX import wulf_inference  # noqa: E402
from utils.dynamic_weights import DynamicWeights  # noqa: E402
from server import detect_language  # noqa: E402


def test_generate_russian_response(monkeypatch):
    def fake_generate(self, prompt, api_key=None):
        return "ответ"

    monkeypatch.setattr(DynamicWeights, "generate_response", fake_generate)
    response = wulf_inference.generate("привет")
    assert response == "ответ"


def test_detect_spanish():
    assert detect_language("hola señor") == "es"


def test_detect_german():
    assert detect_language("straße für") == "de"


def test_detect_french():
    assert detect_language("où est l'hôtel") == "fr"
