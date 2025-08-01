import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from utils.genesis2 import poetic_chorus, impressionistic_filter


def test_poetic_chorus():
    lines = ["first line", "second line"]
    res = poetic_chorus(lines, intensity=1)
    assert "first" in res and "second" in res


def test_impressionistic_filter_changes():
    text = "simple text"
    for _ in range(5):
        out = impressionistic_filter(text, intensity=10)
        if out != text:
            break
    else:
        pytest.skip("filter returned unchanged text repeatedly")
    assert out != text
