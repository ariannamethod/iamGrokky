import sys
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.pop("utils", None)
importlib.invalidate_caches()
sys.modules.pop("utils.plugins.imagine", None)

from utils.plugins.imagine import enhance_prompt  # noqa: E402


def test_enhance_prompt_short():
    p = enhance_prompt("a cat")
    assert p != "a cat"


def test_enhance_prompt_long():
    long_text = "this is a very long prompt that should not be modified by the enhancer" \
        * 2
    assert enhance_prompt(long_text) == long_text
