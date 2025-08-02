import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.complexity import estimate_complexity_and_entropy


def test_estimate_complexity_basic():
    msg = "why paradox recursion self meta " * 20
    complexity, entropy = estimate_complexity_and_entropy(msg)
    assert complexity == 3
    assert 0 <= entropy <= 1
