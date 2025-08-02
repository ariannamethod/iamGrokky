import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.complexity import (
    ThoughtComplexityLogger,
    estimate_complexity_and_entropy,
)


def test_estimate_basic():
    msg = "simple question"
    complexity, entropy = estimate_complexity_and_entropy(msg)
    assert complexity == 1
    assert 0 <= entropy <= 1


def test_estimate_with_keywords():
    msg = "Why recursion often breeds paradox?"
    complexity, entropy = estimate_complexity_and_entropy(msg)
    assert complexity >= 2
    assert entropy > 0


def test_logger_records():
    logger = ThoughtComplexityLogger()
    logger.log_turn("hi", 1, 0.5)
    assert len(logger.logs) == 1
    assert logger.logs[0]["complexity_scale"] == 1
