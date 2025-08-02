from __future__ import annotations

from datetime import datetime
from typing import List


class ThoughtComplexityLogger:
    """Логгер для записи мыслекомплексити и энтропии на каждом ходе."""

    def __init__(self) -> None:
        self.logs: List[dict] = []  # timestamp, message, scale, entropy

    def log_turn(self, message: str, complexity_scale: int, entropy: float) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": message,
            "complexity_scale": complexity_scale,
            "entropy": float(entropy),
        }
        self.logs.append(record)
        print(
            f"LOG@{record['timestamp']} | Complexity: {complexity_scale} | Entropy: {entropy:.3f}"
        )

    def recent(self, n: int = 7) -> List[dict]:
        return self.logs[-n:]


def estimate_complexity_and_entropy(msg: str) -> tuple[int, float]:
    """Эвристическая оценка сложности сообщения и его энтропии."""
    complexity = 1
    if any(word in msg.lower() for word in ["why", "paradox", "recursive", "self", "meta"]):
        complexity += 1
    if len(msg) > 300:
        complexity += 1
    entropy = min(1.0, float(len(set(msg.split()))) / 40.0)
    return min(3, complexity), entropy
