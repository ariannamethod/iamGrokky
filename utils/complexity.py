from datetime import datetime


class ThoughtComplexityLogger:
    """Log message complexity and entropy for each reasoning turn."""

    def __init__(self):
        self.logs = []  # timestamp, message, scale, entropy

    def log_turn(self, message: str, complexity_scale: int, entropy: float):
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

    def recent(self, n: int = 7):
        return self.logs[-n:]


def estimate_complexity_and_entropy(msg: str):
    """Simple heuristic to estimate complexity (1-3) and entropy proxy."""
    complexity = 1
    if any(word in msg.lower() for word in ["why", "paradox", "recursive", "self", "meta"]):
        complexity += 1
    if len(msg) > 300:
        complexity += 1
    entropy = min(1.0, float(len(set(msg.split()))) / 40.0)
    return min(3, complexity), entropy
