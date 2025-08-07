"""Reinforcement-learning utilities for dynamic weighting."""

from __future__ import annotations

import json
import os
from typing import Iterator

from .dynamic_weights import DynamicWeights, apply_pulse


def log_feedback(prompt: str, choice: int, reward: float, log_dir: str = "data/feedback") -> None:
    """Append a feedback entry to a JSONL log.

    Parameters
    ----------
    prompt:
        The user prompt that produced the response.
    choice:
        Index of the response that was shown to the user.
    reward:
        Feedback score where positive values indicate approval.
    log_dir:
        Directory to store ``*.jsonl`` feedback files.
    """

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "log.jsonl")
    entry = {"prompt": prompt, "choice": int(choice), "reward": float(reward)}
    with open(path, "a", encoding="utf-8") as f:  # pragma: no cover - I/O
        f.write(json.dumps(entry) + "\n")


class RLTrainer:
    """Policy-gradient trainer for :class:`~utils.dynamic_weights.DynamicWeights`.

    The trainer reads feedback entries written by :func:`log_feedback` and
    nudges the base weights toward choices that received positive reward while
    reducing those with negative reward.  A very small and simple REINFORCE
    algorithm is used; weights are normalised after each training run.
    """

    def __init__(
        self,
        weights: DynamicWeights,
        log_dir: str = "data/feedback",
        lr: float = 0.1,
    ) -> None:
        self.weights = weights
        self.log_dir = log_dir
        self.lr = lr

    def _entries(self) -> Iterator[dict]:
        if not os.path.isdir(self.log_dir):
            return
        for name in os.listdir(self.log_dir):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(self.log_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:  # pragma: no cover - I/O
                    for line in fh:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

    def train(self) -> None:
        """Run a single training iteration over all logged feedback."""

        probs = apply_pulse(self.weights.base, 0.0)
        for fb in self._entries():
            choice = int(fb.get("choice", 0))
            reward = float(fb.get("reward", 0.0))
            for i in range(len(self.weights.base)):
                grad = ((1.0 if i == choice else 0.0) - probs[i]) * reward
                self.weights.base[i] += self.lr * grad
        total = sum(self.weights.base)
        if total > 0:
            self.weights.base = [max(w, 0.0) / total for w in self.weights.base]
