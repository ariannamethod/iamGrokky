import asyncio
import glob
import json
import os
from typing import Sequence

from .dynamic_weights import DynamicWeights


class RLTrainer:
    """Simple policy-gradient trainer for :class:`DynamicWeights`.

    The trainer reads feedback logs from ``feedback_dir`` and nudges the
    controller's ``base`` weights proportionally to the received reward.
    """

    def __init__(
        self,
        controller: DynamicWeights,
        feedback_dir: str = "data/feedback",
        lr: float = 0.1,
        interval: float = 60.0,
    ) -> None:
        self.controller = controller
        self.feedback_dir = feedback_dir
        self.lr = lr
        self.interval = interval

    def _update_base(self, reward: float, weights: Sequence[float]) -> None:
        if len(weights) != len(self.controller.base):
            return
        for i, w in enumerate(weights):
            self.controller.base[i] += self.lr * reward * w
        total = sum(self.controller.base)
        if total > 0:
            for i in range(len(self.controller.base)):
                self.controller.base[i] /= total

    def process(self) -> None:
        """Process all feedback files once."""
        paths = glob.glob(os.path.join(self.feedback_dir, "*.jsonl"))
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        reward = data.get("reward")
                        weights = data.get("weights")
                        if reward is None or weights is None:
                            continue
                        self._update_base(float(reward), weights)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

    async def run(self) -> None:
        """Continuously train using new feedback."""
        while True:
            self.process()
            await asyncio.sleep(self.interval)
