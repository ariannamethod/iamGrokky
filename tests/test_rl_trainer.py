import json

from utils.dynamic_weights import DynamicWeights
from utils.rl_trainer import RLTrainer


def test_rl_trainer_updates_weights(tmp_path):
    log_dir = tmp_path / "feedback"
    log_dir.mkdir()
    path = log_dir / "log.jsonl"
    feedback = [
        {"prompt": "hi", "choice": 1, "reward": 1.0},
        {"prompt": "hi", "choice": 1, "reward": 1.0},
    ]
    with path.open("w", encoding="utf-8") as f:
        for entry in feedback:
            f.write(json.dumps(entry) + "\n")

    dw = DynamicWeights([0.5, 0.5])
    trainer = RLTrainer(dw, log_dir=str(log_dir), lr=0.1)
    trainer.train()
    assert dw.base[1] > dw.base[0]
