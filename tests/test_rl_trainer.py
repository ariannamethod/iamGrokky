import json
from utils.dynamic_weights import DynamicWeights
from utils.rl_trainer import RLTrainer


def test_rl_trainer_updates_base(tmp_path):
    fb_dir = tmp_path / "feedback"
    fb_dir.mkdir()
    data = [
        {"prompt": "p1", "reward": 1.0, "weights": [0.2, 0.8]},
        {"prompt": "p2", "reward": -0.5, "weights": [0.6, 0.4]},
    ]
    log_file = fb_dir / "log.jsonl"
    with log_file.open("w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    dw = DynamicWeights([1.0, 1.0])
    trainer = RLTrainer(dw, feedback_dir=str(fb_dir), lr=0.1)
    trainer.process()

    assert abs(sum(dw.base) - 1.0) < 1e-6
    assert dw.base[1] > dw.base[0]
