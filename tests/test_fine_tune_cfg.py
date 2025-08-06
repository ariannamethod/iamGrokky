from SLNCX.training.fine_tune import load_config
import os


def test_fine_tune_cfg_parses_dataset():
    cfg = load_config("config/ft.yaml")
    assert "dataset" in cfg
    assert os.path.exists(cfg["dataset"])
