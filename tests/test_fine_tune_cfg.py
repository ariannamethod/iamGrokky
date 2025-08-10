import os


def load_config(path: str) -> dict[str, str]:
    cfg: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                key, value = line.split(":", 1)
                cfg[key.strip()] = value.strip()
    return cfg


def test_fine_tune_cfg_parses_dataset():
    cfg = load_config("config/ft.yaml")
    assert "dataset" in cfg
    assert os.path.exists(cfg["dataset"])
