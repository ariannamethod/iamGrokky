import argparse
import os
import runpy
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path: str
        Path to the YAML file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def run(cfg: Dict[str, Any]) -> None:
    """Execute the NanoGPT runner with the provided config."""
    dataset = cfg.get("dataset")
    if dataset is None:
        raise ValueError("`dataset` must be specified in the config")
    if not os.path.exists(dataset):
        raise FileNotFoundError(dataset)

    out_dir = cfg.get("out_dir", "out")
    # pass configuration to nanogpt_runner via globals
    init_globals = {"out_dir": out_dir, "dataset": dataset}
    runpy.run_module("SLNCX.nanogpt_runner", init_globals=init_globals, run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine tune NanoGPT on a dataset")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dataset", help="Override dataset path from config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"] = args.dataset

    run(cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
