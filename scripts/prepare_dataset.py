import argparse
import json
import random
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_dataset(input_path: Path, output_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    random.Random(seed).shuffle(records)
    for rec in records:
        rec["content"] = clean_text(rec["content"])
    split = int(len(records) * (1 - val_ratio))
    train, val = records[:split], records[split:]
    output_dir.mkdir(parents=True, exist_ok=True)
    def write(split_name: str, items: list[dict[str, str]]):
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write("train", train)
    write("val", val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Clean and split dialogue dataset")
    parser.add_argument("input", help="Path to raw jsonl file")
    parser.add_argument("output", help="Directory to store train/val files")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    prepare_dataset(Path(args.input), Path(args.output), args.val_ratio)
