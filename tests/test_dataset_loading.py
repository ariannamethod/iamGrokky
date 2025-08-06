import json
from pathlib import Path

DATA_DIR = Path('SLNCX/datasets/dialogues')


def _check_file(path: Path) -> None:
    assert path.exists(), f"missing {path}"
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            assert 'role' in obj and 'content' in obj
            assert isinstance(obj['role'], str)
            assert isinstance(obj['content'], str)


def test_train_file_format() -> None:
    _check_file(DATA_DIR / 'train.jsonl')


def test_val_file_format() -> None:
    _check_file(DATA_DIR / 'val.jsonl')
