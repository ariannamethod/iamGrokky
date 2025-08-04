import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def pytest_runtest_makereport(item, call):
    if call.when != "call":
        return
    status = "pass" if call.excinfo is None else "fail"
    log_dir = Path("logs/tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.utcnow().date()}.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "test": item.name,
        "status": status,
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
