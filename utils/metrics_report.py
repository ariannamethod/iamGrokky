"""Cron-friendly metrics report writer."""

from datetime import datetime
from pathlib import Path

from prometheus_client import generate_latest


OUTPUT_DIR = Path("logs/metrics")


def write_report() -> Path:
    """Write current metrics snapshot to a timestamped file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = generate_latest()
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    target = OUTPUT_DIR / f"{ts}.prom"
    target.write_bytes(content)
    return target


if __name__ == "__main__":  # pragma: no cover - manual script
    write_report()
