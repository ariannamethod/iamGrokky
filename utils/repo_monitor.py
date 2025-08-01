import logging
from datetime import datetime

from utils.SUPPERTIME_BIOORCHESTRA import BioOrchestra
from utils.vector_engine import VectorGrokkyEngine

logger = logging.getLogger(__name__)

SNAPSHOT_FILE = "data/repo_snapshot.json"


async def monitor_repository(engine: VectorGrokkyEngine | None = None) -> None:
    """Scan repository for changes and record a short summary to memory."""
    if engine is None:
        engine = VectorGrokkyEngine()

    orchestra = BioOrchestra(repo_root=".", snapshot_file=SNAPSHOT_FILE)
    changes = orchestra.monitor()

    summary_parts = []
    for key in ["added", "modified", "deleted"]:
        if changes.get(key):
            summary_parts.append(f"{key}: {len(changes[key])}")
    summary = "; ".join(summary_parts) if summary_parts else "no changes"

    message = f"Repo scan {datetime.now().isoformat()}: {summary}"
    try:
        await engine.add_memory("repo", message, role="journal")
        logger.info("Repo monitor recorded: %s", summary)
    except Exception as exc:
        logger.error("Failed to record repo snapshot: %s", exc)
