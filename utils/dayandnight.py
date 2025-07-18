import asyncio
from datetime import datetime
import logging

from utils.vector_engine import VectorGrokkyEngine

logger = logging.getLogger(__name__)


async def day_and_night_task(engine: VectorGrokkyEngine | None = None) -> None:
    """Records daily reflections into vector memory."""
    if engine is None:
        engine = VectorGrokkyEngine()

    while True:
        await asyncio.sleep(86400)
        try:
            reflection = await engine.generate_with_xai([
                {
                    "role": "user",
                    "content": (
                        "Подведи краткие впечатления от прошедшего дня, "
                        "упомяни главу, которую ты изучал, "
                        "и свои мысли о диалоге."
                    ),
                }
            ])
        except Exception as exc:
            logger.error("Не удалось получить дневное резюме: %s", exc)
            reflection = f"Сбой генерации: {exc}"

        entry = f"{datetime.now().date()}: {reflection}"
        await engine.add_memory("journal", entry, role="journal")
        logger.info("Daily reflection recorded")


if __name__ == "__main__":
    asyncio.run(day_and_night_task())
