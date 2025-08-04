import asyncio
import os
from datetime import datetime
import logging

from utils.vector_engine import VectorGrokkyEngine

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Подведи краткие впечатления от прошедшего дня, "
    "упомяни главу, которую ты изучал, и свои мысли о диалоге."
)


async def day_and_night_task(
    engine: VectorGrokkyEngine | None = None,
    *,
    interval: int | None = None,
    iterations: int | None = None,
    prompt: str | None = None,
) -> None:
    """Record daily reflections into vector memory.

    Parameters
    ----------
    engine: VectorGrokkyEngine | None
        Optional engine instance. If ``None`` a new one is created.
    interval: int | None
        Interval in seconds between iterations. Defaults to 86400 or
        ``DAYANDNIGHT_INTERVAL`` environment variable.
    iterations: int | None
        Number of iterations to run. ``None`` means run forever.
    prompt: str | None
        Override the reflection prompt.
    """
    if engine is None:
        engine = VectorGrokkyEngine()

    interval = interval or int(os.getenv("DAYANDNIGHT_INTERVAL", "86400"))
    prompt = prompt or DEFAULT_PROMPT

    count = 0
    while True:
        try:
            reflection = await engine.generate_with_xai([
                {"role": "user", "content": prompt}
            ])
        except Exception as exc:
            logger.exception("Не удалось получить дневное резюме")
            reflection = f"Сбой генерации: {exc}"

        entry = f"{datetime.now().date()}: {reflection}"
        await engine.add_memory("journal", entry, role="journal")
        logger.info("Daily reflection recorded")

        count += 1
        if iterations is not None and count >= iterations:
            break

        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(day_and_night_task())
