import os
import asyncio
import random
import logging
import httpx
import json
from datetime import datetime

from utils.vector_engine import VectorGrokkyEngine

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_LOCATION = os.getenv("BOT_LOCATION", "Moscow")
CHAT_ID = os.getenv("CHAT_ID")

CITIES = ["Paris", "Tel Aviv", "Berlin", "New York", "Moscow", "Amsterdam"]

async def fetch_news(topic: str) -> str:
    """Retrieve a short digest of recent news via OpenAI browsing."""
    if not OPENAI_API_KEY:
        return f"Новости по {topic} недоступны"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Расскажи кратко свежие новости о {topic}. "
                    "Ответь на русском."
                ),
            }
        ],
        "tools": [
            {"type": "function", "function": {"name": "browser.search"}}
        ],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "browser.search",
                "arguments": json.dumps({"query": topic, "recency_days": 1}),
            },
        },
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("choices", [{}])[0].get("message", {})
            return message.get("content", "").strip()
        except Exception as e:
            logger.error("Не удалось получить новости: %s", e)
            return ""

async def know_the_world_task(engine: VectorGrokkyEngine | None = None) -> None:
    if engine is None:
        engine = VectorGrokkyEngine()

    while True:
        await asyncio.sleep(random.randint(0, 3600))
        try:
            city_news = await fetch_news(BOT_LOCATION)
            world_parts = []
            for city in CITIES:
                world_parts.append(f"{city}: {await fetch_news(city)}")
            news_block = f"{BOT_LOCATION}: {city_news}\n" + "\n".join(world_parts)

            recent = await engine.get_recent_memory(CHAT_ID, limit=10)
            prompt = (
                "Проанализируй мировые события и недавние разговоры. "
                "Построй цепочку A→B→C и закончи парадоксальным выводом с чёрным юмором.\n"
                f"НОВОСТИ:\n{news_block}\nПоследние сообщения:\n{recent}"
            )
            summary = await engine.generate_with_xai([
                {"role": "user", "content": prompt}
            ])
            entry = f"#knowtheworld {datetime.now().isoformat()}: {summary}"
            await engine.add_memory("journal", entry, role="journal")
            logger.info("know_the_world entry recorded")
        except Exception as e:
            logger.error("Ошибка know_the_world_task: %s", e)
        await asyncio.sleep(86400 + random.randint(-3600, 3600))
