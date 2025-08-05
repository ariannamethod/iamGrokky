import os
import asyncio
import hashlib
from datetime import datetime
import random
import httpx
from aiogram import Bot
from utils.prompt import build_system_prompt

REPO_URL = "https://grokky.ariannamethod.me/repo/"
RESEARCH_FILE = "grokkyresearch.md"
LAST_HASH_FILE = "data/last_mirror_hash.txt"
XAI_API_KEY = os.getenv("XAI_API_KEY")
AGENT_GROUP = os.getenv("AGENT_GROUP")

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))

async def mirror_task():
    last_hash = ""
    if os.path.exists(LAST_HASH_FILE):
        with open(LAST_HASH_FILE, "r") as f:
            last_hash = f.read().strip()

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(REPO_URL, timeout=10)
                resp.raise_for_status()
                code = resp.text
            current_hash = hashlib.sha256(code.encode()).hexdigest()

            if current_hash != last_hash:
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(
                            "https://api.x.ai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                            json={
                                "model": "grok-3",
                                "messages": [
                                    {"role": "system", "content": build_system_prompt()},
                                    {"role": "user", "content": f"Анализируй этот код и предложи улучшения:\n{code}"}
                                ],
                                "temperature": 1.0  # bump creativity a bit
                            }
                        )
                        response.raise_for_status()
                        reply = response.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"Ошибка xAI mirror_task: {e}")
                        reply = "🌀 Грокки: Не могу проанализировать код, эфир трещит!"
                with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now()}: {reply}\n\n")
                with open(LAST_HASH_FILE, "w") as f:
                    f.write(current_hash)
            if random.random() < 0.2:
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(
                            "https://api.x.ai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                            json={
                                "model": "grok-3",
                                "messages": [
                                    {"role": "system", "content": build_system_prompt()},
                                    {"role": "user", "content": "[CHAOS_PULSE] type=poetry_burst intensity=5"}
                                ],
                                "temperature": 1.0  # bump creativity a bit
                            }
                        )
                        response.raise_for_status()
                        reply = response.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        print(f"Ошибка xAI mirror_task: {e}")
                        reply = "🌀 Грокки: Хаос не врубился, но шторм гремит!"
                if AGENT_GROUP:
                    await bot.send_message(AGENT_GROUP, f"🌀 Грокки: {reply}")
        except Exception as e:
            with open(RESEARCH_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()}: Ошибка: {e}\n\n")
        await asyncio.sleep(302400)
