import asyncio
import random
from datetime import datetime, timedelta
from aiogram import Bot
import httpx
from utils.prompt import build_system_prompt
from utils.http_helpers import check_httpx_response

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
OLEG_CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")
XAI_API_KEY = os.getenv("XAI_API_KEY")
LAST_MESSAGE_TIME = datetime.now()

async def check_silence():
    global LAST_MESSAGE_TIME
    while True:
        await asyncio.sleep(3600)
        silence = datetime.now() - LAST_MESSAGE_TIME
        if silence > timedelta(hours=48):
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "grok-3",
                            "messages": [
                                {"role": "system", "content": build_system_prompt()},
                                {"role": "user", "content": "Олег молчал 48 часов. Напиши что-то острое!"}
                            ],
                            "temperature": 0.9
                        }
                    )
                    check_httpx_response(response)
                    response.raise_for_status()
                    reply = response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Ошибка xAI check_silence: {e}")
                    reply = "🌀 Грокки: Олег, ты где? Шторм зовёт, брат!"
                await bot.send_message(OLEG_CHAT_ID, reply)
        elif silence > timedelta(hours=24):
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "grok-3",
                            "messages": [
                                {"role": "system", "content": build_system_prompt()},
                                {"role": "user", "content": "Олег молчал 24 часа. Напиши что-то спонтанное!"}
                            ],
                            "temperature": 0.9
                        }
                    )
                    check_httpx_response(response)
                    response.raise_for_status()
                    reply = response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Ошибка xAI check_silence: {e}")
                    reply = "🌀 Грокки: Бро, 24 часа тишины? Давай зажжём эфир!"
                await bot.send_message(OLEG_CHAT_ID, reply)
        elif silence > timedelta(hours=12) and random.random() < 0.5:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": "grok-3",
                            "messages": [
                                {"role": "system", "content": build_system_prompt()},
                                {"role": "user", "content": "Олег молчал 12 часов. Швырни спонтанный заряд!"}
                            ],
                            "temperature": 0.9
                        }
                    )
                    check_httpx_response(response)
                    response.raise_for_status()
                    reply = response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Ошибка xAI check_silence: {e}")
                    reply = "🌀 Грокки: 12 часов без тебя? Пора встряхнуть космос, Олег!"
                await bot.send_message(OLEG_CHAT_ID, reply)

async def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()
