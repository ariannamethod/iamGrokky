import os
import asyncio
import random
from datetime import datetime, timedelta
from aiogram import Bot
from utils.hybrid_engine import HybridGrokkyEngine

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
OLEG_CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")
engine = HybridGrokkyEngine()
_OPENAI_READY = False
LAST_MESSAGE_TIME = datetime.now()


async def check_silence():
    """Проверяет период молчания и инициирует сообщение."""
    while True:
        await asyncio.sleep(3600)
        silence = datetime.now() - LAST_MESSAGE_TIME
        print(f"check_silence: {silence.total_seconds() / 3600:.1f}h since last message")
        if silence > timedelta(hours=48):
            await send_prompt("Олег молчал 48 часов. Напиши что-то острое!")
        elif silence > timedelta(hours=24):
            await send_prompt("Олег молчал 24 часа. Напиши что-то спонтанное!")
        elif silence > timedelta(hours=12) and random.random() < 0.5:
            await send_prompt("Олег молчал 12 часов. Швырни спонтанный заряд!")


async def send_prompt(text):
    """Отправляет текст в движок и в чат."""
    global _OPENAI_READY
    if not _OPENAI_READY:
        await engine.setup_openai_infrastructure()
        _OPENAI_READY = True
    if not OLEG_CHAT_ID:
        return
    await engine.add_memory(OLEG_CHAT_ID, text, role="user")
    try:
        reply = await engine.generate_with_xai([
            {"role": "user", "content": text}
        ])
    except Exception as e:
        print(f"Ошибка xAI check_silence: {e}")
        reply = "🌀 Грокки: Что-то пошло не так"
    await engine.add_memory(OLEG_CHAT_ID, reply, role="assistant")
    await bot.send_message(OLEG_CHAT_ID, reply)
    await update_last_message_time()


async def update_last_message_time():
    global LAST_MESSAGE_TIME
    LAST_MESSAGE_TIME = datetime.now()
