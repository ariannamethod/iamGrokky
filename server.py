import asyncio
import os
import json
import re
import random
from datetime import datetime
import httpx

from aiogram import Bot, Dispatcher, types
from aiogram.utils.chat_action import ChatActionSender
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from utils.hybrid_engine import HybridGrokkyEngine
from utils.genesis2 import genesis2_handler
from utils.prompt import build_system_prompt

BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
AGENT_GROUP   = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP      = os.getenv("IS_GROUP", "False").lower() == "true"

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher()
engine = HybridGrokkyEngine()

@dp.message(lambda m: any(t in m.text.lower() for t in ["грокки", "grokky"]))
async def handle_grокky(m: types.Message):
    async with ChatActionSender(bot=bot, chat_id=m.chat.id, action="typing"):
        user_id = str(m.from_user.id)
        text    = m.text

        # 1. Добавляем пользователя и сообщение в память OpenAI
        await engine.add_memory(user_id, text, role="user")

        # 2. Обрабатываем CHAOS_PULSE
        if "[CHAOS_PULSE]" in text:
            match = re.match(r"\[CHAOS_PULSE\]\s*type=(\w+)\s*intensity=(\d+)", text)
            if match:
                ctype, cint = match.groups()
                resp = await genesis2_handler(chaos_type=ctype, intensity=int(cint))
                await engine.add_memory(user_id, resp, role="assistant")
                return await m.reply(f"🌀 Грокки: {resp}")

        # 3. Поиск в памяти (optional)
        memory_ctx = ""
        if any(w in text.lower() for w in ["референс", "помнишь"]):
            memory_ctx = await engine.search_memory(user_id, text)

        # 4. Генерация ответа через xAI Grok
        grok_msg = {"role": "user", "content": text}
        resp = await engine.generate_with_xai([grok_msg], context=memory_ctx)

        # 5. Сохраняем в память и шлём пользователю
        await engine.add_memory(user_id, resp, role="assistant")
        await m.reply(f"🌀 Грокки: {resp}")

async def chaos_spark():
    while True:
        await asyncio.sleep(random.randint(1800, 3600))
        if IS_GROUP and random.random() < 0.5:
            reply = await genesis2_handler(
                chaos_type=random.choice(["philosophy","provocation","poetry_burst"]),
                intensity=random.randint(3,10)
            )
            await bot.send_message(AGENT_GROUP, f"🌀 Grокки вбрасывает хаос: {reply}")

async def main():
    # 0. Настройка OpenAI-памяти
    await engine.setup_openai_infrastructure()

    # 1. Запускаем хаос-таск
    asyncio.create_task(chaos_spark())

    # 2. Запускаем Telegram через webhook
    app = web.Application()
    wh_path = f"/webhook/{BOT_TOKEN}"
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=wh_path)
    setup_application(app, dp)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080)))
    await site.start()
    print("🚀 Server started")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
