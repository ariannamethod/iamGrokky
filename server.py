import asyncio
import os
import json
import re
import random
from datetime import datetime

from dotenv import load_dotenv
from fix_webhook import check_webhook, fix_webhook

load_dotenv()

try:
    from aiogram import Bot, Dispatcher, types
    from aiogram.utils.chat_action import ChatActionSender
    from aiohttp import web
    from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
except ModuleNotFoundError:  # pragma: no cover - used only for tests
    class Bot:
        def __init__(self, token):
            self.token = token

        async def send_message(self, chat_id, text):
            print(f"Mock send_message to {chat_id}: {text}")

    class Dispatcher:
        def __init__(self):
            pass

        def message(self, *_args, **_kwargs):
            def decorator(handler):
                return handler
            return decorator

    class types:
        class Message:
            def __init__(self, chat=None, from_user=None, text=""):
                self.chat = chat
                self.from_user = from_user
                self.text = text

    class ChatActionSender:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    class SimpleRequestHandler:
        def __init__(self, dispatcher, bot):
            pass

        def register(self, app, path):
            app[path] = True

    def setup_application(app, dp):
        pass
    from types import SimpleNamespace as web

try:
    from utils.hybrid_engine import HybridGrokkyEngine
except ModuleNotFoundError:  # pragma: no cover - used only for tests
    class HybridGrokkyEngine:
        async def add_memory(self, *args, **kwargs):
            pass

        async def search_memory(self, *args, **kwargs):
            return ""

        async def generate_with_xai(self, *_args, **_kwargs):
            return ""

        async def setup_openai_infrastructure(self):
            pass
from utils.genesis2 import genesis2_handler
from utils.prompt import build_system_prompt

BOT_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN")
AGENT_GROUP   = os.getenv("AGENT_GROUP", "-1001234567890")
IS_GROUP      = os.getenv("IS_GROUP", "False").lower() == "true"

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher()
engine = HybridGrokkyEngine()

# Trigger phrases that activate Grokky in group chats
GROKKY_TRIGGERS = [
    "грокки",
    "grokky",
    "grokkki",
    "грок",
    "grok",
    "grokki",
    "groky",
]


def get_user_id_from_message(message: dict) -> str:
    """Extract user id from a Telegram message."""
    user = message.get("from", {})
    user_id = user.get("id")
    if user_id is not None:
        return str(user_id)
    return os.getenv("CHAT_ID", "")

@dp.message(lambda m: m.text and any(t in m.text.lower() for t in GROKKY_TRIGGERS))
async def handle_grокky(m: types.Message):
    chat_id = str(m.chat.id)
    user_id = str(m.from_user.id)
    text = m.text

    # Random delay before responding
    if chat_id == AGENT_GROUP:
        await asyncio.sleep(random.randint(60, 300))
    else:
        await asyncio.sleep(random.randint(10, 30))

    # 30% chance to ignore trivial acknowledgements
    if text.strip().lower() in {"окей", "угу", "да", "ok"} and random.random() < 0.3:
        return

    async with ChatActionSender(bot=bot, chat_id=m.chat.id, action="typing"):
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

    # Возможное последующее сообщение через 30-60 минут
    asyncio.create_task(schedule_followup(chat_id))


async def schedule_followup(chat_id: str):
    await asyncio.sleep(random.randint(1800, 3600))
    if random.random() < 0.25:
        follow = await genesis2_handler(
            chaos_type="reflection",
            intensity=random.randint(1, 5)
        )
        await bot.send_message(chat_id, f"🌀 Грокки: {follow}")

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
    # 0. Проверяем и при необходимости исправляем webhook
    if not check_webhook():
        print("Attempting automatic webhook fix...")
        fix_webhook()

    # 1. Настройка OpenAI-памяти
    await engine.setup_openai_infrastructure()

    # 2. Запускаем хаос-таск
    asyncio.create_task(chaos_spark())

    # 3. Запускаем Telegram через webhook
    app = web.Application()
    # Telegram should send updates to `/webhook` without the token
    wh_path = "/webhook"
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
