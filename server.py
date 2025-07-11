import os
import asyncio
import json
from datetime import datetime
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from aiohttp import web

from utils.hybrid_engine import HybridGrokkyEngine
from utils.genesis2 import genesis2_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.prompt import build_system_prompt, get_chaos_response

# Настройки логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "https://grokky.ariannamethod.me")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("PORT", 8000))
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()
engine = HybridGrokkyEngine()

# Обработка голосовых сообщений
VOICE_ENABLED = {}

@dp.message(Command("voiceon"))
async def cmd_voiceon(message: Message):
    VOICE_ENABLED[message.chat.id] = True
    await message.reply("🌀 Грокки включил обработку голоса!")

@dp.message(Command("voiceoff")) 
async def cmd_voiceoff(message: Message):
    VOICE_ENABLED[message.chat.id] = False
    await message.reply("🌀 Грокки выключил обработку голоса!")

@dp.message(Command("status"))
async def cmd_status(message: Message):
    await message.reply(f"🌀 Грокки функционирует! Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@dp.message()
async def message_handler(message: Message):
    try:
        # Обновление времени последнего сообщения
        await update_last_message_time()
        
        # Проверяем, личный это чат или группа
        is_group = message.chat.type in ['group', 'supergroup']
        
        # Для личного чата - отвечаем на все сообщения
        # В группе отвечаем только на сообщения с упоминанием бота или командами
        if not is_group or message.text and ('@grokky_bot' in message.text.lower() or 
                                           '[chaos_pulse]' in message.text.lower()):
            chat_id = str(message.chat.id)
            user_id = str(message.from_user.id)
            
            # Сохраняем сообщение пользователя в память
            await engine.add_memory(user_id, message.text, role="user")
            
            # Специальная обработка для команды [CHAOS_PULSE]
            if message.text and '[chaos_pulse]' in message.text.lower():
                intensity = 5  # Значение по умолчанию
                
                # Извлечение параметров, если они указаны
                parts = message.text.lower().split()
                for part in parts:
                    if part.startswith('type='):
                        chaos_type = part.split('=')[1]
                    if part.startswith('intensity='):
                        try:
                            intensity = int(part.split('=')[1])
                        except ValueError:
                            pass
                
                # Создаем промпт для генерации хаоса
                system_prompt = build_system_prompt(
                    chat_id=chat_id, 
                    is_group=is_group,
                    agent_group=AGENT_GROUP
                )
                
                # Отправляем на обработку в генезис
                result = await genesis2_handler(
                    ping="CHAOS PULSE ACTIVATED",
                    raw=True,
                    system_prompt=system_prompt,
                    intensity=intensity,
                    is_group=is_group
                )
                
                await bot.send_message(
                    message.chat.id, 
                    f"🌀 {result.get('answer', get_chaos_response())}"
                )
                
                # Сохраняем ответ в память
                await engine.add_memory(user_id, result.get('answer', ''), role="assistant")
                return
            
            # Обычная обработка сообщения
            # Ищем контекст в памяти
            context = await engine.search_memory(user_id, message.text)
            
            # Генерируем ответ с помощью xAI Grok-3
            reply = await engine.generate_with_xai(
                [{"role": "user", "content": message.text}],
                context=context
            )
            
            # Отправляем ответ пользователю
            await bot.send_message(message.chat.id, reply)
            
            # Сохраняем ответ в память
            await engine.add_memory(user_id, reply, role="assistant")
            
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await message.reply(f"🌀 Грокки: {get_chaos_response()}")

# Запуск сервера
async def on_startup(bot: Bot) -> None:
    # Установка вебхука
    await bot.set_webhook(WEBHOOK_URL)
    logger.info(f"Установлен вебхук на {WEBHOOK_URL}")
    
    # Запуск фоновых задач
    asyncio.create_task(check_silence())
    asyncio.create_task(mirror_task())

async def on_shutdown(bot: Bot) -> None:
    await bot.delete_webhook()
    logger.info("Удален вебхук")

# Создание и запуск приложения
app = web.Application()

# Обработчик вебхука
webhook_handler = SimpleRequestHandler(
    dispatcher=dp,
    bot=bot,
    secret_token=TELEGRAM_BOT_TOKEN
)
webhook_handler.register(app, path=WEBHOOK_PATH)

# Путь проверки работоспособности
async def healthz(request):
    return web.Response(text="OK")
app.router.add_get("/healthz", healthz)

# Запуск сервера
if __name__ == "__main__":
    # Регистрация обработчиков
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    
    # Запуск веб-сервера
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)
