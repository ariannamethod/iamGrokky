import os
import asyncio
import json
from datetime import datetime
import logging
import sys

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiohttp import web

from utils.hybrid_engine import HybridGrokkyEngine
from utils.genesis2 import genesis2_handler
from utils.howru import check_silence, update_last_message_time
from utils.mirror import mirror_task
from utils.prompt import build_system_prompt, get_chaos_response

# Настройки логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("Не задан TELEGRAM_BOT_TOKEN! Завершение работы.")
    sys.exit(1)

# Используем правильный адрес для Railway
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "https://grokky-production.up.railway.app")
WEBHOOK_PATH = "/webhook"  # Просто /webhook без токена
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("PORT", 8080))
CHAT_ID = os.getenv("CHAT_ID")
AGENT_GROUP = os.getenv("AGENT_GROUP")

logger.info(f"Запуск бота с webhook на {WEBHOOK_URL}")
logger.info(f"Токен бота: {TELEGRAM_BOT_TOKEN[:5]}...{TELEGRAM_BOT_TOKEN[-5:]}")

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
        if not message.text:
            logger.info(f"Получено сообщение без текста от {message.from_user.id}")
            return
            
        logger.info(f"Получено сообщение от {message.from_user.id}: {message.text[:20]}...")
        
        # Обновление времени последнего сообщения
        await update_last_message_time()
        
        # Проверяем, личный это чат или группа
        is_group = message.chat.type in ['group', 'supergroup']
        
        # Для личного чата - отвечаем на все сообщения
        # В группе отвечаем только на сообщения с упоминанием бота или командами
        if not is_group or (message.text and ('@grokky_bot' in message.text.lower() or 
                                           '[chaos_pulse]' in message.text.lower())):
            chat_id = str(message.chat.id)
            user_id = str(message.from_user.id)
            
            # Сохраняем сообщение пользователя в память
            await engine.add_memory(user_id, message.text, role="user")
            
            # Специальная обработка для команды [CHAOS_PULSE]
            if message.text and '[chaos_pulse]' in message.text.lower():
                intensity = 5  # Значение по умолчанию
                chaos_type = None
                
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
                    is_group=is_group,
                    chaos_type=chaos_type
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
        logger.error(f"Ошибка при обработке сообщения: {e}", exc_info=True)
        try:
            await message.reply(f"🌀 Грокки: {get_chaos_response()}")
        except Exception as send_error:
            logger.error(f"Не удалось отправить ответ об ошибке: {send_error}")

# Обработчик вебхука напрямую
async def handle_webhook(request):
    try:
        # Получаем данные запроса
        data = await request.json()
        logger.info(f"Получено обновление от Telegram: {data.get('update_id')}")
        
        # Обновления для диспетчера
        await dp.feed_update(bot, types.Update(**data))
        
        return web.Response(text='OK')
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}", exc_info=True)
        return web.Response(status=500)

# Запуск сервера
async def on_startup(app):
    # Установка вебхука
    try:
        await bot.delete_webhook(drop_pending_updates=True)  # Сначала удалим старый вебхук
        await asyncio.sleep(1)  # Даем время на обработку
        await bot.set_webhook(url=WEBHOOK_URL)
        logger.info(f"Установлен вебхук на {WEBHOOK_URL}")
    except Exception as e:
        logger.error(f"Ошибка при установке вебхука: {e}", exc_info=True)
    
    # Запуск фоновых задач
    try:
        asyncio.create_task(check_silence(bot=bot))
        asyncio.create_task(mirror_task(bot=bot))
        logger.info("Фоновые задачи запущены")
    except Exception as e:
        logger.error(f"Ошибка при запуске фоновых задач: {e}", exc_info=True)

async def on_shutdown(app):
    await bot.delete_webhook()
    logger.info("Удален вебхук")

# Создание и запуск приложения
app = web.Application()

# Регистрация маршрутов
app.router.add_post(WEBHOOK_PATH, handle_webhook)
app.router.add_get("/healthz", lambda request: web.Response(text="OK"))
app.router.add_get("/", lambda request: web.Response(text="Грокки жив и работает!"))

# Хуки запуска и остановки
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Запуск сервера
if __name__ == "__main__":
    logger.info(f"Запуск сервера на {WEBAPP_HOST}:{WEBAPP_PORT}")
    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)
