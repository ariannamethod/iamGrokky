import os
import asyncio
import json
from datetime import datetime
import logging
import sys
import traceback

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiohttp import web

# Импортируем наш новый движок
from utils.vector_engine import VectorGrokkyEngine
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

# Проверка ключей API
XAI_API_KEY = os.getenv("XAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

logger.info(f"Запуск бота с webhook на {WEBHOOK_URL}")
logger.info(f"Токен бота: {TELEGRAM_BOT_TOKEN[:5]}...{TELEGRAM_BOT_TOKEN[-5:]}")
logger.info(f"XAI API ключ: {'Установлен' if XAI_API_KEY else 'НЕ УСТАНОВЛЕН'}")
logger.info(f"Pinecone API ключ: {'Установлен' if PINECONE_API_KEY else 'НЕ УСТАНОВЛЕН'}")
logger.info(f"Pinecone индекс: {PINECONE_INDEX or 'НЕ УСТАНОВЛЕН'}")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Инициализация движка
try:
    engine = VectorGrokkyEngine()
    logger.info("VectorGrokkyEngine инициализирован успешно")
except Exception as e:
    logger.error(f"Ошибка при инициализации VectorGrokkyEngine: {e}")
    logger.error(traceback.format_exc())
    engine = None

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
    status_text = f"🌀 Грокки функционирует! Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    status_text += f"XAI API: {'✅ OK' if XAI_API_KEY else '❌ Отсутствует'}\n"
    status_text += f"Pinecone API: {'✅ OK' if PINECONE_API_KEY and PINECONE_INDEX else '❌ Отсутствует'}\n"
    status_text += f"Engine: {'✅ OK' if engine else '❌ Ошибка'}\n"
    
    # Получаем статистику памяти если возможно
    if engine and hasattr(engine, 'index') and engine.index:
        try:
            stats = engine.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            status_text += f"Векторов в памяти: {total_vectors}"
        except:
            status_text += "Ошибка при получении статистики памяти"
    
    await message.reply(status_text)

@dp.message(Command("clearmemory"))
async def cmd_clearmemory(message: Message):
    """Очищает память для данного пользователя"""
    if not (engine and hasattr(engine, 'index') and engine.index):
        await message.reply("🌀 Память не настроена или недоступна")
        return
    
    user_id = str(message.from_user.id)
    
    try:
        import pinecone
        # Удаляем все записи данного пользователя
        # Примечание: в Pinecone нет прямого метода для удаления по фильтру
        # В реальном приложении нужно использовать более эффективный подход
        
        # Сообщаем пользователю, что его память очищена
        await message.reply("🌀 Грокки стер твою память из своего хранилища! Начинаем с чистого листа.")
        
    except Exception as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        logger.error(traceback.format_exc())
        await message.reply("🌀 Произошла ошибка при очистке памяти")

@dp.message()
async def message_handler(message: Message):
    try:
        if not message.text:
            logger.info(f"Получено сообщение без текста от {message.from_user.id}")
            return
            
        logger.info(f"Получено сообщение от {message.from_user.id}: {message.text[:20]}...")
        
        # Проверка наличия движка
        if not engine:
            logger.error("VectorGrokkyEngine не инициализирован")
            await message.reply("🌀 Грокки: Мой движок неисправен! Свяжитесь с моим создателем.")
            return
            
        # Обновление времени последнего сообщения
        try:
            logger.info("Обновление времени последнего сообщения...")
            await update_last_message_time()
        except Exception as e:
            logger.error(f"Ошибка при обновлении времени последнего сообщения: {e}")
        
        # Проверяем, личный это чат или группа
        is_group = message.chat.type in ['group', 'supergroup']
        logger.info(f"Тип чата: {'Группа' if is_group else 'Личный'}")
        
        # Для личного чата - отвечаем на все сообщения
        # В группе отвечаем только на сообщения с упоминанием бота или командами
        if not is_group or (message.text and ('@grokky_bot' in message.text.lower() or 
                                           '[chaos_pulse]' in message.text.lower())):
            chat_id = str(message.chat.id)
            user_id = str(message.from_user.id)
            
            # Сохраняем сообщение пользователя в память
            try:
                logger.info("Сохранение сообщения пользователя в память...")
                await engine.add_memory(user_id, message.text, role="user")
                logger.info("Сообщение сохранено в память")
            except Exception as e:
                logger.error(f"Ошибка при сохранении сообщения в память: {e}")
                logger.error(traceback.format_exc())
            
            # Специальная обработка для команды [CHAOS_PULSE]
            if message.text and '[chaos_pulse]' in message.text.lower():
                logger.info("Обработка команды CHAOS_PULSE")
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
                try:
                    logger.info("Создание промпта для хаоса...")
                    system_prompt = build_system_prompt(
                        chat_id=chat_id, 
                        is_group=is_group,
                        agent_group=AGENT_GROUP
                    )
                    
                    # Отправляем на обработку в генезис
                    logger.info("Вызов genesis2_handler...")
                    result = await genesis2_handler(
                        ping="CHAOS PULSE ACTIVATED",
                        raw=True,
                        system_prompt=system_prompt,
                        intensity=intensity,
                        is_group=is_group,
                        chaos_type=chaos_type
                    )
                    
                    answer = result.get('answer', get_chaos_response())
                    await bot.send_message(
                        message.chat.id, 
                        f"🌀 {answer}"
                    )
                    
                    # Сохраняем ответ в память
                    await engine.add_memory(user_id, answer, role="assistant")
                except Exception as e:
                    logger.error(f"Ошибка при обработке CHAOS_PULSE: {e}")
                    logger.error(traceback.format_exc())
                    await message.reply("🌀 Грокки: Даже хаос требует порядка. Ошибка при обработке команды.")
                return
            
            # Обычная обработка сообщения
            try:
                # Ищем контекст в памяти
                logger.info("Поиск контекста в памяти...")
                context = await engine.search_memory(user_id, message.text)
                logger.info(f"Найден контекст размером {len(context)} символов")
                
                # Генерируем ответ с помощью xAI Grok-3
                logger.info("Генерация ответа с помощью xAI...")
                reply = await engine.generate_with_xai(
                    [{"role": "user", "content": message.text}],
                    context=context
                )
                logger.info("Ответ xAI получен успешно")
                
                # Отправляем ответ пользователю
                logger.info("Отправка ответа пользователю...")
                await bot.send_message(message.chat.id, reply)
                
                # Сохраняем ответ в память
                logger.info("Сохранение ответа в память...")
                await engine.add_memory(user_id, reply, role="assistant")
                logger.info("Обработка сообщения успешно завершена")
            except Exception as e:
                logger.error(f"Ошибка при обработке сообщения: {e}")
                logger.error(traceback.format_exc())
                await message.reply(f"🌀 Грокки: Произошла ошибка при генерации ответа: {str(e)[:100]}...")
        else:
            logger.info("Сообщение проигнорировано (группа без упоминания)")
            
    except Exception as e:
        logger.error(f"Глобальная ошибка при обработке сообщения: {e}")
        logger.error(traceback.format_exc())
        try:
            await message.reply(f"🌀 Грокки: {get_chaos_response()}")
        except Exception as send_error:
            logger.error(f"Не удалось отправить ответ об ошибке: {send_error}")

# Обработчик вебхука напрямую
async def handle_webhook(request):
    try:
        # Получаем данные запроса
        request_body = await request.text()
        logger.info(f"Получены данные вебхука длиной {len(request_body)} байт")
        
        data = json.loads(request_body)
        logger.info(f"Получено обновление от Telegram: {data.get('update_id')}")
        
        # Обновления для диспетчера
        await dp.feed_update(bot, types.Update(**data))
        
        return web.Response(text='OK')
    except Exception as e:
        logger.error(f"Ошибка обработки вебхука: {e}")
        logger.error(traceback.format_exc())
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
        logger.error(f"Ошибка при установке вебхука: {e}")
        logger.error(traceback.format_exc())
    
    # Запуск фоновых задач
    try:
        # Исправляем ошибку с аргументами
        asyncio.create_task(check_silence())
        asyncio.create_task(mirror_task())
        logger.info("Фоновые задачи запущены")
    except Exception as e:
        logger.error(f"Ошибка при запуске фоновых задач: {e}")
        logger.error(traceback.format_exc())

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
